import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from models.base import Base
from layers.Layers import LayerNorm, Residual, RotaryEmbedding, RelPosBias2d
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def FeedForward(dim, mult=4, dropout=0.):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden, bias=False),
        nn.GELU(),
        LayerNorm(dim_hidden),
        nn.Linear(dim_hidden, dim, bias=False)
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        causal=False,
        dropout=0.,
        norm_context=False,
        rel_pos_bias=False,
        encoded_fmap_size=None
    ):
        super().__init__()
        self.causal = causal
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)

        inner_dim = heads * dim_head
        context_dim = context_dim if context_dim is not None else dim
        self.norm_context = LayerNorm(
            context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, inner_dim, bias=False),
            Rearrange('b n (h d) -> b h n d', h=heads)
        )

        # needed for classifier free guidance for transformers
        # by @crowsonkb, adopted by the paper

        self.null_kv = nn.Parameter(torch.randn(dim_head))

        # one-headed key / value attention, from Shazeer's multi-query paper, adopted by Alphacode and PaLM

        self.to_kv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, dim_head, bias=False)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

        # positional bias

        self.rel_pos_bias = None

        if rel_pos_bias:
            assert encoded_fmap_size is not None
            self.rel_pos_bias = RelPosBias2d(encoded_fmap_size, heads)

    def forward(
        self,
        x,
        context=None,
        context_mask=None
    ):
        batch, device = x.shape[0], x.device

        x = self.norm(x)

        q = self.to_q(x) * self.scale

        context = context if context is not None else x
        context = self.norm_context(context)

        kv = self.to_kv(context)

        null_kv = repeat(self.null_kv, 'd -> b 1 d', b=batch)
        kv = torch.cat((null_kv, kv), dim=1)

        sim = einsum('b h i d, b j d -> b h i j', q, kv)

        if self.rel_pos_bias is not None:
            pos_bias = self.rel_pos_bias(sim)
            sim = sim + pos_bias

        mask_value = -torch.finfo(sim.dtype).max

        if context_mask is not None:
            context_mask = F.pad(context_mask, (1, 0), value=True)
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(
                (i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        out = einsum('b h i j, b j d -> b h i d', attn, kv)

        return self.to_out(out)


class Parti(Base):

     def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        dropout=0.,
        ff_mult=4,
        vae=None,
        vae_image_size=None,
        vae_codebook_size=None,
        t5_name='google/t5-v1_1-base',
        text_embed_dim=None,
        cond_drop_prob=0.25,
        max_text_len=128
    ):
        super().__init__()

        # text conditioning

        text_embed_dim = default(text_embed_dim, get_encoded_dim(t5_name))
        self.encode_texts = partial(t5_encode_text, name=t5_name)
        self.max_text_len = max_text_len

        assert cond_drop_prob > 0.
        # classifier free guidance for transformers - @crowsonkb
        self.cond_drop_prob = cond_drop_prob

        # vae and image handling

        assert exists(vae) ^ exists(vae_codebook_size)
        self.vae = vae

        codebook_size = default(vae_codebook_size, vae.codebook_size)
        image_size = default(vae_image_size, vae.image_size)

        self.start_token = nn.Parameter(torch.randn(dim))
        self.image_token_embed = nn.Embedding(codebook_size, dim)

        self.image_encoded_dim = vae.get_encoded_fmap_size(image_size)

        self.axial_height_pos = nn.Parameter(
            torch.randn(self.image_encoded_dim, dim))
        self.axial_width_pos = nn.Parameter(
            torch.randn(self.image_encoded_dim, dim))

        # projecting to logits

        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, causal=True, encoded_fmap_size=self.image_encoded_dim,
                          rel_pos_bias=True, dim_head=dim_head, heads=heads, dropout=dropout),
                Attention(dim, context_dim=text_embed_dim,
                          dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, mult=ff_mult, dropout=dropout)
            ]))

        self.final_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, codebook_size, bias=False)
        self.to_logits.weight = self.image_token_embed.weight

        # default device

        if exists(vae):
            self.to(next(vae.parameters()).device)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        texts,
        *,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        return_pil_images = False
    ):
        device = next(self.parameters()).device

        text_token_embeds, text_mask = self.encode_texts(texts, output_device = device)

        batch = text_token_embeds.shape[0]

        image_seq_len = self.image_encoded_dim ** 2

        image_tokens = torch.empty((batch, 0), device = device, dtype = torch.long)

        for _ in range(image_seq_len):
            logits = self.forward_with_cond_scale(
                text_token_embeds = text_token_embeds,
                text_mask = text_mask,
                image_token_ids = image_tokens
            )[:, -1]

            filtered_logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            sampled = rearrange(sampled, 'b -> b 1')
            image_tokens = torch.cat((image_tokens, sampled), dim = -1)

        image_tokens = rearrange(image_tokens, 'b (h w) -> b h w', h = self.image_encoded_dim)

        if not exists(self.vae):
            return image_tokens

        with torch.no_grad():
            fmap = self.vae.get_fmap_from_codebook(image_tokens)
            images = self.vae.decode(fmap)

        if not return_pil_images:
            return images

        pil_images = list(map(T.ToPILImage(), images.unbind(dim = 0)))
        return pil_images

    def forward_with_cond_scale(self, *args, cond_scale = 3, **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        texts: List[str] = None,
        text_token_embeds = None,
        text_mask = None,
        images = None,
        image_token_ids = None,
        cond_drop_prob = None,
        return_loss = False
    ):
        assert exists(texts) ^ exists(text_token_embeds)
        assert exists(images) ^ exists(image_token_ids)
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # encoding images

        if not exists(image_token_ids):
            assert exists(self.vae), 'vae must be given if you want to encode the image live'

            with torch.no_grad():
                _, image_token_ids, _ = self.vae.encode(images, return_indices_and_loss = True)

            image_token_ids = rearrange(image_token_ids, 'b ... -> b (...)')

        if return_loss:
            assert image_token_ids.shape[-1] > 1, 'not enough image tokens given to return a loss'
            image_token_ids, labels = image_token_ids[:, :-1], image_token_ids

        image_token_emb = self.image_token_embed(image_token_ids)

        # add axial positional embedding

        axial_pos_emb = rearrange(self.axial_width_pos, 'w d -> 1 w d') + rearrange(self.axial_height_pos, 'h d -> h 1 d')
        axial_pos_emb = rearrange(axial_pos_emb, 'h w d -> (h w) d')

        batch, seq_len, device = *image_token_emb.shape[:2], image_token_emb.device

        image_token_emb = image_token_emb + axial_pos_emb[:seq_len]

        # add start token

        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = batch)
        image_token_emb = torch.cat((start_tokens, image_token_emb), dim = 1)

        # text

        if not exists(text_token_embeds):
            with torch.no_grad():
                text_token_embeds, text_mask = self.encode_texts(texts, output_device = device)

        if not exists(text_mask):
            text_mask = torch.ones(text_token_embeds.shape[:2], dtype = torch.bool)

        # enforce max text len

        text_token_embeds, text_mask = map(lambda t: t[:, :self.max_text_len], (text_token_embeds, text_mask))

        # classifier free guidance conditional dropout

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        # attend

        x = image_token_emb
        x = self.init_norm(x)

        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context = text_token_embeds, context_mask = text_mask) + x
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = 0
        )

        return loss
