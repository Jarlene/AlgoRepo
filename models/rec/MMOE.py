from models.base import Base
from argparse import Namespace


class MMOE(Base):
    def __init__(self, args: Namespace) -> None:
        super(MMOE, self).__init__(**args)
