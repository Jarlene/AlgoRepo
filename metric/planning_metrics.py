
from typing import Any
import torch
from torchmetrics import Metric


class AverageDisplacementError(Metric):
    """
    Metric representing the displacement L2 error averaged from all poses of a trajectory.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_state("sum_squared_error", default=torch.tensor(
            0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", default=torch.tensor(
            0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.sum_squared_error += torch.norm(
            preds[..., :2] - target[..., :2])
        self.n_observations += preds.numel()

    def compute(self) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """

        return self.sum_squared_error/self.n_observations


class FinalDisplacementError(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("sum_squared_error", default=torch.tensor(
            0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", default=torch.tensor(
            0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.sum_squared_error += torch.norm(
            preds[..., -1, :2] - target[..., -1, :2])
        self.n_observations += preds.shape[0]

    def compute(self) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """

        return self.sum_squared_error/self.n_observations


class AverageHeadingError(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("heading_error", default=torch.tensor(
            0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", default=torch.tensor(
            0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        heading_err = torch.abs(preds[..., 2] - target[..., 2])
        self.heading_error += torch.sum(torch.atan2(torch.sin(heading_err),
                                                    torch.cos(heading_err)))
        self.n_observations += preds.numel()

    def compute(self) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        return self.heading_error/self.n_observations


class FinalHeadingError(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.add_state("heading_error", default=torch.tensor(
            0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", default=torch.tensor(
            0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        heading_err = torch.abs(preds[..., -1, 2] - target[..., -1, 2])
        self.heading_error += torch.sum(torch.atan2(torch.sin(heading_err),
                                                    torch.cos(heading_err)))
        self.n_observations += preds.shape[0]

    def compute(self) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        return self.heading_error/self.n_observations
