from typing import Any
from typing import TypeVar, Generic, Iterator
import torch.nn as nn
import torch as tc

DataT = TypeVar("RawDataT")

class TaskAdaptor(Generic[DataT]):
    """this class is used to adapt the raw data to the input data of the model."""

    def __init__(self, data: list[DataT]):
        self.data = data

    def get_loss(self, model: nn.Module, datapoint: DataT) -> tc.Tensor:
        raise NotImplementedError()



