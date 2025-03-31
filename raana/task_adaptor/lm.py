from transformers import PreTrainedTokenizer, PreTrainedModel
import torch as tc
import torch.nn as nn

from .base import TaskAdaptor

class LMAdaptor(TaskAdaptor[str]):
    def __init__(self, data: list[str], tokenizer: PreTrainedTokenizer | None = None):
        super().__init__(data)
        self.tokenizer = tokenizer
        
    def __call__(self, tokenizer: PreTrainedTokenizer):   
        return LMAdaptor(self.data, tokenizer = tokenizer)

    def get_loss(self, model: PreTrainedModel, datapoint: str) -> tc.Tensor:
        input = self.tokenizer(datapoint, return_tensors = "pt")
        input_ids = input.input_ids.to(model.device)
        
        output = model(input_ids, labels = input_ids)
        return output.loss

