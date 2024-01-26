

from typing import Generic, TypeVar, BinaryIO, Dict, Optional, Tuple

import torch
from torch import nn
import pydantic
from torch.optim import Optimizer


C = TypeVar('C', bound=pydantic.BaseModel)


class PersistableModel(nn.Module, Generic[C]):

    def config(self) -> C:
        raise NotImplementedError


M = TypeVar('M', bound=PersistableModel)


class ModelPersister(Generic[M, C]):
    CONFIG_KEY = "__config"
    OPTIMIZER_KEY = "__optimizer"
    
    def instantiate_model(self, config: C) -> M:
        raise NotImplementedError

    def persist_model(self, model: M, io_handle: BinaryIO, optimizer: Optional[Optimizer] = None) -> None:
        state_dict = model.state_dict()
        config = model.config()
        state_dict_with_config = {
            **state_dict,
            **{ModelPersister.CONFIG_KEY: config},
            **({ModelPersister.OPTIMIZER_KEY: optimizer.state_dict()} if optimizer else {})
        }
        torch.save(state_dict_with_config, io_handle)
   
    def restore(self, io_handle: BinaryIO) -> Tuple[M, Optional[Optimizer]]:
        state_dict: Dict = torch.load(io_handle)
        config: C = state_dict.pop(ModelPersister.CONFIG_KEY)
        optimizer = state_dict.pop(ModelPersister.OPTIMIZER_KEY, None)
        model = self.instantiate_model(config)
        model.load_state_dict(state_dict=state_dict)
        return model, optimizer
