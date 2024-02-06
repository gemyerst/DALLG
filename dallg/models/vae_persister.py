
from .vae import DiscreteVAE, Config
from ..persistence import ModelPersister


class VaePersister(ModelPersister[DiscreteVAE, Config]):

    def instantiate_model(self, config: Config) -> DiscreteVAE:
        return DiscreteVAE(config)
