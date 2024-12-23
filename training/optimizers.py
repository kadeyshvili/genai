from utils.class_registry import ClassRegistry
from torch.optim import Adam


optimizers_registry = ClassRegistry()


@optimizers_registry.add_to_registry(name="adam")
class Adam_(Adam):
    def __init__(self, params, config):
        lr = config["lr"]
        weight_decay = config["weight_decay"]
        super().__init__(params, lr=lr, weight_decay=weight_decay)