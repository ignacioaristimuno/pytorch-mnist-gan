from dataclasses import dataclass
from enum import Enum


configs_dictionary = {
    "VANILLA_MNIST": "VanillaGAN",
    "DCGAN_MNIST": "DCGAN",
    "CIFAR_MNIST": "CIFAR",
}


class Task(Enum):
    """Enum of available tasks for training the GAN model"""

    VANILLA_MNIST = "VANILLA_MNIST"
    DCGAN_MNIST = "DCGAN_MNIST"
    CIFAR_MNIST = "CIFAR_MNIST"


@dataclass
class GANModel:
    """Class for selecting among available models for training the GAN"""

    model: str
    configs: str = ""

    def __post_init__(self):
        self.configs = configs_dictionary[self.model]
