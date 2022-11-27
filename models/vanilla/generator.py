from yaml import safe_load
import torch.nn as nn

from logger import custom_logger


def get_vanilla_generator_defaults():
    """Function for getting the default values for the Generator's init"""

    with open("config.yaml", "r") as file:
        settings = safe_load(file)

    generator_settings = {
        "noise_dim": settings["Generator"]["NoiseDim"],
        "n_gpus": settings["Training"]["GPUs"],
    }
    return generator_settings


class VanillaGenerator(nn.Module):
    """Generator class for creating MNIST digit images from random noise"""

    def __init__(self, noise_dim: int, n_gpus: int, output_image_dim: int = 28 * 28):
        super(VanillaGenerator, self).__init__()
        self.logger = custom_logger(self.__class__.__name__)
        self.logger.info(f"Using Vanilla Generator for MNIST")
        self.n_gpus = n_gpus
        self.noise_dim = noise_dim
        self.n_out = output_image_dim
        self.sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.noise_dim, 256),
            nn.SELU(True),
            nn.Linear(256, 512),
            nn.SELU(True),
            nn.Linear(512, 1024),
            nn.SELU(True),
            nn.Linear(1024, self.n_out),
            nn.Tanh(),
        )

    def forward(self, input):
        if input.is_cuda and self.n_gpus > 1:
            output = nn.parallel.data_parallel(
                self.sequential, input, range(self.n_gpus)
            )
        else:
            output = self.sequential(input)
        output = output.view(-1, 1, 28, 28)
        return output
