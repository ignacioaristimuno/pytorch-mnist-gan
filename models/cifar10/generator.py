from yaml import safe_load
import torch.nn as nn

from logger import custom_logger


def get_cifar_generator_defaults():
    """Function for getting the default values for the Generator's init"""

    with open("config.yaml", "r") as file:
        settings = safe_load(file)

    generator_settings = {
        "noise_dim": settings["Generator"]["NoiseDim"],
        "gen_filters": settings["Generator"]["GeneratorFilters"],
        "n_channels": settings["Dataset"]["Channels"]["CIFAR"],
        "n_gpus": settings["Training"]["GPUs"],
    }
    return generator_settings


class CIFARGenerator(nn.Module):
    """Generator class for creating CIFAR images from random noise"""

    def __init__(self, noise_dim: int, gen_filters: int, n_channels: int, n_gpus: int):
        super(CIFARGenerator, self).__init__()
        self.logger = custom_logger(self.__class__.__name__)
        self.logger.info(f"Using DCGAN for CIFAR as Generator")
        self.n_gpus = n_gpus

        self.sequential = nn.Sequential(
            # Conv Params: in_channels, out_channels, kernel_size, stride, padding, bias
            nn.ConvTranspose2d(noise_dim, gen_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_filters * 8),
            nn.SELU(True),
            nn.ConvTranspose2d(gen_filters * 8, gen_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_filters * 4),
            nn.SELU(True),
            nn.ConvTranspose2d(gen_filters * 4, gen_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_filters * 2),
            nn.SELU(True),
            nn.ConvTranspose2d(gen_filters * 2, gen_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_filters),
            nn.SELU(True),
            nn.ConvTranspose2d(gen_filters, n_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if input.is_cuda and self.n_gpus > 1:
            output = nn.parallel.data_parallel(
                self.sequential, input, range(self.n_gpus)
            )
        else:
            output = self.sequential(input)
        return output


def H_out_conv_transpose_calculator(
    H: int,
    kernel: int,
    stride: int = 1,
    padding: int = 0,
    out_padding: int = 0,
    dilatation: int = 1,
):
    return (
        ((H - 1) * stride) - (2 * padding) + (dilatation * kernel - 1) + out_padding + 1
    )
