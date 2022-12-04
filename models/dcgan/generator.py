from yaml import safe_load
import torch.nn as nn

from logger import custom_logger


def get_dcgan_generator_defaults():
    """Function for getting the default values for the Generator's init"""

    with open("config.yaml", "r") as file:
        settings = safe_load(file)

    generator_settings = {
        "noise_dim": settings["Generator"]["NoiseDim"],
        "gen_filters": settings["Generator"]["GeneratorFilters"],
        "n_channels": settings["Dataset"]["Channels"]["MNIST"],
        "n_gpus": settings["Training"]["GPUs"],
    }
    return generator_settings


class DCGANGenerator(nn.Module):
    """Generator class for creating MNIST digit images from random noise"""

    def __init__(self, noise_dim: int, gen_filters: int, n_channels: int, n_gpus: int):
        super(DCGANGenerator, self).__init__()
        self.logger = custom_logger(self.__class__.__name__)
        self.logger.info(f"Using DCGAN as Generator")
        self.n_gpus = n_gpus
        self.sequential = nn.Sequential(
            # Converts Z (1, 1, 100) into (4, 4, disc_filters*8)
            nn.ConvTranspose2d(
                noise_dim,
                gen_filters * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(gen_filters * 8),
            nn.SELU(True),
            # Converts (4, 4, disc_filters*8) into (8, 8, disc_filters*4)
            nn.ConvTranspose2d(gen_filters * 8, gen_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_filters * 4),
            nn.SELU(True),
            # Converts (8, 8, disc_filters*4) into (16, 16, disc_filters*2)
            nn.ConvTranspose2d(gen_filters * 4, gen_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_filters * 2),
            nn.SELU(True),
            # Converts (16, 16, disc_filters*4) into (32, 32, disc_filters*2)
            nn.ConvTranspose2d(gen_filters * 2, gen_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_filters),
            nn.SELU(True),
            # Converts (32, 32, disc_filters*2) into (28, 28, n_channels)
            nn.ConvTranspose2d(
                gen_filters, n_channels, kernel_size=1, stride=1, padding=2, bias=False
            ),
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
