from yaml import safe_load
import torch.nn as nn

from logger import custom_logger


def get_cifar_discriminator_defaults():
    """Function for getting the default values for the Discriminator's init"""

    with open("config.yaml", "r") as file:
        settings = safe_load(file)

    discriminator_settings = {
        "disc_filters": settings["Discriminator"]["DiscriminatorFilters"],
        "n_channels": settings["Dataset"]["Channels"],
        "n_gpus": settings["Training"]["GPUs"],
    }
    return discriminator_settings


class CIFARDiscriminator(nn.Module):
    """Discriminator class for classifying between real and fake CIFAR images"""

    def __init__(self, n_channels: int, disc_filters: int, n_gpus: int):
        super(CIFARDiscriminator, self).__init__()
        self.logger = custom_logger(self.__class__.__name__)
        self.logger.info(f"Using DCGAN for CIFAR as Discriminator")
        self.ngpu = n_gpus
        self.sequential = nn.Sequential(
            # Converts input image (64, 64, 3) into (32, 32, 3)
            nn.Conv2d(
                n_channels, disc_filters, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # Converts (32, 32, 3) into (16, 16, 3*2)
            nn.Conv2d(
                disc_filters,
                disc_filters * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(disc_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Converts (16, 16, 3*2) into (8, 8, 3*4)
            nn.Conv2d(
                disc_filters * 2,
                disc_filters * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(disc_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Converts (8, 8, 3*4) into (4, 4, 3*8)
            nn.Conv2d(
                disc_filters * 4,
                disc_filters * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(disc_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Converts (4, 4, 3*8) into (1, 1, 1) -> Classify real or fake (not MNIST classes)
            nn.Conv2d(
                disc_filters * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.sequential, input, range(self.ngpu))
        else:
            output = self.sequential(input)
        return output.view(-1, 1).squeeze(1)


def H_out_conv_calculator(
    H: int, kernel: int, stride: int = 1, padding: int = 0, dilatation: int = 1
):
    return int(1 + (H + (2 * padding) - (dilatation * (kernel - 1)) - 1) / stride)
