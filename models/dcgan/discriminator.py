from yaml import safe_load
import torch.nn as nn


def get_discriminator_defaults():
    """Function for getting the default values for the Discriminator's init"""

    with open('config.yaml', 'r') as file:
        settings = safe_load(file)
    
    discriminator_settings = {
        'disc_filters': settings['Discriminator']['DiscriminatorFilters'],
        'n_channels': settings['Dataset']['Channels'],
        'n_gpus': settings['Training']['GPUs']
    }
    return discriminator_settings


class DCGANDiscriminator(nn.Module):
    """Discriminator class for classifying between real and fake MNIST digit images"""

    def __init__(self, n_channels: int, disc_filters: int, n_gpus: int):
        super(DCGANDiscriminator, self).__init__()
        self.ngpu = n_gpus
        self.main = nn.Sequential(
            # Converts input image (28, 28, n_channels) into (15, 15, n_channels)
            nn.Conv2d(n_channels, disc_filters, kernel_size=2, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Converts (15, 15, n_channels) into (8, 8, n_channels*2)
            nn.Conv2d(disc_filters, disc_filters * 2, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(disc_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Converts (8, 8, n_channels*2) into (4, 4, n_channels*4)
            nn.Conv2d(disc_filters * 2, disc_filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(disc_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Converts (4, 4, n_channels*4) into (1, 1, 1) -> Classify real or fake (not MNIST classes)
            nn.Conv2d(disc_filters * 4, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


def H_out_conv_calculator(H: int, kernel: int, stride: int=1, padding: int=0, dilatation: int=1):
    return int(1 + (H + (2 * padding) - (dilatation * (kernel-1)) - 1)/stride)
