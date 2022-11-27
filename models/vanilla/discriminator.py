from yaml import safe_load
import torch.nn as nn

from logger import custom_logger


def get_vanilla_discriminator_defaults():
    """Function for getting the default values for the Generator's init"""

    with open('config.yaml', 'r') as file:
        settings = safe_load(file)
    
    generator_settings = {
        'n_gpus': settings['Training']['GPUs']
    }
    return generator_settings


class VanillaDiscriminator(nn.Module):
    """Generator class for creating MNIST digit images from random noise"""

    def __init__(self, n_gpus: int, input_image_dim: int=28*28):
        super(VanillaDiscriminator, self).__init__()
        self.logger = custom_logger(self.__class__.__name__)
        self.logger.info(f"Using Vanilla Generator")
        self.n_gpus = n_gpus
        self.input_dim = input_image_dim
        self.sequential = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.Dropout(0.2),
            nn.SELU(True),
            
            nn.Linear(1024, 512),
            nn.Dropout(0.15),
            nn.SELU(True),
            
            nn.Linear(512, 128),
            nn.Dropout(0.1),
            nn.SELU(True),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        input = input.view(-1, 784)
        if input.is_cuda and self.n_gpus > 1:
            output = nn.parallel.data_parallel(self.sequential, input, range(self.n_gpus))
        else:
            output = self.sequential(input)
        return output
