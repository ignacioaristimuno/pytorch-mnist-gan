import random
import os
from yaml import safe_load
import torch
from torch import optim
from torch import nn
import torch.backends.cudnn as cudnn

from datasets.mnist_dataset import get_dataloader
from models.dcgan.generator import DCGANGenerator, get_generator_defaults
from models.dcgan.discriminator import DCGANDiscriminator, get_discriminator_defaults
from training import train


# Set random seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)

cudnn.benchmark = True


# Settings
with open('config.yaml', 'r') as file:
    settings = safe_load(file)


NOISE_DIM = settings['Generator']['NoiseDim']

BATCH_SIZE = settings['Dataset']['BatchSize']
N_WORKERS = settings['Dataset']['Workers']

N_EPOCHS = settings['Training']['N_EPOCHS']
GENERATOR_LR = settings['Training']['GeneratorLR']
DISCRIMINATOR_LR = settings['Training']['DiscriminatorLR']
ADAM_BETA_1 = settings['Training']['AdamBeta1']


# Device
# DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Current device: {DEVICE.upper()}")

DEVICE = torch.device(DEVICE)


# DataLoader
dataloader = get_dataloader(BATCH_SIZE, N_WORKERS)


# Weights initialization
def weights_initialization(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


# Create Generator
generator = DCGANGenerator(**get_generator_defaults()).to(DEVICE)
generator.apply(weights_initialization)

print(generator)


# Create Discriminator
discriminator = DCGANDiscriminator(**get_discriminator_defaults()).to(DEVICE)
discriminator.apply(weights_initialization)

print(discriminator)


# Loss function
criterion = nn.BCELoss()

# Create batch of latent vectors to keep track of training progress of the generator
fixed_noise = torch.randn(64, NOISE_DIM, 1, 1, device=DEVICE)

# Labels' convention
real_label = 1
fake_label = 0

# Optimizers
optimizerG = optim.Adam(generator.parameters(), lr=GENERATOR_LR, betas=(ADAM_BETA_1, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=DISCRIMINATOR_LR, betas=(ADAM_BETA_1, 0.999))


# Folders for saving results
RESULTS_FOLDER = 'results/MNIST_DCGAN'

if not os.path.isdir(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
if not os.path.isdir(f'{RESULTS_FOLDER}/random_results'):
    os.makedirs(f'{RESULTS_FOLDER}/random_results', exist_ok=True)
if not os.path.isdir(f'{RESULTS_FOLDER}/fixed_results'):
    os.makedirs(f'{RESULTS_FOLDER}/fixed_results', exist_ok=True)


# Training
train(dataloader, generator, discriminator, DEVICE, optimizerG, optimizerD, criterion, NOISE_DIM, N_EPOCHS, None)



# fake_images = torch.randn((64, 100), device=DEVICE).view(-1, 100, 1, 1).detach()
# G_result = generator(fake_images)
# G_result = G_result.detach().numpy()
# plt.imshow(G_result[13].squeeze())
# plt.show()
