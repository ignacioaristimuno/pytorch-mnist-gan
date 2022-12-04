import random
import os
from yaml import safe_load
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
import torch.backends.cudnn as cudnn

from datasets.mnist_dataset import get_mnist_dataloader
from datasets.cifar10_dataset import get_cifar_dataloader
from models.dcgan.generator import DCGANGenerator, get_dcgan_generator_defaults
from models.dcgan.discriminator import (
    DCGANDiscriminator,
    get_dcgan_discriminator_defaults,
)
from models.vanilla.generator import VanillaGenerator, get_vanilla_generator_defaults
from models.vanilla.discriminator import (
    VanillaDiscriminator,
    get_vanilla_discriminator_defaults,
)
from models.cifar10.generator import CIFARGenerator, get_cifar_generator_defaults
from models.cifar10.discriminator import (
    CIFARDiscriminator,
    get_cifar_discriminator_defaults,
)
from training import train_gan
from models.available_models import Task, GANModel


# Select which GAN to train
model_settings = GANModel(model=Task.CIFAR_MNIST.value)


# Set random seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)

cudnn.benchmark = True


# Settings
with open("config.yaml", "r") as file:
    settings = safe_load(file)


NOISE_DIM = settings["Generator"]["NoiseDim"]

BATCH_SIZE = settings["Dataset"]["BatchSize"]
N_WORKERS = settings["Dataset"]["Workers"]

N_EPOCHS = settings["Training"][model_settings.configs]["N_EPOCHS"]
GENERATOR_LR = settings["Training"][model_settings.configs]["GeneratorLR"]
DISCRIMINATOR_LR = settings["Training"][model_settings.configs]["DiscriminatorLR"]
ADAM_BETA_1 = settings["Training"][model_settings.configs]["AdamBeta1"]


# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Current device: {DEVICE.upper()}")

DEVICE = torch.device(DEVICE)


# DataLoader
if model_settings.configs == "CIFAR":
    dataloader = get_cifar_dataloader(BATCH_SIZE, N_WORKERS)
else:
    dataloader = get_mnist_dataloader(BATCH_SIZE, N_WORKERS)


# Weights initialization
def weights_initialization(model):
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


# Create Generator
if model_settings.model == "VANILLA_MNIST":
    generator = VanillaGenerator(**get_vanilla_generator_defaults()).to(DEVICE)
elif model_settings.model == "DCGAN_MNIST":
    generator = DCGANGenerator(**get_dcgan_generator_defaults()).to(DEVICE)
    generator.apply(weights_initialization)
elif model_settings.model == "CIFAR_MNIST":
    generator = CIFARGenerator(**get_cifar_generator_defaults()).to(DEVICE)

print(generator)


# Create Discriminator
if model_settings.model == "VANILLA_MNIST":
    discriminator = VanillaDiscriminator(**get_vanilla_discriminator_defaults()).to(
        DEVICE
    )
elif model_settings.model == "DCGAN_MNIST":
    discriminator = DCGANDiscriminator(**get_dcgan_discriminator_defaults()).to(DEVICE)
    discriminator.apply(weights_initialization)
elif model_settings.model == "CIFAR_MNIST":
    discriminator = CIFARDiscriminator(**get_cifar_discriminator_defaults()).to(DEVICE)

print(discriminator)


# Loss function
criterion = nn.BCELoss()

# Create batch of latent vectors to keep track of training progress of the generator
fixed_noise = torch.randn(64, NOISE_DIM, 1, 1, device=DEVICE)

# Labels' convention
real_label = 1
fake_label = 0

# Optimizers
optimizerG = optim.Adam(
    generator.parameters(), lr=GENERATOR_LR, betas=(ADAM_BETA_1, 0.999)
)
optimizerD = optim.Adam(
    discriminator.parameters(), lr=DISCRIMINATOR_LR, betas=(ADAM_BETA_1, 0.999)
)


# Folders for saving results
RESULTS_FOLDER = f"results/{model_settings.model}"

if not os.path.isdir(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
if not os.path.isdir(f"{RESULTS_FOLDER}/random_results"):
    os.makedirs(f"{RESULTS_FOLDER}/random_results", exist_ok=True)
if not os.path.isdir(f"{RESULTS_FOLDER}/fixed_results"):
    os.makedirs(f"{RESULTS_FOLDER}/fixed_results", exist_ok=True)


# Training
train_gan(
    dataloader,
    generator,
    discriminator,
    DEVICE,
    optimizerG,
    optimizerD,
    criterion,
    NOISE_DIM,
    N_EPOCHS,
    None,
    RESULTS_FOLDER,
)


# Testing on fake images
fake_images = torch.randn((25, 100), device=DEVICE).view(-1, 100, 1, 1).detach()

G_result = generator(fake_images)
G_result = G_result.cpu()
G_result = G_result.detach().numpy()
G_result.shape

# setting values to rows and column variables
fig = plt.figure(figsize=(14, 14))
rows = 5
columns = 5

# Plot results
for k, array in enumerate(G_result):
    ax = plt.subplot(rows, columns, k + 1)
    im = ax.imshow(array.reshape(28, 28), cmap="gray")

    plt.tight_layout()
plt.show()
