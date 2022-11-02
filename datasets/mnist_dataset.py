
from yaml import safe_load
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_dataloader(batch_size: int, n_workers: int) -> DataLoader:
    dataset = datasets.MNIST(
                root='./data', download=True,
                transform=transforms.Compose([
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=n_workers
                )
    return dataloader
