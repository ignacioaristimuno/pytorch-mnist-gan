from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_cifar_dataloader(batch_size: int, n_workers: int) -> DataLoader:
    """Function for downloading the CIFAR 10 dataset and loading it into a DataLoader"""

    dataset = datasets.CIFAR10(
        root="./data",
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers
    )
    return dataloader
