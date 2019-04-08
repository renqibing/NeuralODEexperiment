"""Ingredient for making mnist data loaders."""

import torch
from examples.cifar_10_dataset import MyCifar
from sacred import Ingredient
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10

data_ingredient = Ingredient('dataset')


@data_ingredient.config
def data_config():
    """Config for data source and loading"""
    batch_size = 128
    isgpu = True
    val_split = 0.05
    num_workers = 2


@data_ingredient.capture
def make_dataloaders(batch_size,
                     num_workers,
                     val_split,
                     isgpu,
                     _log):
    # if isinstance(device, list):
    #     device = device[0]
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32,padding=4),
         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )

    # target_transform = transforms.Compose([transforms.ToTensor()])

    # dset = MyCifar("cifar_10", download=False,transform= transform_train)
    # test_dset = MyCifar("cifar_10", download=False, train=False,transform= transform_test)

    dset = CIFAR10("cifar_10", download=False, transform=transform_train)
    test_dset = CIFAR10("cifar_10", download=False, train=False, transform=transform_test)

    _log.info("Loaded dataset on 'cuda'")

    total = len(dset)
    train_num = int(total * (1 - val_split))
    val_num = total - train_num

    _log.info("Split dataset into {%d} train samples and {%d} \
    validation samples"%(train_num,val_num))

    train, val = torch.utils.data.dataset.random_split(dset,
                                                       [train_num, val_num])

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,)


    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True, )

    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False, )

    # next(iter(train_loader))

    return dset, train_loader, val_loader, test_loader
