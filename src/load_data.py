# ---------------------------------------------------------
# Author: Dr Pantelis Georgiades
#         Computation-based Science and Technology Resarch
#         Centre (CaSToRC) - The Cyprus Institute
# License: MIT
# ---------------------------------------------------------

import argparse

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ---------------------------------------------------------

from .utils import load_configs

# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", 
                    help="Directory where the iamge dataset is stored")
parser.add_argument("--config",
                    help="Json configuration file.")
args = parser.parse_args()

# ---------------------------------------------------------
def load_dataset(datadir, batch_size=32, shuffle=True, config=None):
    """
    Loads the images from a folder using pytorch
    
    :param datadir: Directory where images are stored
    :param batch_size: batch size for pytorch training
    :param shuffle: Boolean, whether to shuffle training data or not

    :return dataloader: torch DataLoader object
    """
    # If the user enters the path to the config load it else raise error
    if isinstance(config, str):
        config = load_configs(config)
    elif config is None:
        raise ValueError("ValueError: No config file found.")
    # If there is an integer seed defined in the config set it for torch
    if isinstance(config['torch']['seed'], int):
        print(f"Setting torch seed to {config['torch']['seed']}")
        torch.manual_seed(config['torch']['seed'])
    # Trainsformation for training the neural network
    transform = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(240),
                                 transforms.ToTensor()])
    # Use the ImageFolder function to load the image data
    dataset = datasets.ImageFolder(datadir, transform=transform)
    # Pass the ImageFolder to the DataLoader method. This will return batches of images
    # and their labels.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# ---------------------------------------------------------
def im_view(dataset, num_view=1, save=False, path_save=None):
    """
    View a selected number of imaged from a batch along with their labels
    
    :param dataset: Pytorch DataLoader object (from load_dataset())
    :param num_view: int, Indicates how many images to plot (must be <= batch_size)
    :param save: Boolean, whether to save the images or not
    :param path_save: Path to save the image to
    """
    # Get a set of images from the dataloader
    imgs, labels = next(iter(dataset))
    # Plot
    fig, axes = plt.subplots(ncols=num_view, nrows=1, sharex=True, sharey=True,
                             figsize=(5*num_view if 5*num_view < 20 else 20, 5))
    if num_view == 1:
        axes.imshow(imgs[0].permute(1, 2, 0))
        axes.set_title(f"label: {labels[0]}")
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['left'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.tick_params(axis='both', length=0)
        axes.set_xticklabels('')
        axes.set_yticklabels('')
        if save:
            plt.savefig(path_save)
        plt.show()
    elif num_view > 0:
        for i, ax in enumerate(axes):
            ax.imshow(imgs[i].permute(1, 2, 0))
            ax.set_title(f"label: {labels[i]}")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='both', length=0)
            ax.set_xticklabels('')
            ax.set_yticklabels('')
        if save:
            plt.savefig(path_save)
        plt.show()
    elif num_view < 0:
        raise ValueError("ValueError: num_view neews to be a positive integer.")
    else:
        raise ValueError("ValueError: num_view neews to be a positive integer.")

# ---------------------------------------------------------
def load_augment_data(datadir, batch_size=32, shuffle=True, config=None):
    """
    Augments the training data according to a set of input parameters

    :param datadir: Directory where images are stored
    :param batch_size: batch size for pytorch training
    :param shuffle: Boolean, whether to shuffle training data or not
    :param config: Json configuration file
    """
    # Config
    # If the user enters the path to the config load it else raise error
    if isinstance(config, str):
        config = load_configs(config)
    elif config is None:
        raise ValueError("ValueError: No config file found.")
    # If there is an integer seed defined in the config set it for torch
    if isinstance(config['torch']['seed'], int):
        print(f"Setting torch seed to {config['torch']['seed']}")
        torch.manual_seed(config['torch']['seed'])
    # Trainsformation for training the neural network
    transform = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(240),
                                 transforms.RandomHorizontalFlip(config['dataset']['RandomHorizontalFlip']),
                                 transforms.RandomVerticalFlip(config['dataset']['RandomVerticalFlip']),
                                 transforms.RandomHorizontalFlip(config['dataset']['RandomHorizontalFlip']),
                                 transforms.RandomRotation(config['dataset']['RandomRotation']),
                                 transforms.RandomHorizontalFlip(config['dataset']['RandomHorizontalFlip']),
                                 transforms.RandomInvert(config['dataset']['RandomInvert']),
                                 transforms.ToTensor()
                                 ])
    # Use the ImageFolder function to load the image data
    dataset = datasets.ImageFolder(datadir, transform=transform)
    # Pass the ImageFolder to the DataLoader method. This will return batches of images
    # and their labels.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
