# system libraries
from __future__ import print_function, division
import os
from pathlib import Path

# utils libraries
from torch.utils.data import Dataset, DataLoader
import numpy as np

# image processing libraries
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from skimage import io, transform
import torch
torch.manual_seed(42) # for reproducibility

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

def show_image(image, mask):
    ''' Shows an image and its mask overlayed '''
    image[mask==255] = 0
    plt.imshow(image)
    plt.pause(0.001)

class SRDataset(Dataset):
    ''' specular reflection dataset to be used in pytorch processing'''

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):

        # if batch 
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # fetch image and mask from dataset path
        img_path = os.path.join(self.img_dir, str(idx).zfill(5) + '.png')
        mask_path = os.path.join(self.mask_dir, str(idx).zfill(5) + '.png')
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        sample = {'image': img, 'mask': mask}

        # apply transformations if any
        if self.transform: 
            img = self.transform(sample['image'])
            mask = self.transform(sample['mask'])
        
        return img, mask

def show_batch(img_dir, mask_dir):
    ''' Shows a batch of 4 images and their masks overlayed '''
    sr_dataset = SRDataset(img_dir, mask_dir)

    fig = plt.figure()

    for i in range(len(sr_dataset)):
        sample = sr_dataset[i]

        print(i, sample[0].shape, sample[1].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_image(sample[0], sample[1])

        if i == 3:
            plt.show()
            break
    plt.pause(1000)

def make_dataset(img_dir, mask_dir):
    prepared_dataset = SRDataset(img_dir, 
                                mask_dir, 
                                transform=transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomHorizontalFlip(p=0.5)]))

    return prepared_dataset
def create_dataloaders(img_dir, mask_dir, batch_size):
    ''' Creates dataloaders for training, validation, and testing '''

    full_dataset = make_dataset(img_dir, mask_dir)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # create test dataset
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    val_size = int(0.25 * train_size)
    train_size -= val_size

    # create train and val datasets
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # create iterators 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__": 
    img_dir = Path(r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_img")
    mask_dir = Path(r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_mask")

    # # visualize batch of images and masks
    # show_batch(img_dir, mask_dir)

    train_loader, val_loader, test_loader = create_dataloaders(img_dir, mask_dir, batch_size=4)
