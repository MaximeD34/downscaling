import sys
sys.path.append('../../downscaling_module')

import torch
from torch.utils.data import Dataset

from data.transforms import transforms


class ConvLSTMDataset(Dataset):
    def __init__(self, lr_tensor, hr_tensor, transform=None):
        """
        Args:
            lr_tensor (torch.Tensor): Low-resolution tensor of shape (S, T, C, 1, 50)
            hr_tensor (torch.Tensor): High-resolution tensor of shape (S, T, C, 5, 250)
        """
        self.transform = transform
        self.lr_tensor = lr_tensor
        self.hr_tensor = hr_tensor

        sample = {'lr': lr_tensor, 'hr': hr_tensor}
        if self.transform:
            sample = self.transform(sample)

        self.lr_tensor = sample['lr']
        self.hr_tensor = sample['hr']


    def __len__(self):
        # Assuming S is the number of simulations
        return self.lr_tensor.shape[0]

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the simulation to retrieve
        Returns:
            tuple: (lr_sample, hr_sample) where lr_sample is the low-resolution input
                   and hr_sample is the high-resolution output. We discard the first 100 time steps because they are empty.
        """
        lr_sample = self.lr_tensor[idx]
        hr_sample = self.hr_tensor[idx]

        return lr_sample, hr_sample

if __name__ == "__main__":
    #load .pt files
    lr_tensor = torch.load('/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataLR.pt')["data"]
    hr_tensor = torch.load('/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataHR.pt')["data"]

    dataset = ConvLSTMDataset(lr_tensor, hr_tensor, transform=transforms)
    
    lr_sample, hr_sample = dataset[0]
    print("Shapes of the sample:", lr_sample.shape, hr_sample.shape)

    # # Save the normalized uni tensors
    # torch.save({"data": lr_sample}, '/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataLR_normalized.pt')
    # torch.save({"data": hr_sample}, '/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataHR_normalized.pt')