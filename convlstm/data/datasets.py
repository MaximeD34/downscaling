import torch
from torch.utils.data import Dataset


def min_max_scale(tensor, epsilon=1e-8):
    """
    Normalize the tensor to the range [-1, 1] independently for each channel.
    Args:
        tensor (torch.Tensor): Input tensor of shape (S, T, C, ...)
    Returns:
        torch.Tensor: Normalized tensor
    """
    # Compute the minimum and maximum values across all dimensions except the channel dimension
    min_vals = torch.amin(tensor, dim=(0, 1, 3, 4), keepdim=True)
    max_vals = torch.amax(tensor, dim=(0, 1, 3, 4), keepdim=True)
    
    print(f"min_vals: {min_vals}, min_vals.shape: {min_vals.shape}")
    print(f"max_vals: {max_vals}, max_vals.shape: {max_vals.shape}")
    
    scaled_tensor = 2 * (tensor - min_vals) / (max_vals - min_vals + epsilon) - 1
    return scaled_tensor

class ConvLSTMDataset(Dataset):
    def __init__(self, lr_tensor, hr_tensor):
        """
        Args:
            lr_tensor (torch.Tensor): Low-resolution tensor of shape (S, T, C, 1, 50)
            hr_tensor (torch.Tensor): High-resolution tensor of shape (S, T, C, 5, 250)
        """
        self.lr_tensor = min_max_scale(lr_tensor)
        self.hr_tensor = min_max_scale(hr_tensor)

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
        lr_sample = self.lr_tensor[idx, 100:]
        hr_sample = self.hr_tensor[idx, 100:]
        return lr_sample, hr_sample

if __name__ == "__main__":
    #load .pt files
    lr_tensor = torch.load('/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataLR.pt')["data"]
    hr_tensor = torch.load('/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataHR.pt')["data"]

    dataset = ConvLSTMDataset(lr_tensor, hr_tensor)
    print(f"Number of simulations: {len(dataset)}")
    lr_sample, hr_sample = dataset[0]

     # Save the normalized tensors
    torch.save({"data": dataset.lr_tensor}, '/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataLR_normalized.pt')
    torch.save({"data": dataset.hr_tensor}, '/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataHR_normalized.pt')
    