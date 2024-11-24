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
    
    scaled_tensor = 2 * (tensor - min_vals) / (max_vals - min_vals + epsilon) - 1
    return scaled_tensor

class ConvLSTMDataset(Dataset):
    def __init__(self, lr_tensor, hr_tensor, transform=None):
        """
        Args:
            lr_tensor (torch.Tensor): Low-resolution tensor of shape (S, T, C, 1, 50)
            hr_tensor (torch.Tensor): High-resolution tensor of shape (S, T, C, 5, 250)
        """
        self.transform = transform
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

        sample = {'lr': lr_sample, 'hr': hr_sample}
        if self.transform:
            sample = self.transform(sample)

        lr_sample = sample['lr']
        hr_sample = sample['hr']

        return lr_sample, hr_sample

# from data.transforms import BilinearInterpolation

if __name__ == "__main__":
    #load .pt files
    lr_tensor = torch.load('/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataLR.pt')["data"]
    hr_tensor = torch.load('/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataHR.pt')["data"]

    transform = BilinearInterpolation()
    dataset = ConvLSTMDataset(lr_tensor, hr_tensor, transform=transform)
    # print(f"Number of simulations: {len(dataset)}")
    lr_sample, hr_sample = dataset[0]

    # Save the normalized uni tensors
    torch.save({"data": lr_sample}, '/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataLR_normalized.pt')
    torch.save({"data": hr_sample}, '/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataHR_normalized.pt')

    print("NEW SHAPES 3", lr_sample.shape, hr_sample.shape)

    lr_sample_stack = lr_sample.unsqueeze(0)
    hr_sample_stack = hr_sample.unsqueeze(0)

    for i in range(len(dataset)):
        lr_sample, hr_sample = dataset[i]
        lr_sample_stack = torch.cat((lr_sample_stack, lr_sample.unsqueeze(0)), 0)
        hr_sample_stack = torch.cat((hr_sample_stack, hr_sample.unsqueeze(0)), 0)

    print("NEW SHAPES concatenated", lr_sample_stack.shape, hr_sample_stack.shape)

    # Save the normalized stacked tensors
    torch.save({"data": lr_sample_stack}, '/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataLR_normalized_stacked.pt')
    torch.save({"data": hr_sample_stack}, '/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataHR_normalized_stacked.pt')