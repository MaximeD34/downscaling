import torch
import torch.nn.functional as F
from torchvision.transforms import Compose


def cutSequence(sample):
    """
    Cut the sequence length of the 'lr' and 'hr' tensors in the sample to remove the 100 first time steps.
    Args:
        sample (dict): Dictionary containing 'lr' and 'hr' tensors.
    Returns:
        dict: Updated sample with trimmed tensors.
    """
    print("Transforming with cutSequence")

    sample['lr'] = sample['lr'][:, 100:]
    sample['hr'] = sample['hr'][:, 100:]

    return sample

def bilinearInterpolation(sample):
    """
    Apply bilinear interpolation to match the low-resolution sample to the high-resolution dimensions.
    Args:
        sample (dict): Dictionary containing 'lr' (low-resolution tensor) and 'hr' (high-resolution tensor).
    Returns:
        dict: Updated sample with the resized 'lr' tensor.
    """

    print("Transforming with bilinearInterpolation")

    lr_sample, hr_sample = sample['lr'], sample['hr']

    # Print shapes for debugging
    print("Before interpolation:")
    print("LR:", lr_sample.shape, "HR:", hr_sample.shape)


    # Get the target spatial dimensions from hr_sample
    _, _, _, H_hr, W_hr = hr_sample.shape

    # Reshape the LR sample to combine S and T into the batch dimension
    N = lr_sample.shape[0] * lr_sample.shape[1]  # Flatten S and T
    lr_sample_reshaped = lr_sample.reshape(N, lr_sample.shape[2], lr_sample.shape[3], lr_sample.shape[4])

    print("Reshaped LR:", lr_sample_reshaped.shape)
    print("HR:", hr_sample.shape)

    # Perform bilinear interpolation on lr_sample to match hr_sample dimensions
    lr_resized = F.interpolate(
        lr_sample_reshaped,
        size=(H_hr, W_hr),
        mode='bilinear',
        align_corners=False
    )

    lr_resized = lr_resized.reshape(lr_sample.shape[0], lr_sample.shape[1], lr_sample.shape[2], H_hr, W_hr)

    print("After interpolation:")
    print("LR resized:", lr_resized.shape, "HR:", hr_sample.shape)

    sample['lr'] = lr_resized

    return sample

def min_max_scale(sample, epsilon=1e-8):
    """
    Normalize the 'lr' and 'hr' tensors in the sample to the range [-1, 1].
    Args:
        sample (dict): Dictionary containing 'lr' and 'hr' tensors.
    Returns:
        dict: Updated sample with normalized tensors.
    """

    print("Transforming with min_max_scale")

    for key in ['lr', 'hr']:
        tensor = sample[key]

        # Compute the minimum and maximum values across all dimensions except the channel dimension
        min_vals = torch.amin(tensor, dim=(0, 1, 3, 4), keepdim=True)
        max_vals = torch.amax(tensor, dim=(0, 1, 3, 4), keepdim=True)

        sample[key] = 2 * (tensor - min_vals) / (max_vals - min_vals + epsilon) - 1

    return sample

transforms = Compose([
    cutSequence,
    bilinearInterpolation,
    min_max_scale
])