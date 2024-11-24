import torch
import torch.nn.functional as F

class BilinearInterpolation:
    def __call__(self, sample):
        lr_sample, hr_sample = sample['lr'], sample['hr']

        print(lr_sample.shape, hr_sample.shape)
        # Get the target spatial dimensions from hr_sample
        _, _, H_hr, W_hr = hr_sample.shape
        # Perform bilinear interpolation on lr_sample to match hr_sample dimensions
        lr_resized = F.interpolate(
            lr_sample,
            size=(H_hr, W_hr),
            mode='bilinear',
            align_corners=False
        )

        print("NEW SHAPES", lr_resized.shape, hr_sample.shape)

        return {'lr': lr_resized, 'hr': hr_sample}