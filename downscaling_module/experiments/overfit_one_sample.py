import sys
sys.path.append('../../downscaling_module')

# overfit_one_sample.py
import torch
from data.datasets import ConvLSTMDataset
from models.convlstm_commented import ConvLSTM
import torch.nn as nn
import torch.optim as optim
from data.datasets import transforms

from icecream import ic

from pytorch_msssim import SSIM

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Load the data
    lr_data = torch.load('serialized_data/dataLR.pt')['data']  # Shape: (S, T, C, H_lr, W_lr)
    hr_data = torch.load('serialized_data/dataHR.pt')['data']  # Shape: (S, T, C, H_hr, W_hr)

    # Define the transform
    transform = transforms

    # Create the dataset
    dataset = ConvLSTMDataset(lr_data, hr_data, transform=transform)

    # Get a single sample
    lr_sample, hr_sample = dataset[49]  # Shapes: (T, C, H, W)

    # Add batch dimension
    lr_sample = lr_sample.unsqueeze(0)  # Shape: (1, T, C, H, W)
    hr_sample = hr_sample.unsqueeze(0)  # Shape: (1, T, C, H, W)

    # #DEBUG -------------------------
    # lr_sample = torch.ones((1, 10, 2, 5, 250)) / (-2)
    # hr_sample = torch.ones((1, 10, 2, 5, 250)) / (-2)

    ic(torch.min(lr_sample), torch.max(lr_sample))
    ic(torch.min(hr_sample), torch.max(hr_sample))

    # Move tensors to device
    lr_sample = lr_sample.to(device)
    hr_sample = hr_sample.to(device)

    # #DEBUG -------------------------
    # lr_sample = lr_sample[:, :, 0, :, :]
    # hr_sample = hr_sample[:, :, 0, :, :]
    # lr_sample.unsqueeze_(2)
    # hr_sample.unsqueeze_(2)

    print("Shapes of the sample:", lr_sample.shape, hr_sample.shape)

    # Model parameters
    input_dim = lr_sample.shape[2]    # Number of input channels C
    # output_dim = hr_sample.shape[2]   # Number of output channels
    hidden_dim = [2]             # Hidden dimensions for each layer
    kernel_size = (3, 3)              # Kernel size for each layer
    num_layers = 1   # Number of layers
    num_epochs = 3000                  # Increase the number of epochs to ensure overfitting
    learning_rate = 0.001

    print("Input dimension:", input_dim, "Hidden dimensions:", hidden_dim, "Kernel size:", kernel_size, "Number of layers:", num_layers)

    # Initialize the model
    model = ConvLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        num_layers=num_layers,
        batch_first=True,
        bias=True,
        return_all_layers=False        
    ).to(device)

    # Loss and optimizer
    # criterion = nn.MSELoss()

    criterion_SSIM = SSIM(win_size=5, data_range=1.0, channel=2, size_average=True)
    criterion_MSE = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        outputs = model(lr_sample)[0][0] # Shape: (1, T, output_dim, H, W)

        # ic(type(outputs))
        # ic(len(outputs[0]))
        # ic(len(outputs[0][0]))
        # ic(len(outputs[1]))
        # ic(outputs[0][0].shape)
        # ic(len(outputs[1][0]))
        # ic(outputs[1][0][0].shape)

        # Compute loss
        # Compute loss
        batch_size, T, C, H, W = outputs.shape

        # Reshape outputs and hr_sample to collapse the time dimension into the batch dimension
        outputs_reshaped = outputs.view(batch_size * T, C, H, W)
        hr_sample_reshaped = hr_sample.view(batch_size * T, C, H, W)

        # Compute SSIM loss
        loss = (1 - criterion_SSIM(outputs_reshaped, hr_sample_reshaped)) + criterion_MSE(outputs_reshaped, hr_sample_reshaped)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

    # Save the trained model
    torch.save(model.state_dict(), 'convlstm_overfit_model.pth')

if __name__ == '__main__':
    main()