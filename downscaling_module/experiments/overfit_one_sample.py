import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))



# overfit_one_sample.py
import torch
from data.datasets import ConvLSTMDataset
from models.convlstm import ConvLSTMModel
from data.transforms import BilinearInterpolation
import torch.nn as nn
import torch.optim as optim

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Load the data
    lr_data = torch.load('serialized_data/dataLR.pt')['data']  # Shape: (S, T, C, H_lr, W_lr)
    hr_data = torch.load('serialized_data/dataHR.pt')['data']  # Shape: (S, T, C, H_hr, W_hr)

    # Define the transform
    transform = BilinearInterpolation()

    # Create the dataset
    dataset = ConvLSTMDataset(lr_data, hr_data, transform=transform)

    # Get a single sample
    lr_sample, hr_sample = dataset[0]  # Shapes: (T, C, H, W)

    # Optionally, you can trim the sequence length for faster testing
    # For example, use only the first 10 time steps
    seq_len = 50
    lr_sample = lr_sample[300:300+seq_len]
    hr_sample = hr_sample[300:300+seq_len]

    # Add batch dimension
    lr_sample = lr_sample.unsqueeze(0)  # Shape: (1, T, C, H, W)
    hr_sample = hr_sample.unsqueeze(0)  # Shape: (1, T, C, H, W)

    # Move tensors to device
    lr_sample = lr_sample.to(device)
    hr_sample = hr_sample.to(device)

    print("Shapes of the sample:", lr_sample.shape, hr_sample.shape)

    # Model parameters
    input_dim = lr_sample.shape[2]    # Number of input channels C
    output_dim = hr_sample.shape[2]   # Number of output channels
    hidden_dim = [16, 32]             # Hidden dimensions for each layer
    kernel_size = (3, 3)              # Kernel size for each layer
    num_layers = len(hidden_dim)
    num_epochs = 2000                  # Increase the number of epochs to ensure overfitting
    learning_rate = 0.001

    # Initialize the model
    model = ConvLSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        num_layers=num_layers,
        output_dim=output_dim
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        outputs = model(lr_sample)  # Shape: (1, T, output_dim, H, W)

        # Compute loss
        loss = criterion(outputs, hr_sample)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

    # Save the trained model
    torch.save(model, 'convlstm_overfit_model.pth')

if __name__ == '__main__':
    main()