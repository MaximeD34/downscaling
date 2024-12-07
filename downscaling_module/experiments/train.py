import sys

sys.path.append('../../')
sys.path.append('../../downscaling_module')


# train.py
import torch
from torch.utils.data import DataLoader
from data.datasets import ConvLSTMDataset
from models.convlstm_commented import ConvLSTM
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

    # Create the dataloader
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model parameters
    input_dim = lr_data.shape[2]  # Number of input channels C
    output_dim = hr_data.shape[2]  # Number of output channels
    hidden_dim = [16, 32]  # Hidden dimensions for each layer
    kernel_size = (3, 3)  # Kernel size for each layer
    num_layers = len(hidden_dim)
    num_epochs = 10
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
        epoch_loss = 0.0
        for batch_idx, (lr_batch, hr_batch) in enumerate(dataloader):
            # lr_batch and hr_batch shapes: (batch_size, T, C, H, W)
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            # Forward pass
            outputs = model(lr_batch)  # Shape: (batch_size, T, output_dim, H, W)

            # Compute loss
            loss = criterion(outputs, hr_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss/len(dataloader):.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'convlstm_model.pth')

if __name__ == '__main__':
    main()