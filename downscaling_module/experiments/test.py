# Add parent directory to path for imports
import sys
import os
sys.path.append('/home/maxime/DL-ML/downscalling/experimentations/convlstm')
sys.path.append('../data')
sys.path.append('../models')
sys.path.append('../data/transforms')
sys.path.append('../../')
sys.path.append('/home/maxime/DL-ML/downscalling/experimentations/')

import torch
from models.convlstm import ConvLSTMModel  
from data.datasets import ConvLSTMDataset

from data.transforms import BilinearInterpolation

# Load data and create dataset
lr_tensor = torch.load('/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataLR.pt')['data']
hr_tensor = torch.load('/home/maxime/DL-ML/downscalling/experimentations/serialized_data/dataHR.pt')['data']
dataset = ConvLSTMDataset(lr_tensor, hr_tensor, transform=BilinearInterpolation())

print(lr_tensor.shape, hr_tensor.shape)

# Get a sample for testing
lr, hr = dataset[0]

print("Shapes of test the sample:", lr.shape, hr.shape)

# Define model architecture matching training parameters
input_dim = lr.shape[1]    # Number of channels from data
output_dim = hr.shape[1]   # Number of channels from data
hidden_dim = [16, 32]      # Same as training
kernel_size = (3, 3)       # Same as training
num_layers = len(hidden_dim)

# Instantiate the model
model = ConvLSTMModel(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    kernel_size=kernel_size,
    num_layers=num_layers,
    output_dim=output_dim
)

# Load model and move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('/home/maxime/DL-ML/downscalling/experimentations/convlstm_overfit_model.pth')
model.to(device)
model.eval()

# Make predictions
lr = lr.to(device)
with torch.no_grad():
    pred = model(lr.unsqueeze(0))
pred = pred.squeeze(0)

# Save the prediction
torch.save({"data": pred}, '/home/maxime/DL-ML/downscalling/experimentations/serialized_data/prediction_1.pt')