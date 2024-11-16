import numpy as np
import torch

def loadOne(n, parent_path, quality):

    '''
    Returns a tensor of shape (C, row, col) where C is the number of channels, row is the number of rows and col is the number of columns
    '''

    data_h = np.loadtxt(f'{parent_path}/h_evol_{quality}/h_evol_{n}.txt', delimiter=';')
    data_q = np.loadtxt(f'{parent_path}/q_norm_evol_{quality}/q_norm_evol_{n}.txt', delimiter=';')

    #convert both to tensor and combine them for dim (2, dim(data_h))
    data_h = torch.tensor(data_h)
    data_q = torch.tensor(data_q)

    if quality == "LR":
        #add a dimension for the channel
        data_h = data_h.unsqueeze(0)
        data_q = data_q.unsqueeze(0)

    data = torch.stack((data_h, data_q))

    return data

def loadOneSimulation(n, parent_path, quality, simulation_length=601):

    '''
    Returns a tensor of shape (T, C, row, col) where T is the number of time steps, C is the number of channels, row is the number of rows and col is the number of columns
    '''

    shape = loadOne(n*simulation_length, parent_path, quality).shape
    data_all = torch.empty((simulation_length, *shape))

    for i in range(simulation_length):
        data_all[i] = loadOne(n*simulation_length + i, parent_path, quality)

    return data_all

def loadAllSimulation(parent_path, quality, simulation_length=601, simulation_number=14):

    '''
    Returns a tensor of shape (S, T, C, row, col) where S is the number of simulations, T is the number of time steps, C is the number of channels, row is the number of rows and col is the number of columns
    '''

    shape = loadOneSimulation(0, parent_path, quality).shape
    data_all = torch.empty((simulation_number, *shape))

    for i in range(simulation_number):
        data_all[i] = loadOneSimulation(i, parent_path, quality)

    return data_all