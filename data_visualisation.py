import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_single(data, title=None):

    '''
    Input : a tensor of shape (2, row, col)
    Plots a single image composed of two plots (h and q)
    '''


    fig, axs = plt.subplots(1, 2, figsize=(10, 2))

    data = data.cpu()
    axs[0].imshow(data[0], aspect='auto')
    axs[1].imshow(data[1], aspect='auto')

    #add titles and color bars
    axs[0].set_title('h')
    axs[1].set_title('q')
    fig.colorbar(axs[0].imshow(data[0], aspect='auto'), ax=axs[0])
    fig.colorbar(axs[1].imshow(data[1], aspect='auto'), ax=axs[1])

    if title:
        plt.suptitle(title)

    plt.show()

def plot_two(data1, data2, title=None):

    '''
    Input : 2 tensors of shape (2, row, col)
    Plots two image both composed of two plots (h and q)
    '''
    fig, axs = plt.subplots(1, 2, figsize=(10, 2))

    data1 = data1.cpu()
    data2 = data2.cpu()

    min_h = min(data1[0].min(), data2[0].min())
    max_h = max(data1[0].max(), data2[0].max())

    min_q = min(data1[1].min(), data2[1].min())
    max_q = max(data1[1].max(), data2[1].max())

    axs[0].imshow(data1[0], aspect='auto', vmin=min_h, vmax=max_h)
    axs[1].imshow(data1[1], aspect='auto', vmin=min_q, vmax=max_q)

    # add titles and color bars
    axs[0].set_title('h')
    axs[1].set_title('q')

    fig.colorbar(axs[0].imshow(data1[0], aspect='auto', vmin=min_h, vmax=max_h), ax=axs[0])
    fig.colorbar(axs[1].imshow(data1[1], aspect='auto', vmin=min_q, vmax=max_q), ax=axs[1])

    fig, axs = plt.subplots(1, 2, figsize=(10, 2))

    axs[0].imshow(data2[0], aspect='auto', vmin=min_h, vmax=max_h)
    axs[1].imshow(data2[1], aspect='auto', vmin=min_q, vmax=max_q)

    # add titles and color bars
    axs[0].set_title('h')
    axs[1].set_title('q')

    fig.colorbar(axs[0].imshow(data2[0], aspect='auto', vmin=min_h, vmax=max_h), ax=axs[0])

    fig.colorbar(axs[1].imshow(data2[1], aspect='auto', vmin=min_q, vmax=max_q), ax=axs[1])

    if title:
        plt.suptitle(title)

    plt.show()

def plot_two_sequences(data1, data2, title=None):
    '''
    Input : 2 tensors of shape (T, 2, row, col)
    Plots T*two image both composed of two plots (h and q)
    '''

    T = data1.shape[0]

    minh = min(data1[:, 0].min(), data2[:, 0].min())
    maxh = max(data1[:, 0].max(), data2[:, 0].max())



def plot_batch(data, start, stop, step):
    # data of shape torch.Size([10, 2, 5, 250])
    # plot 10 * 2 heatmaps of shape (5, 250) next to each other

    stop = stop+1

    #assert inputs
    assert start < stop, 'start must be smaller than stop'
    assert stop <= data.shape[0], 'stop must be smaller than the first dimension of data'
    assert step > 0, 'step must be positive'
    assert start >= 0, 'start must be positive'

    min_h = data[start:stop, 0].min()
    max_h = data[start:stop, 0].max()

    min_q = data[start:stop, 1].min()
    max_q = data[start:stop, 1].max()

    fig, axs = plt.subplots((stop-start) // step, 2, figsize=(10, 5))

    for i in range((stop-start) // step):
        axs[i, 0].imshow(data[start + i*step, 0], aspect='auto', vmin=min_h, vmax=max_h)
        axs[i, 1].imshow(data[start + i*step, 1], aspect='auto', vmin=min_q, vmax=max_q)

        # add titles and color bars
        axs[i, 0].set_title(f'h {start + i*step}')
        axs[i, 1].set_title(f'q {start + i*step}')
        fig.colorbar(axs[i, 0].imshow(data[start + i*step, 0], aspect='auto', vmin=min_h, vmax=max_h), ax=axs[i, 0], )
        fig.colorbar(axs[i, 1].imshow(data[start + i*step, 1], aspect='auto',vmin=min_q, vmax=max_q), ax=axs[i, 1])

    plt.show()

def plot_compare_between_simulation(data, start, stop, step):
    # data of shape torch.Size([N, T, 2, 5, 250])
    # plot 14 lines for each simulation composed of two plots (h and q) for each time step

    '''
    Plots the data for each simulation between start and stop time steps with a step size of step

    Parameters:
    data: torch.tensor of shape (N, T, 2, row, col)
    start: int, starting time step
    stop: int, stopping time step
    step: int, step size between time steps

    '''

    stop = stop + 1

    # assert inputs
    assert start < stop, 'start must be smaller than stop'
    assert stop <= data.shape[1], 'stop must be smaller than the second dimension (time steps) of data'
    assert step > 0, 'step must be positive'
    assert start >= 0, 'start must be positive'

    assert stop - start < 3, 'only three time steps can be compared'

    num_simulations = 14
    num_time_steps = (stop - start) // step

    fig, axs = plt.subplots(num_simulations * num_time_steps, 2, figsize=(10, 5 * num_simulations * num_time_steps))

    for t in range(num_time_steps):
        idx = start + t * step
        for sim in range(num_simulations):
            row = sim + t * num_simulations
            axs[row, 0].imshow(data[sim, idx, 0], aspect='auto')
            axs[row, 1].imshow(data[sim, idx, 1], aspect='auto')

            # add titles and color bars
            axs[row, 0].set_title(f'Sim {sim} h T={idx}')
            axs[row, 1].set_title(f'Sim {sim} q T={idx}')
            fig.colorbar(axs[row, 0].imshow(data[sim, idx, 0], aspect='auto'), ax=axs[row, 0])
            fig.colorbar(axs[row, 1].imshow(data[sim, idx, 1], aspect='auto'), ax=axs[row, 1])

    plt.tight_layout()
    plt.show()

def get_bilinear_interpolation(dataLR, target_shape):

    '''
    Input dataLR: 2D tensor of shape (row, col)
    Input target_shape: tuple of (row, col)

    Output: 2D tensor of shape (row, col)

    Returns the bilinear interpolation of the input dataLR to the target_shape
    '''

    return F.interpolate(dataLR.unsqueeze(0).unsqueeze(0), size=target_shape, mode="bilinear").squeeze(0).squeeze(0)


def plot_bilinear_interpolation(dataLR, target_shape):

    '''
    Input dataLR: 2D tensor of shape (C, row, col)
    Input target_shape: tuple of (row, col)

    Output: None

    Plots the bilinear interpolation of the input dataLR to the target_shape
    '''

    upscaled_h = get_bilinear_interpolation(dataLR[0], target_shape)
    upscaled_q = get_bilinear_interpolation(dataLR[1], target_shape)

    plot_single(torch.stack([upscaled_h, upscaled_q], dim=0), title='Bilinear Interpolation')