import matplotlib.pyplot as plt

def plot_single(data):
    fig, axs = plt.subplots(1, 2, figsize=(10, 2))

    axs[0].imshow(data[0], aspect='auto')
    axs[1].imshow(data[1], aspect='auto')

    #add titles and color bars
    axs[0].set_title('h')
    axs[1].set_title('q')
    fig.colorbar(axs[0].imshow(data[0], aspect='auto'), ax=axs[0])
    fig.colorbar(axs[1].imshow(data[1], aspect='auto'), ax=axs[1])

    plt.show()

def plot_batch(data, start, stop, step):
    # data of shape torch.Size([10, 2, 5, 250])
    # plot 10 * 2 heatmaps of shape (5, 250) next to each other

    stop = stop+1

    #assert inputs
    assert start < stop, 'start must be smaller than stop'
    assert stop <= data.shape[0], 'stop must be smaller than the first dimension of data'
    assert step > 0, 'step must be positive'
    assert start >= 0, 'start must be positive'


    fig, axs = plt.subplots((stop-start) // step, 2, figsize=(10, 5))

    for i in range((stop-start) // step):
        axs[i, 0].imshow(data[start + i*step, 0], aspect='auto')
        axs[i, 1].imshow(data[start + i*step, 1], aspect='auto')

        # add titles and color bars
        axs[i, 0].set_title(f'h {start + i*step}')
        axs[i, 1].set_title(f'q {start + i*step}')
        fig.colorbar(axs[i, 0].imshow(data[start + i*step, 0], aspect='auto'), ax=axs[i, 0])
        fig.colorbar(axs[i, 1].imshow(data[start + i*step, 1], aspect='auto'), ax=axs[i, 1])

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