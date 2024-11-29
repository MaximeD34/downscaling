import torch.nn as nn
import torch

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize a single ConvLSTM cell.

        Parameters:
        - input_dim (int): Number of channels in the input tensor.
        - hidden_dim (int): Number of channels in the hidden state.
        - kernel_size (int, int): Height and width of the convolutional kernel.
        - bias (bool): Whether to include a bias term in the convolution.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim          # Number of input channels (C_in)
        self.hidden_dim = hidden_dim        # Number of hidden channels (C_hidden)
        self.kernel_size = kernel_size      # Kernel size (K_h, K_w)
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)  # Padding to maintain spatial dimensions
        self.bias = bias

        # The convolution layer processes concatenated input and hidden state
        # Input channels: input_dim + hidden_dim
        # Output channels: 4 * hidden_dim (for i, f, o, g gates)
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        """
        Forward pass of the ConvLSTM cell.

        Parameters:
        - input_tensor (torch.Tensor): Input at the current time step.
          Shape: (batch_size, input_dim, height, width)
        - cur_state (tuple): Tuple of (h_cur, c_cur) representing the current hidden and cell states.
          Each of shape: (batch_size, hidden_dim, height, width)

        Returns:
        - h_next (torch.Tensor): Next hidden state.
          Shape: (batch_size, hidden_dim, height, width)
        - c_next (torch.Tensor): Next cell state.
          Shape: (batch_size, hidden_dim, height, width)
        """
        h_cur, c_cur = cur_state

        # Concatenate input and previous hidden state along the channel dimension
        # combined shape: (batch_size, input_dim + hidden_dim, height, width)
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # Apply convolution to the combined tensor
        # combined_conv shape: (batch_size, 4 * hidden_dim, height, width)
        combined_conv = self.conv(combined)

        # Split the convolution output into four chunks along the channel dimension
        # Each gate tensor shape: (batch_size, hidden_dim, height, width)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # Input gate activation
        i = torch.sigmoid(cc_i)
        # Forget gate activation
        f = torch.sigmoid(cc_f)
        # Output gate activation
        o = torch.sigmoid(cc_o)
        # Cell gate activation
        g = torch.tanh(cc_g)

        # Compute the next cell state
        # Element-wise operations maintain the shape: (batch_size, hidden_dim, height, width)
        c_next = f * c_cur + i * g

        # Compute the next hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initialize the hidden and cell states with zeros.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - image_size (tuple): Spatial dimensions of the input (height, width).

        Returns:
        - (h, c): Tuple of initialized hidden and cell states.
          Each of shape: (batch_size, hidden_dim, height, width)
        """
        height, width = image_size
        device = self.conv.weight.device  # Ensure the states are on the same device as the model
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )

class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM network.

    Parameters:
    - input_dim (int): Number of channels in the input tensor.
    - hidden_dim (int or list): Number of channels in hidden states.
      If int, the same hidden_dim is used for all layers. If list, it must have length equal to num_layers.
    - kernel_size (tuple or list): Convolutional kernel size for each layer.
      If tuple, the same kernel_size is used for all layers. If list, it must have length equal to num_layers.
    - num_layers (int): Number of ConvLSTM layers.
    - batch_first (bool): If True, input and output tensors are provided as (batch, time, channels, height, width).
    - bias (bool): Whether to include bias terms in convolution operations.
    - return_all_layers (bool): If True, return outputs from all layers; otherwise, only return the last layer.

    Input Shape:
    - When batch_first=True: (batch_size, seq_len, input_dim, height, width)
    - When batch_first=False: (seq_len, batch_size, input_dim, height, width)

    Output:
    - layer_output_list: List of outputs for each layer (if return_all_layers=True)
      Each element has shape: (batch_size, seq_len, hidden_dim, height, width)
    - last_state_list: List of the last hidden and cell states for each layer
      Each element is a tuple (h_n, c_n) with shapes: (batch_size, hidden_dim, height, width)
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        # Ensure hidden_dim and kernel_size are lists of length num_layers
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)

        self.input_dim = input_dim          # Number of input channels
        self.hidden_dim = hidden_dim        # List of hidden_dims for each layer
        self.kernel_size = kernel_size      # List of kernel_sizes for each layer
        self.num_layers = num_layers        # Number of layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Create ConvLSTMCell modules for each layer
        self.cell_list = nn.ModuleList([
            ConvLSTMCell(
                input_dim=self.input_dim if i == 0 else self.hidden_dim[i - 1],
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            )
            for i in range(self.num_layers)
        ])

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass through the ConvLSTM network.

        Parameters:
        - input_tensor (torch.Tensor): Input sequence tensor.
          Shape (batch_size, seq_len, input_dim, height, width) if batch_first=True.
        - hidden_state (optional): Initial hidden and cell states for all layers.
          If not provided, zeros are used.

        Returns:
        - layer_output_list: List of outputs from each layer.
          Each output has shape: (batch_size, seq_len, hidden_dim, height, width)
        - last_state_list: List of the last (hidden_state, cell_state) tuples for each layer.
          Each state has shape: (batch_size, hidden_dim, height, width)
        """
        if not self.batch_first:
            # If input is (seq_len, batch_size, ...), transpose to (batch_size, seq_len, ...)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        # Input to the first layer
        cur_layer_input = input_tensor  # Shape: (batch_size, seq_len, input_dim, height, width)

        # Iterate over layers
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]  # Hidden and cell states for current layer
            output_inner = []

            # Iterate over time steps
            for t in range(seq_len):
                # Get input at time t for the current layer
                x_t = cur_layer_input[:, t, :, :, :]  # Shape: (batch_size, input_dim, height, width)

                # Forward through ConvLSTMCell
                h, c = self.cell_list[layer_idx](input_tensor=x_t, cur_state=(h, c))

                # Append current hidden state to output
                output_inner.append(h)

            # Stack outputs along the time dimension
            layer_output = torch.stack(output_inner, dim=1)  # Shape: (batch_size, seq_len, hidden_dim, height, width)

            # Prepare input for the next layer
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.return_all_layers:
            # Only return the output and state of the last layer
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """
        Initialize hidden states for all layers.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - image_size (tuple): Spatial dimensions (height, width).

        Returns:
        - init_states: List of (h_0, c_0) tuples for each layer.
          Each state has shape: (batch_size, hidden_dim, height, width)
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        Ensure that parameter is a list of appropriate length.

        Parameters:
        - param: Parameter to extend (int, tuple, or list).
        - num_layers (int): Number of layers.

        Returns:
        - param_list: List of parameters for each layer.
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param