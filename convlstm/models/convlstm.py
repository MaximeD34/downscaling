# convlstm.py
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim      # Number of channels in input tensor
        self.hidden_dim = hidden_dim    # Number of channels in hidden state

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Concatenate input and previous hidden state along channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)  # (batch_size, input_dim + hidden_dim, H, W)

        # Compute all gate activations in one convolutional operation
        combined_conv = self.conv(combined)

        # Split the activations into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # Apply non-linearities
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Compute next cell state and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        h_cur = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        c_cur = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        return h_cur, c_cur

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # List of hidden dimensions
        self.kernel_size = kernel_size  # List of kernel sizes
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Ensure hidden_dim and kernel_size are lists
        if not isinstance(self.hidden_dim, list):
            self.hidden_dim = [self.hidden_dim] * self.num_layers

        if not isinstance(self.kernel_size, list):
            self.kernel_size = [self.kernel_size] * self.num_layers

        assert len(self.hidden_dim) == self.num_layers
        assert len(self.kernel_size) == self.num_layers

        cell_list = []

        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell = ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            )

            cell_list.append(cell)

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        input_tensor: (batch_size, seq_len, C, H, W)
        hidden_state: None.
        """
        if not self.batch_first:
            # (seq_len, batch_size, C, H, W) -> (batch_size, seq_len, C, H, W)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        bsz, seq_len, _, height, width = input_tensor.size()

        hidden_state = self._init_hidden(batch_size=bsz, image_size=(height, width))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []

        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size)
            )
        return init_states

class ConvLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim):
        super(ConvLSTMModel, self).__init__()
        self.convlstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=False
        )
        self.conv = nn.Conv2d(
            in_channels=hidden_dim[-1],
            out_channels=output_dim,
            kernel_size=1,
            padding=0,
            bias=True
        )

    def forward(self, x):
        # x: (batch_size, seq_len, C, H, W)
        layer_output_list, last_state_list = self.convlstm(x)

        # Get the outputs from the last layer
        output = layer_output_list[0]  # Shape: (batch_size, seq_len, hidden_dim, H, W)

        # Apply convolution to each time step
        outputs = []
        for t in range(output.size(1)):
            x = self.conv(output[:, t, :, :, :])  # Shape: (batch_size, output_dim, H, W)
            outputs.append(x)

        outputs = torch.stack(outputs, dim=1)  # Shape: (batch_size, seq_len, output_dim, H, W)

        return outputs