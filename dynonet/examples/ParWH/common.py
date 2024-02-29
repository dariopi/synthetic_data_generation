import torch
import torch.nn as nn


class ParallelWHDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data):
        """
        Args:
            data (torch.Tensor): Tensor with data organized in.
        """
        self.data = torch.tensor(data)
        self.n_amp, self.n_real, self.seq_len, self.n_channels = data.shape
        self.len = self.n_amp * self.n_real
        self._data = self.data.view(self.n_amp * self.n_real, self.seq_len, self.n_channels)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self._data[idx, :, [0]], self._data[idx, :, [1]]


class StaticNonLin(nn.Module):

    def __init__(self):
        super(StaticNonLin, self).__init__()

        self.net_1 = nn.Sequential(
            nn.Linear(1, 20),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(20, 1)
        )

        self.net_2 = nn.Sequential(
            nn.Linear(1, 20),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, u_lin):

        y_nl_1 = self.net_1(u_lin[..., [0]])  # Process blocks individually
        y_nl_2 = self.net_2(u_lin[..., [1]])  # Process blocks individually
        y_nl = torch.cat((y_nl_1, y_nl_2), dim=-1)

        return y_nl


class StaticMimoNonLin(nn.Module):

    def __init__(self):
        super(StaticMimoNonLin, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 20),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, u_lin):

        y_nl = self.net(u_lin)  # Process blocks individually
        return y_nl