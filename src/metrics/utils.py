import numpy as np
import torch
import torch.nn as nn


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data."""
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = np.array([data_x[i] for i in train_idx])
    test_x = np.array([data_x[i] for i in test_idx])
    train_t = np.array([data_t[i] for i in train_idx])
    test_t = np.array([data_t[i] for i in test_idx])

    no_hat = len(data_x_hat)
    idx_hat = np.random.permutation(no_hat)
    train_idx_hat = idx_hat[:int(no_hat * train_rate)]
    test_idx_hat = idx_hat[int(no_hat * train_rate):]

    train_x_hat = np.array([data_x_hat[i] for i in train_idx_hat])
    test_x_hat = np.array([data_x_hat[i] for i in test_idx_hat])
    train_t_hat = np.array([data_t_hat[i] for i in train_idx_hat])
    test_t_hat = np.array([data_t_hat[i] for i in test_idx_hat])

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
    """Returns Maximum sequence length and each sequence length."""
    time = [len(seq) for seq in data]
    max_seq_len = max(time)
    return time, max_seq_len


class RNNCell(nn.Module):
    def __init__(self, cell_type, hidden_dim):
        super(RNNCell, self).__init__()
        assert cell_type in ['gru', 'lstm']

        if cell_type == 'gru':
            self.rnn_cell = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        elif cell_type == 'lstm':
            self.rnn_cell = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        output, _ = self.rnn_cell(x)
        return output


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation."""
    Z_mb = []
    for t in T_mb:
        temp = torch.zeros((max_seq_len, z_dim))
        temp[:t, :] = torch.rand((t, z_dim))
        Z_mb.append(temp)
    return np.array(Z_mb)


def batch_generator(data, time, batch_size):
    """Mini-batch generator."""
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = [data[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]

    return np.array(X_mb), np.array(T_mb)
