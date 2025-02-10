import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from metrics.utils import train_test_divide, extract_time, batch_generator

def discriminative_score_metrics(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data

    Args:
        - ori_data: original data
        - generated_data: generated synthetic data

    Returns:
        - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 2
    batch_size = 128

    class Discriminator(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(Discriminator, self).__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, seq_lengths):
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, h_n = self.gru(packed_input)
            y_hat_logit = self.fc(h_n.squeeze(0))
            y_hat = self.sigmoid(y_hat_logit)
            return y_hat_logit, y_hat

    # Initialize model, loss function, and optimizer
    discriminator = Discriminator(dim, hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(discriminator.parameters())

    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)

    # Convert to PyTorch tensors
    train_x, train_x_hat = torch.tensor(train_x, dtype=torch.float32).to(device), torch.tensor(train_x_hat, dtype=torch.float32).to(device)
    test_x, test_x_hat = torch.tensor(test_x, dtype=torch.float32).to(device), torch.tensor(test_x_hat, dtype=torch.float32).to(device)
    train_t, train_t_hat = torch.tensor(train_t, dtype=torch.int32).to(device), torch.tensor(train_t_hat, dtype=torch.int32).to(device)
    test_t, test_t_hat = torch.tensor(test_t, dtype=torch.int32).to(device), torch.tensor(test_t_hat, dtype=torch.int32).to(device)

    # Training loop
    for i in range(iterations):
        optimizer.zero_grad()
        X_mb, T_mb = batch_generator(train_x.cpu().numpy(), train_t.cpu().numpy(), batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat.cpu().numpy(), train_t_hat.cpu().numpy(), batch_size)

        X_mb, X_hat_mb = torch.tensor(X_mb, dtype=torch.float32).to(device), torch.tensor(X_hat_mb, dtype=torch.float32).to(device)
        T_mb, T_hat_mb = torch.tensor(T_mb, dtype=torch.int32).to(device), torch.tensor(T_hat_mb, dtype=torch.int32).to(device)

        optimizer.zero_grad()

        y_logit_real, _ = discriminator(X_mb, T_mb)
        y_logit_fake, _ = discriminator(X_hat_mb, T_hat_mb)

        d_loss_real = criterion(y_logit_real, torch.ones_like(y_logit_real))
        d_loss_fake = criterion(y_logit_fake, torch.zeros_like(y_logit_fake))
        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        optimizer.step()


    # Test the performance on the testing set
    with torch.no_grad():
        y_pred_real_curr = discriminator(test_x, test_t)[1].cpu().numpy()
        y_pred_fake_curr = discriminator(test_x_hat, test_t_hat)[1].cpu().numpy()

    y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
    y_label_final = np.concatenate((np.ones([len(y_pred_real_curr), ]), np.zeros([len(y_pred_fake_curr), ])), axis=0)

    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score
