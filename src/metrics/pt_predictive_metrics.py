import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error
from metrics.utils import extract_time


def predictive_score_metrics(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

  Args:
      - ori_data: original data
      - generated_data: generated synthetic data

  Returns:
      - predictive_score: MAE of the predictions on the original data
  """
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Extract sequence lengths
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)

    # Network parameters
    hidden_dim = dim // 2
    iterations = 5000
    batch_size = 128

    class Predictor(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(Predictor, self).__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            output, _ = self.gru(x)
            y_hat = torch.sigmoid(self.fc(output))
            return y_hat

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = Predictor(dim - 1, hidden_dim).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(predictor.parameters())

    # Training loop
    predictor.train()
    for itt in range(iterations):
        idx = np.random.permutation(len(generated_data))[:batch_size]
        X_mb = torch.tensor([generated_data[i][:-1, :(dim - 1)] for i in idx], dtype=torch.float32).to(device)
        Y_mb = torch.tensor([generated_data[i][1:, (dim - 1)].reshape(-1, 1) for i in idx], dtype=torch.float32).to(
            device)

        optimizer.zero_grad()
        y_pred = predictor(X_mb)
        loss = criterion(y_pred, Y_mb)
        loss.backward()
        optimizer.step()

    # Evaluation on original data
    predictor.eval()
    idx = np.random.permutation(len(ori_data))[:no]
    X_mb = torch.tensor([ori_data[i][:-1, :(dim - 1)] for i in idx], dtype=torch.float32).to(device)
    Y_mb = [ori_data[i][1:, (dim - 1)].reshape(-1, 1) for i in idx]

    with torch.no_grad():
        pred_Y_curr = predictor(X_mb).cpu().numpy()

    # Compute MAE
    MAE_temp = sum(mean_absolute_error(Y_mb[i], pred_Y_curr[i]) for i in range(no)) / no

    return MAE_temp
