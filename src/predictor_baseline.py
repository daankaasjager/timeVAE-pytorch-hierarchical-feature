import paths
from data_utils import load_data, split_data, scale_data
from metrics.pt_predictive_metrics import predictive_score_metrics
import numpy as np

def get_predictor_baseline(dataset_name):
    data = load_data(data_dir=paths.DATASETS_DIR, dataset=dataset_name)

    # split data into train/valid splits
    train_data, valid_data = split_data(data, valid_perc=0.2, shuffle=True)

    # scale data
    scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)

    runs = 5
    scores = []
    for _ in range(runs):
        scores.append(predictive_score_metrics(scaled_train_data, scaled_valid_data))
    return np.mean(scores), np.std(scores)

if __name__ == "__main__":
    dataset_percentages = [2, 20]
    datasets = []
    for dataset_percentage in dataset_percentages:
        datasets.append(f"air_subsampled_train_perc_{dataset_percentage}")
        datasets.append(f"stockv_subsampled_train_perc_{dataset_percentage}")

    for dataset in datasets:
        print(dataset)
        mean, std = get_predictor_baseline(dataset)
        print(f"{mean:.4f} ({std:.4f})")

