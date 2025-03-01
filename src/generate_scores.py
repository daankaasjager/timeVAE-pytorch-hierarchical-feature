import os

import numpy as np

import paths
from data_utils import (
    load_yaml_file,
    load_data,
    split_data,
    scale_data,
    inverse_transform_data,
    save_scaler,
    save_data,
)
from metrics.pt_discriminative_metrics import discriminative_score_metrics
from metrics.pt_predictive_metrics import predictive_score_metrics
from vae.vae_utils import (
    instantiate_vae_model,
    train_vae,
    save_vae_model,
    get_prior_samples,
)
from visualize import visualize_and_save_tsne


def run_vae_pipeline(dataset_name: str, vae_type: str, n_score_runs: int, n_epochs: int, interpretable: bool):
    # ----------------------------------------------------------------------------------
    # Load data, perform train/valid split, scale data

    # read data
    data = load_data(data_dir=paths.DATASETS_DIR, dataset=dataset_name)

    # split data into train/valid splits
    train_data, valid_data = split_data(data, valid_perc=0.0, shuffle=True)

    # scale data
    scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)

    # ----------------------------------------------------------------------------------
    # Instantiate and train the VAE Model

    # load hyperparameters from yaml file
    if interpretable and dataset_name.startswith("air"):
        print("Loading air hyperparameters")
        hyperparameters = load_yaml_file(paths.HYPERPARAMETERS_AIR)[vae_type]
    elif interpretable and dataset_name.startswith("stockv"):
        print("Loading stockv hyperparameters")
        hyperparameters = load_yaml_file(paths.HYPERPARAMETERS_STOCKV)[vae_type]
    else:
        print("Loading default hyperparameters")
        hyperparameters = load_yaml_file(paths.HYPERPARAMETERS_FILE_PATH)[vae_type]

    # instantiate the model
    _, sequence_length, feature_dim = scaled_train_data.shape
    vae_model = instantiate_vae_model(
        vae_type=vae_type,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        **hyperparameters,
    )

    # train vae
    train_vae(
        vae=vae_model,
        train_data=scaled_train_data,
        max_epochs=n_epochs,
        verbose=0,
    )

    # ----------------------------------------------------------------------------------
    # Save scaler and model
    model_save_dir = os.path.join(paths.MODELS_DIR, dataset_name)
    # save scaler
    save_scaler(scaler=scaler, dir_path=model_save_dir)
    # Save vae
    save_vae_model(vae=vae_model, dir_path=model_save_dir)

    # Generate prior samples, visualize and save them

    # Generate prior samples
    prior_samples = get_prior_samples(vae_model, num_samples=train_data.shape[0])

    # visualize t-sne of original and prior samples
    visualize_and_save_tsne(
        samples1=scaled_train_data,
        samples1_name="Original",
        samples2=prior_samples,
        samples2_name="Generated (Prior)",
        scenario_name=f"Model-{vae_type} Dataset-{dataset_name}",
        save_dir=os.path.join(paths.TSNE_DIR, dataset_name),
        max_samples=2000,
        show_image=False,
    )

    disc_scores = []
    pred_scores = []
    for score_run in range(n_score_runs):
        disc_scores.append(discriminative_score_metrics(scaled_train_data, prior_samples))
        pred_scores.append(predictive_score_metrics(scaled_train_data, prior_samples))

    print(f"Discriminative Score: {np.mean(disc_scores):.4f}")
    print(f"Predictive Score: {np.mean(pred_scores):.4f}")

    # inverse transformer samples to original scale and save to dir
    inverse_scaled_prior_samples = inverse_transform_data(prior_samples, scaler)
    save_data(
        data=inverse_scaled_prior_samples,
        output_file=os.path.join(
            os.path.join(paths.GEN_DATA_DIR, dataset_name),
            f"{vae_type}_{dataset_name}_prior_samples.npz",
        ),
    )
    return disc_scores, pred_scores


if __name__ == "__main__":
    n_runs = 3
    n_score_runs = 2
    n_epochs = 1000
    dataset_percentages = [2, 20]
    # check `/data/` for available datasets
    datasets = []
    for dataset_percentage in dataset_percentages:
        datasets.append(f"air_subsampled_train_perc_{dataset_percentage}")
        datasets.append(f"stockv_subsampled_train_perc_{dataset_percentage}")

    # models: h_timeVAE, timeVAE
    model_names = ["timeVAE", "h_timeVAE"]
    interpretable = True
    model_disc_scores = []
    model_pred_scores = []
    for model_name in model_names:
        final_disc_scores = []
        final_pred_scores = []
        for dataset_name in datasets:
            disc_scores = []
            pred_scores = []
            for run in range(n_runs):
                print(f"Run {run+1}/{n_runs}, dataset {dataset_name}, model {model_name}", flush=True)
                curr_disc_scores, curr_pred_scores = run_vae_pipeline(dataset_name, model_name, n_score_runs, n_epochs, interpretable)
                disc_scores += curr_disc_scores
                pred_scores += curr_pred_scores
            print(f"Final Discriminative Score: {np.mean(disc_scores):.4f} ({np.std(disc_scores):.4f})", flush=True)
            print(f"Final Predictive Score: {np.mean(pred_scores):.4f} ({np.std(pred_scores):.4f})", flush=True)
            path = os.path.join(paths.SCORES_DIR, dataset_name, model_name)
            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, "disc_scores.npy"), disc_scores)
            np.save(os.path.join(path, "pred_scores.npy"), pred_scores)
            final_disc_scores.append((np.mean(disc_scores), np.std(disc_scores)))
            final_pred_scores.append((np.mean(pred_scores), np.std(pred_scores)))
        model_disc_scores.append(final_disc_scores)
        model_pred_scores.append(final_pred_scores)

    for j in range(len(model_disc_scores)):
        final_disc_scores = model_disc_scores[j]
        final_pred_scores = model_pred_scores[j]
        print(model_names[j], flush=True)
        for i in range(len(datasets)):
            print(f"Dataset: {datasets[i]}")
            print(f"Discriminative: {final_disc_scores[i][0]:.4f} ({final_disc_scores[i][1]:.4f})")
            print(f"Predictive: {final_pred_scores[i][0]:.4f} ({final_pred_scores[i][1]:.4f})")

