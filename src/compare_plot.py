import os

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import paths
from data_utils import (
    load_yaml_file,
    load_data,
    split_data,
    scale_data,
)
from vae.vae_utils import (
    instantiate_vae_model,
    train_vae,
    get_prior_samples, load_vae_model,
)


def visualize_multiple_tsne(
        dataset_names: list,
        perc_list: list,
        vae_type: str,
        model_name: str,
        save_dir: str,
        max_samples: int = 1000,
) -> None:
    """
    Visualize t-SNE for multiple datasets and percentages in a 5x4 grid of subplots.

    Args:
        dataset_names (list): List of dataset names to process.
        perc_list (list): List of percentages to process for each dataset.
        vae_type (str): Type of vae to use.
        model_name (str): Name of the VAE model to use.
        save_dir (str): Directory path to save the resulting plots.
        max_samples (int): Maximum number of samples to use in the t-SNE plots.
    """
    fig, axes = plt.subplots(len(perc_list), len(dataset_names), figsize=(8, 8))

    total_iterations = len(dataset_names) * len(perc_list)
    iteration_count = 0

    for i, dataset_name in enumerate(dataset_names):
        for j, perc in enumerate(perc_list):
            iteration_count += 1
            print(f"Processing {dataset_name} at {perc}%... ({iteration_count}/{total_iterations})")

            # Generate dataset filename
            dataset_filename = f"{dataset_name}_subsampled_train_perc_{perc}"

            # Load data, train the model, and visualize t-SNE for the current dataset and percentage
            scenario_name = f"{dataset_name} | p = {perc}"

            # Load data
            data = load_data(data_dir=paths.DATASETS_DIR, dataset=dataset_filename)

            # Split data
            train_data, valid_data = split_data(data, valid_perc=0.1, shuffle=True)

            # Scale data
            scaled_train_data, _, scaler = scale_data(train_data, valid_data)

            # Load VAE model and generate prior samples
            hyperparameters = load_yaml_file(paths.HYPERPARAMETERS_FILE_PATH)[vae_type]

            model_save_dir = os.path.join(paths.MODELS_DIR, dataset_filename)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vae_model = load_vae_model(vae_type, model_save_dir, hyperparameters).to(device)

            # Generate prior samples
            prior_samples = get_prior_samples(vae_model, num_samples=train_data.shape[0])

            samples1 = scaled_train_data
            samples2 = prior_samples
            samples1_2d = np.mean(samples1, axis=2)
            samples2_2d = np.mean(samples2, axis=2)

            used_samples = min(samples1_2d.shape[0], max_samples)

            # Combine the original and generated samples
            combined_samples = np.vstack(
                [samples1_2d[:used_samples], samples2_2d[:used_samples]]
            )

            # Compute the t-SNE of the combined samples
            tsne = TSNE(n_components=2, perplexity=40, n_iter=300, random_state=42)
            tsne_samples = tsne.fit_transform(combined_samples)

            # Create a DataFrame for the t-SNE samples
            tsne_df = pd.DataFrame(
                {
                    "tsne_1": tsne_samples[:, 0],
                    "tsne_2": tsne_samples[:, 1],
                    "sample_type": ["Original"] * used_samples
                                   + ["Generated (Prior)"] * used_samples,
                }
            )

            # Plot the t-SNE samples in the corresponding subplot
            ax = axes[j, i]
            for sample_type, color in zip(["Original", "Generated (Prior)"], ["red", "blue"]):
                indices = tsne_df["sample_type"] == sample_type
                ax.scatter(
                    tsne_df.loc[indices, "tsne_1"],
                    tsne_df.loc[indices, "tsne_2"],
                    color=color,
                    alpha=0.5,
                    s=50,
                )
            ax.set_title(f"{scenario_name}")

    # Set the overall title for the plot
    plt.suptitle(f"{model_name} t-SNE plot", fontsize=20)

    # Adjust layout and save the entire figure
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{model_name}_tsne_grid.png"))
    print(f"All plots saved to {save_dir}")
    plt.show()

if __name__ == "__main__":
    dataset_names = ["air", "stockv"]
    perc_list = [2, 20]
    vae_type = "h_timeVAE"
    model_name = "H-TimeVAE-split"
    save_dir = os.path.join(paths.TSNE_DIR, "tsne_plots")

    visualize_multiple_tsne(
        dataset_names=dataset_names,
        perc_list=perc_list,
        vae_type=vae_type,
        model_name=model_name,
        save_dir=save_dir,
        max_samples=700,
    )