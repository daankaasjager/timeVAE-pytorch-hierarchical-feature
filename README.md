Repository for Daan Kaasjager and Roben de Lange's Unsupervised Deep Learning project.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Usage

### Single run
For a single run, execute the command below. At the bottom of ``src/vae_pipeline.py``, you can select the desired model and dataset for your run.
You can choose h_timeVAE or timeVAE. The values set in ``src/config/hyperparameters.yaml`` determines what variant of each model is used. This is explained further below.

   ```bash
   python src/vae_pipeline.py
   ```

### Obtaining results
In order to reproduce our predictive and discriminative scores by executing a full training run, the command below can be executed.
In these runs, the hyperparameters are taken from ``src/config/hyperparameters-air.yaml`` and ``src/config/hyperparameters-stockv.yaml`` are
used. A full run will only run one of the variants of both TimeVAE and H-TimeVAE, based on what the hyperparameters are set to.

   ```bash
   python src/generate_scores.py
   ```
The t-SNE plot grids presented in the paper can be reproduced by executing the command below after training. You can adjust the model type at the bottom of the ``src/compare_plot.py`` file.

   ```bash
   python src/compare_plot.py
   ```

The 'Original' predictor scores from the results are obtained by executing the command below.
   ```bash
   python src/predictor_baseline.py
   ```

### Outputs:
   - Trained models are saved in `./outputs/models/<dataset_name>/`.
   - Generated synthetic data is saved in `./outputs/gen_data/<dataset_name>/` in `.npz` format.
   - t-SNE plots are saved in `./outputs/tsne/<dataset_name>/` in `.png` format.

## Hyperparameters
Shared:
- `latent_dim`: Number of latent dimensions.
- `hidden_layer_sizes`: Number of hidden units or filters.
- `reconstruction_wt`: Weight for the reconstruction loss.
- `batch_size`: Training batch size.
- `trend_poly`: Degree of polynomial trend component.
- `custom_seas`: Custom seasonalities as a list of tuples.
- `use_residual_conn`: Use residual connection.

For H-TimeVAE:
- `layers_per_conv_block`: Number of Convolutional layers per Convolutional block.
- `hierarchical_levels`: Amount of hierarchical levels in the latent space.
- `latent_indexes`: Determines which latent variable goes to which decoder block. Must be a list of four. Each element represents the following decoder blocks in order: [Level, Trend, Seasonality, Residual]

Every model type needs a specific set of hyperparameter values, these are specified here:

### Hyperparameter set-up for TimeVAE-base:
- `trend_poly: 0`
- `custom_seas: null`

### Hyperparameter set-up for TimeVAE-int:
- `trend_poly: 6`

For air dataset:
- ```  
   custom_seas:
    - [24, 1]
    - [7, 24]
  ```
For stockv dataset:
- ```  
   custom_seas:
    - [7, 1]
  ```

### Hyperparameter set-up for H-TimeVAE-split:
- Same as TimeVAE int AND:
- ```
    latent_indexes:
    - 3
    - 3
    - 3
    - 3
  ```
  
### Hyperparameter set-up for H-TimeVAE-last:
- Same as TimeVAE int AND:
- ```
    latent_indexes:
    - 3
    - 2
    - 1
    - 0
  ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
