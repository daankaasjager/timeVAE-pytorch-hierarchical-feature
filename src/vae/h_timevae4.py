import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

from vae.vae_base import BaseVariationalAutoencoder, Sampling


class TrendLayer(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, trend_poly):
        super(TrendLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.trend_dense1 = nn.Linear(self.latent_dim, self.feat_dim * self.trend_poly)
        self.trend_dense2 = nn.Linear(self.feat_dim * self.trend_poly, self.feat_dim * self.trend_poly)

    def forward(self, z):
        trend_params = F.relu(self.trend_dense1(z))
        trend_params = self.trend_dense2(trend_params)
        trend_params = trend_params.view(-1, self.feat_dim, self.trend_poly)

        lin_space = torch.arange(0, float(self.seq_len), 1, device=z.device) / self.seq_len 
        poly_space = torch.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], dim=0) 

        trend_vals = torch.matmul(trend_params, poly_space) 
        trend_vals = trend_vals.permute(0, 2, 1) 
        return trend_vals


class SeasonalLayer(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, custom_seas):
        super(SeasonalLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.custom_seas = custom_seas

        self.dense_layers = nn.ModuleList([
            nn.Linear(latent_dim, feat_dim * num_seasons)
            for num_seasons, len_per_season in custom_seas
        ])
        

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = torch.arange(num_seasons).unsqueeze(1) + torch.zeros(
            (num_seasons, len_per_season), dtype=torch.int32
        )
        season_indexes = season_indexes.view(-1)
        season_indexes = season_indexes.repeat(self.seq_len // len_per_season + 1)[: self.seq_len]
        return season_indexes

    def forward(self, z):
        N = z.shape[0]
        ones_tensor = torch.ones((N, self.feat_dim, self.seq_len), dtype=torch.int32, device=z.device)

        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)
            season_params = season_params.view(-1, self.feat_dim, num_seasons)

            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            ).to(z.device)

            dim2_idxes = ones_tensor * season_indexes_over_time.view(1, 1, -1)
            season_vals = torch.gather(season_params, 2, dim2_idxes)

            all_seas_vals.append(season_vals)

        all_seas_vals = torch.stack(all_seas_vals, dim=-1) 
        all_seas_vals = torch.sum(all_seas_vals, dim=-1)  
        all_seas_vals = all_seas_vals.permute(0, 2, 1)  

        return all_seas_vals

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len, self.feat_dim)
    

class LevelModel(nn.Module):
    def __init__(self, latent_dim, feat_dim, seq_len):
        super(LevelModel, self).__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.level_dense1 = nn.Linear(self.latent_dim, self.feat_dim)
        self.level_dense2 = nn.Linear(self.feat_dim, self.feat_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        level_params = self.relu(self.level_dense1(z))
        level_params = self.level_dense2(level_params)
        level_params = level_params.view(-1, 1, self.feat_dim)

        ones_tensor = torch.ones((1, self.seq_len, 1), dtype=torch.float32, device=z.device)
        level_vals = level_params * ones_tensor
        return level_vals


class ResidualConnection(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, encoder_last_dense_dim):
        super(ResidualConnection, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        
        self.dense = nn.Linear(latent_dim, encoder_last_dense_dim)
        self.deconv_layers = nn.ModuleList()
        in_channels = hidden_layer_sizes[-1]
        
        for i, num_filters in enumerate(reversed(hidden_layer_sizes[:-1])):
            self.deconv_layers.append(
                nn.ConvTranspose1d(in_channels, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            in_channels = num_filters
            
        self.deconv_layers.append(
            nn.ConvTranspose1d(in_channels, feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        L_in = encoder_last_dense_dim // hidden_layer_sizes[-1] 
        for i in range(len(hidden_layer_sizes)):
            L_in = (L_in - 1) * 2 - 2 * 1 + 3 + 1 
        L_final = L_in 

        self.final_dense = nn.Linear(feat_dim * L_final, seq_len * feat_dim)

    def forward(self, z):
        batch_size = z.size(0)
        x = F.relu(self.dense(z))
        x = x.view(batch_size, -1, self.hidden_layer_sizes[-1])
        x = x.transpose(1, 2)
        
        for deconv in self.deconv_layers[:-1]:
            x = F.relu(deconv(x))
        x = F.relu(self.deconv_layers[-1](x))
        
        x = x.flatten(1)
        x = self.final_dense(x)
        residuals = x.view(-1, self.seq_len, self.feat_dim)
        return residuals
    

class HTimeVAEEncoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, hidden_layer_amount, latent_dim, hierarchical_levels = 4):
        super(HTimeVAEEncoder, self).__init__()
        print(seq_len, feat_dim, latent_dim)
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_amount = hidden_layer_amount
        self.hierarchical_levels = hierarchical_levels

        # Stem block.
        stem_layers = nn.ModuleList()
        stem_layers.append(nn.Conv1d(feat_dim, hidden_layer_sizes[0], kernel_size=3, stride=2, padding=1))
        stem_layers.append(nn.ReLU())
        self.stem = nn.Sequential(*stem_layers)

        # Create the convolutional blocks.
        self.conv_blocks = nn.ModuleList()
        for i in range(hierarchical_levels):
            start = i * self.hidden_layer_amount
            end = (i + 1) * self.hidden_layer_amount + 1
            self.conv_blocks.append(self.init_conv_block(hidden_layer_sizes[start:end]))

        # Determine the flattened dimensions after each conv block.
        self.encoder_last_dense_dims = self._get_last_dense_dim(seq_len, feat_dim, hidden_layer_sizes)
        print("encoder_last_dense_dims", self.encoder_last_dense_dims)
        self.encoder_last_dense_dims = list(reversed(self.encoder_last_dense_dims))
        print(
            "encoder_last_dense_dims after reverse",
            self.encoder_last_dense_dims,)  
        
        # Instead of concatenating previous latent variables, add them (after a learned transform).
        # For the first level, use an identity.
        self.residual_transforms = nn.ModuleList()
        for i in range(hierarchical_levels):
            if i == 0:
                self.residual_transforms.append(nn.Identity())
            else:
                # Map the previous latent (of dimension latent_dim) to match the flattened conv output.
                self.residual_transforms.append(
                    nn.Linear(self.latent_dim, self.encoder_last_dense_dims[i])
                )

        # Create the z_mean and z_log_var layers for each level.
        self.z_mean_layers = nn.ModuleList()
        self.z_log_var_layers = nn.ModuleList()
        # Note: Now the input dimension for level i is just the flattened conv output (with residual added)
        for i in range(hierarchical_levels):
            input_dim = self.encoder_last_dense_dims[i]
            self.z_mean_layers.append(nn.Linear(input_dim, self.latent_dim))
            self.z_log_var_layers.append(nn.Linear(input_dim, self.latent_dim))

        self.flatten = nn.Flatten()

    def init_conv_block(self, hidden_layer_sizes):
        block = nn.ModuleList()
        for i, num_filters in enumerate(hidden_layer_sizes[1:]):
            block.append(nn.Conv1d(hidden_layer_sizes[i], num_filters, kernel_size=3, stride=2, padding=1))
            block.append(nn.ReLU())
        return nn.Sequential(*block)

    def compute_flattened_size(self, input_size, hidden_layer_sizes, kernel_size, stride, padding):
        output_size = input_size  # Start with the input size
        for i in range(1, len(hidden_layer_sizes)):  # Iterate through layers
            output_size = (output_size - kernel_size + 2 * padding) // stride + 1
        flattened_size = output_size * hidden_layer_sizes[-1]  # Multiply by the last layer's channels
        return flattened_size

    def _get_last_dense_dim(self, seq_len, feat_dim, hidden_layer_sizes):
        dims = []
        with torch.no_grad():
            x = torch.randn(1, feat_dim, seq_len)
            x = self.stem(x)
            for conv_block in self.conv_blocks:
                for conv in conv_block:
                    x = conv(x)
                dims.append(x.numel())
        return dims

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.stem(x)
        # We store conv outputs in reverse order (as before)
        conv_outputs = [None] * self.hierarchical_levels
        for i in range(self.hierarchical_levels):
            x = self.conv_blocks[i](x)
            conv_outputs[self.hierarchical_levels - i - 1] = x

        z_means = []
        z_log_vars = []
        z_list = []
        # For each level, compute the latent distribution from the flattened conv output.
        for i in range(self.hierarchical_levels):
            dense_input = self.flatten(conv_outputs[i])
            if i != 0:
                # Instead of concatenation, add a transformed version of the previous latent.
                residual = self.residual_transforms[i](z_list[i - 1])
                dense_input = dense_input + residual

            z_mean = self.z_mean_layers[i](dense_input)
            z_log_var = self.z_log_var_layers[i](dense_input)
            z_means.append(z_mean)
            z_log_vars.append(z_log_var)
            z_list.append(Sampling()([z_mean, z_log_var]))
        return z_means, z_log_vars, z_list


class HTimeVAEDecoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, trend_poly=0, custom_seas=None, use_residual_conn=True, encoder_last_dense_dim=None):
        super(HTimeVAEDecoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn

        self.level_model = LevelModel(self.latent_dim, self.feat_dim, self.seq_len)

        if self.trend_poly > 0:
            self.trend_layer = TrendLayer(self.seq_len, self.feat_dim, self.latent_dim, self.trend_poly)
        else:
            self.trend_layer = None

        if self.custom_seas is not None and len(self.custom_seas) > 0:
            self.seasonal_layer = SeasonalLayer(self.seq_len, self.feat_dim, self.latent_dim, self.custom_seas)
        else:
            self.seasonal_layer = None

        if use_residual_conn:
            self.residual_conn = ResidualConnection(seq_len, feat_dim, hidden_layer_sizes, latent_dim, encoder_last_dense_dim)

    def forward(self, z):
        outputs = self.level_model(z[0])

        if self.trend_layer is not None:
            trend_vals = self.trend_layer(z[1].to(next(self.trend_layer.parameters()).device))
            outputs += trend_vals

        if self.seasonal_layer is not None:
            seasonal_vals = self.seasonal_layer(z[2].to(next(self.seasonal_layer.parameters()).device))
            outputs += seasonal_vals


        if self.use_residual_conn:
            residuals = self.residual_conn(z[3])
            outputs += residuals

        return outputs



class HTimeVAE(BaseVariationalAutoencoder):
    model_name = "h_timeVAE"

    def __init__(
        self,
        hidden_layer_sizes=None,
        trend_poly=0,
        custom_seas=None,
        use_residual_conn=True,
        **kwargs,
    ):
        super(HTimeVAE, self).__init__(**kwargs)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [15, 30, 60, 90, 120, 150, 180, 210, 240]
        self.layer_per_conv_block = 2
        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _get_encoder(self):
        return HTimeVAEEncoder(self.seq_len, self.feat_dim, self.hidden_layer_sizes, self.layer_per_conv_block, self.latent_dim)

    def _get_decoder(self):
        return HTimeVAEDecoder(self.seq_len, self.feat_dim, self.hidden_layer_sizes, self.latent_dim, self.trend_poly, self.custom_seas, self.use_residual_conn, self.encoder.encoder_last_dense_dims[-1])

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, f"{self.model_name}_weights.pth"))

        if self.custom_seas is not None:
            self.custom_seas = [(int(num_seasons), int(len_per_season)) for num_seasons, len_per_season in self.custom_seas]

        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "trend_poly": self.trend_poly,
            "custom_seas": self.custom_seas,
            "use_residual_conn": self.use_residual_conn,
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)

    @classmethod
    def load(cls, model_dir: str) -> "HTimeVAE":
        params_file = os.path.join(model_dir, f"{cls.model_name}_parameters.pkl")
        dict_params = joblib.load(params_file)
        vae_model = HTimeVAE(**dict_params)
        vae_model.load_state_dict(torch.load(os.path.join(model_dir, f"{cls.model_name}_weights.pth")))
        return vae_model