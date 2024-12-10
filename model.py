# model.py

import torch
import torch.nn as nn
import datetime
import time
import torch.nn.init as init
# Function to check for NaN values
def check_for_nans(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"NaNs detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Infs detected in {name}")
class AgePredictionModel(nn.Module):
    def __init__(self, input_size_mlp1=78, input_size_mlp2=514, input_size_mlp3=1, lstm_hidden_size=128):
        super(AgePredictionModel, self).__init__()

        # MLP1 for x_features with BatchNorm and ReLU
        self.mlp1 = nn.Sequential(
            nn.Linear(input_size_mlp1, 128),
            nn.BatchNorm1d(128),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),   # Batch Normalization
            nn.ReLU()
        )

        # MLP2 for x_icd and x_disease with BatchNorm and ReLU
        self.mlp2 = nn.Sequential(
            nn.Linear(input_size_mlp2, 128),
            nn.BatchNorm1d(128),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),   # Batch Normalization
            nn.ReLU()
        )

        # LSTM for combining outputs of mlp1 and mlp2
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, batch_first=True)

        # MLP3 for x_sex with BatchNorm and ReLU
        self.mlp3 = nn.Sequential(
            nn.Linear(input_size_mlp3, 32),
            nn.BatchNorm1d(32),   # Batch Normalization
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),   # Batch Normalization
            nn.ReLU()
        )

        # Learnable distance weight for each system
        self.distance_weights = nn.ParameterDict({
            'cardiovascular': nn.Parameter(torch.randn(1) * 1.5),
            'pulmonary': nn.Parameter(torch.randn(1) * 1.5),
            'musculoskeletal': nn.Parameter(torch.randn(1) * 1.5),
            'immune': nn.Parameter(torch.randn(1) * 1.5),
            'renal': nn.Parameter(torch.randn(1) * 1.5),
            'hepatic': nn.Parameter(torch.randn(1) * 1.5),
            'metabolic': nn.Parameter(torch.randn(1) * 1.5)
        })

        # Output layer for each system
        self.output_layers_1 = nn.ModuleDict({
            'cardiovascular': nn.Linear(144, 64),
            'pulmonary': nn.Linear(144, 64),
            'musculoskeletal': nn.Linear(144, 64),
            'immune': nn.Linear(144, 64),
            'renal': nn.Linear(144, 64),
            'hepatic': nn.Linear(144, 64),
            'metabolic': nn.Linear(144, 64)
        })

        # Output layer for each system
        self.output_layers_2 = nn.ModuleDict({
            'cardiovascular': nn.Linear(64, 1),
            'pulmonary': nn.Linear(64, 1),
            'musculoskeletal': nn.Linear(64, 1),
            'immune': nn.Linear(64, 1),
            'renal': nn.Linear(64, 1),
            'hepatic': nn.Linear(64, 1),
            'metabolic': nn.Linear(64, 1)
        })

    def forward(self, x_icd, x_disease, x_features, x_sex, system_mask):
        # 替换 NaN 值
        # start_time = time.time()
        x_icd = x_icd.squeeze(0)
        x_disease = x_disease.squeeze(0)
        x_features = x_features.squeeze(0)


        x_icd = torch.nan_to_num(x_icd, nan=0.0)
        x_disease = torch.nan_to_num(x_disease, nan=0.0)
        x_features = torch.nan_to_num(x_features, nan=0.0)
        # x_sex = torch.nan_to_num(x_sex, nan=0.0)
        system_mask = system_mask.squeeze(0)
        # print('x_sex:', x_sex.shape)
        # Check if the shape is (20000, 1)
        # 处理 x_sex
        if x_sex.shape != (20000, 1):
            x_sex = x_sex.squeeze()  # Remove singleton dimensions
            if x_sex.shape[0] == 20000 or x_sex.shape[0] == int(600000 % 20000):
                x_sex = x_sex.view(-1, 1)  # Reshape to (512, 1)
            else:
                raise ValueError("Tensor cannot be reshaped to (512, 1)")
        check_for_nans(x_icd, "x_icd")
        check_for_nans(x_disease, "x_disease")
        check_for_nans(x_features, "x_features")
        check_for_nans(x_sex, "x_sex")
        # x_icd(batch_size,15,514),x_disease(batch_size,15,514),x_features(batch_size,1,78),x_sex(batch_size,1,1),system_mask(batch_size,7)
        # Forward pass through the model

        x_icd_out = self.mlp2(x_icd)
        check_for_nans(x_icd_out, "x_icd_out")

        x_disease_out = self.mlp2(x_disease)
        check_for_nans(x_disease_out, "x_disease_out")

        x_features_out = self.mlp1(x_features).unsqueeze(1)
        check_for_nans(x_features_out, "x_features_out")

        # Combine MLP2 outputs (concatenate along sequence length)
        x_combined = torch.cat((x_icd_out, x_disease_out), dim=1)

        # Concatenate and pass through LSTM
        lstm_input = torch.cat((x_combined, x_features_out), dim=1)
        lstm_out, _ = self.lstm(lstm_input)
        check_for_nans(lstm_out, "lstm_out")

        lstm_last_out = lstm_out[:, -1, :]
        check_for_nans(lstm_last_out, "lstm_last_out")

        x_sex_out = self.mlp3(x_sex).squeeze(1)
        check_for_nans(x_sex_out, "x_sex_out")

        x_combined = torch.cat((lstm_last_out, x_sex_out), dim=1)
        check_for_nans(x_combined, "x_combined")

        outputs = {}
        system_names = list(self.output_layers_1.keys())
        for i, system in enumerate(system_names):
            distance_weight = self.distance_weights[system]
            # 仅在系统患病时应用距离权重
            system_1 = system_mask[:, i].unsqueeze(1)
            system_2 = 1 + distance_weight * system_1
            adjusted_output = self.output_layers_1[system](x_combined) * (1 + distance_weight * system_mask[:, i].unsqueeze(1))
            outputs[system] = nn.ReLU(adjusted_output)
        # print('output_layers_1:', time.time() - start_time)
        check_for_nans(x_icd_out, "x_icd_out")
        check_for_nans(x_disease_out, "x_disease_out")
        check_for_nans(x_features_out, "x_features_out")
        check_for_nans(x_sex_out, "x_sex_out")
        # 计算经过 output_layer_2 的最终输出
        final_outputs = {}
        for system, output_layer in self.output_layers_2.items():
            # 直接将 adjusted_output 传递给 output_layer_2
            final_output = output_layer(outputs[system])
            final_outputs[system] = final_output
        # print('output_layers_2:', time.time() - start_time)
        # print(final_outputs)
        return nn.ReLU(final_outputs)
class AgePredictionModel(nn.Module):
    def __init__(self, input_size_mlp1=78, input_size_mlp2=514, input_size_mlp3=1, lstm_hidden_size=128):
        super(AgePredictionModel, self).__init__()

        # MLP1 for x_features
        self.mlp1_fc1 = nn.Linear(input_size_mlp1, 128)
        self.mlp1_bn1 = nn.BatchNorm1d(128)  # Batch Normalization
        self.mlp1_relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.mlp1_fc2 = nn.Linear(128, 64)
        self.mlp1_bn2 = nn.BatchNorm1d(64)   # Batch Normalization
        self.mlp1_relu2 = nn.LeakyReLU(negative_slope=0.01)

        # MLP2 for x_icd and x_disease
        self.mlp2_fc1 = nn.Linear(input_size_mlp2, 128)
        self.mlp2_bn1 = nn.BatchNorm1d(128)  # Batch Normalization
        self.mlp2_relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.mlp2_fc2 = nn.Linear(128, 64)
        self.mlp2_bn2 = nn.BatchNorm1d(64)   # Batch Normalization
        self.mlp2_relu2 = nn.LeakyReLU(negative_slope=0.01)

        # LSTM for combining outputs of mlp1 and mlp2
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, batch_first=True)

        # MLP3 for x_sex
        self.mlp3_fc1 = nn.Linear(input_size_mlp3, 32)
        self.mlp3_bn1 = nn.BatchNorm1d(32)   # Batch Normalization
        self.mlp3_relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.mlp3_fc2 = nn.Linear(32, 16)
        self.mlp3_bn2 = nn.BatchNorm1d(16)   # Batch Normalization
        self.mlp3_relu2 = nn.LeakyReLU(negative_slope=0.01)

        self.ReLU_1 = nn.LeakyReLU(negative_slope=0.01)
        self.ReLU_2 = nn.LeakyReLU(negative_slope=0.01)

        # Learnable distance weight for each system
        self.distance_weights = nn.ParameterDict({
            'cardiovascular': nn.Parameter(torch.randn(1) * 0.1),
            'pulmonary': nn.Parameter(torch.randn(1) * 0.1),
            'musculoskeletal': nn.Parameter(torch.randn(1) * 0.1),
            'immune': nn.Parameter(torch.randn(1) * 0.1),
            'renal': nn.Parameter(torch.randn(1) * 0.1),
            'hepatic': nn.Parameter(torch.randn(1) * 0.1),
            'metabolic': nn.Parameter(torch.randn(1) * 0.1)
        })

        # Output layer for each system
        self.output_layers_1 = nn.ModuleDict({
            'cardiovascular': nn.Linear(144, 64),
            'pulmonary': nn.Linear(144, 64),
            'musculoskeletal': nn.Linear(144, 64),
            'immune': nn.Linear(144, 64),
            'renal': nn.Linear(144, 64),
            'hepatic': nn.Linear(144, 64),
            'metabolic': nn.Linear(144, 64)
        })

        # Output layer for each system
        self.output_layers_2 = nn.ModuleDict({
            'cardiovascular': nn.Linear(64, 1),
            'pulmonary': nn.Linear(64, 1),
            'musculoskeletal': nn.Linear(64, 1),
            'immune': nn.Linear(64, 1),
            'renal': nn.Linear(64, 1),
            'hepatic': nn.Linear(64, 1),
            'metabolic': nn.Linear(64, 1)
        })

        # Initialize weights with Xavier uniform
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier initialization for the Linear layers
        init.xavier_uniform_(self.mlp1_fc1.weight)
        init.xavier_uniform_(self.mlp1_fc2.weight)
        init.xavier_uniform_(self.mlp2_fc1.weight)
        init.xavier_uniform_(self.mlp2_fc2.weight)
        init.xavier_uniform_(self.mlp3_fc1.weight)
        init.xavier_uniform_(self.mlp3_fc2.weight)

        # You can initialize biases to zero, or use another initialization scheme
        init.zeros_(self.mlp1_fc1.bias)
        init.zeros_(self.mlp1_fc2.bias)
        init.zeros_(self.mlp2_fc1.bias)
        init.zeros_(self.mlp2_fc2.bias)
        init.zeros_(self.mlp3_fc1.bias)
        init.zeros_(self.mlp3_fc2.bias)

    def forward(self, x_icd, x_disease, x_features, x_sex, system_mask):
        # 替换 NaN 值
        x_icd = x_icd.squeeze(0)
        x_disease = x_disease.squeeze(0)
        x_features = x_features.squeeze(0)
        x_icd = torch.nan_to_num(x_icd, nan=0.0)
        x_disease = torch.nan_to_num(x_disease, nan=0.0)
        x_features = torch.nan_to_num(x_features, nan=0.0)
        system_mask = system_mask.squeeze(0)

        # 处理 x_sex
        if x_sex.shape != (20000, 1):
            x_sex = x_sex.squeeze()  # Remove singleton dimensions
            if x_sex.shape[0] == 20000 or x_sex.shape[0] == int(1325056 % 20000):
                x_sex = x_sex.view(-1, 1)  # Reshape to (512, 1)
            else:
                raise ValueError("Tensor cannot be reshaped to (512, 1)")

        # Forward pass through the model
        # MLP1 for x_features
        x_features_out = self.mlp1_relu1(self.mlp1_bn1(self.mlp1_fc1(x_features)))  # Apply FC1 -> BN -> ReLU
        x_features_out = self.mlp1_relu2(self.mlp1_bn2(self.mlp1_fc2(x_features_out)))  # Apply FC2 -> BN -> ReLU
        x_features_out = x_features_out.unsqueeze(1)

        # MLP2 for x_icd and x_disease
        x_icd_out = self.mlp2_relu1(self.mlp2_bn1(self.mlp2_fc1(x_icd).permute(0,2,1)).permute(0,2,1))  # Apply FC1 -> BN -> ReLU
        x_icd_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_icd_out).permute(0,2,1)).permute(0,2,1))  # Apply FC2 -> BN -> ReLU

        x_disease_out = self.mlp2_relu1(self.mlp2_bn1(self.mlp2_fc1(x_disease).permute(0,2,1)).permute(0,2,1))  # Apply FC1 -> BN -> ReLU
        x_disease_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_disease_out).permute(0,2,1)).permute(0,2,1))  # Apply FC2 -> BN -> ReLU

        # Combine MLP2 outputs (concatenate along sequence length)
        x_combined = torch.cat((x_icd_out, x_disease_out), dim=1)

        # Concatenate and pass through LSTM
        lstm_input = torch.cat((x_combined, x_features_out), dim=1)
        lstm_out, _ = self.lstm(lstm_input)

        lstm_last_out = lstm_out[:, -1, :]  # Take the last LSTM output

        # MLP3 for x_sex
        x_sex_out = self.mlp3_relu1(self.mlp3_bn1(self.mlp3_fc1(x_sex)))  # Apply FC1 -> BN -> ReLU
        x_sex_out = self.mlp3_relu2(self.mlp3_bn2(self.mlp3_fc2(x_sex_out)))  # Apply FC2 -> BN -> ReLU
        x_sex_out = x_sex_out.squeeze(1)

        # Concatenate LSTM output with MLP3 output
        x_combined = torch.cat((lstm_last_out, x_sex_out), dim=1)

        outputs = {}
        system_names = list(self.output_layers_1.keys())
        for i, system in enumerate(system_names):
            distance_weight = self.distance_weights[system]
            adjusted_output = self.output_layers_1[system](x_combined) * (1 + distance_weight * system_mask[:, i].unsqueeze(1))
            outputs[system] = self.ReLU_1(adjusted_output)

        # Compute final outputs
        final_outputs = {}
        for system, output_layer in self.output_layers_2.items():
            final_output = output_layer(outputs[system])
            final_outputs[system] = self.ReLU_2(final_output)

        return final_outputs