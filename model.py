# model.py

import torch
import torch.nn as nn
import datetime
import time
import torch.nn.init as init
from sympy.matrices.expressions.matadd import combine


# Function to check for NaN values
def check_for_nans(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"NaNs detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Infs detected in {name}")

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

    def forward(self, x_icd, x_disease, x_drugs, x_features, x_sex, system_mask,args):
        # 替换 NaN 值
        x_icd = x_icd.squeeze(0)
        x_disease = x_disease.squeeze(0)
        x_features = x_features.squeeze(0)
        x_drugs = x_drugs.squeeze(0)
        x_icd = torch.nan_to_num(x_icd, nan=0.0)
        x_disease = torch.nan_to_num(x_disease, nan=0.0)
        x_drugs = torch.nan_to_num(x_drugs, nan=0.0)
        x_features = torch.nan_to_num(x_features, nan=0.0)
        system_mask = system_mask.squeeze(0)

        '''
        Feature data
        '''
        if args.model =='Phenotypes':
            # Forward pass through the model
            # MLP1 for x_features
            x_features_out = self.mlp1_relu1(self.mlp1_bn1(self.mlp1_fc1(x_features)))  # Apply FC1 -> BN -> ReLU
            x_features_out = self.mlp1_relu2(self.mlp1_bn2(self.mlp1_fc2(x_features_out)))  # Apply FC2 -> BN -> ReLU
            x_features_out = x_features_out.unsqueeze(1)

            lstm_out, _ = self.lstm(x_features_out)
            # lstm_out, _ = self.lstm(x_combined)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128

        elif args.model =='ICD':
            # MLP2 for x_icd and x_disease
            x_icd_out = self.mlp2_relu1(
                self.mlp2_bn1(self.mlp2_fc1(x_icd).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC1 -> BN -> ReLU
            x_icd_out = self.mlp2_relu2(
                self.mlp2_bn2(self.mlp2_fc2(x_icd_out).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC2 -> BN -> ReLU
            # lstm_input = torch.cat((x_combined, x_features_out), dim=1)
            lstm_out, _ = self.lstm(x_icd_out)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128

        elif args.model =='Disease':
            x_disease_out = self.mlp2_relu1(
                self.mlp2_bn1(self.mlp2_fc1(x_disease).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC1 -> BN -> ReLU
            x_disease_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_disease_out).permute(0, 2, 1)).permute(0, 2,
                                                                                                                 1))  # Apply FC2 -> BN -> ReLU
            # Concatenate and pass through LSTM
            lstm_out, _ = self.lstm(x_disease_out)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128

        elif args.model =='Drug':
            x_drug_out = self.mlp2_relu1(self.mlp2_bn1(self.mlp2_fc1(x_drugs).permute(0, 2, 1)).permute(0, 2, 1))
            x_drug_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_drug_out).permute(0, 2, 1)).permute(0, 2, 1))
            # Concatenate and pass through LSTM
            lstm_out, _ = self.lstm(x_drug_out)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128
        elif args.model =='Phenotypes and ICD':
            # MLP1 for x_features
            x_features_out = self.mlp1_relu1(self.mlp1_bn1(self.mlp1_fc1(x_features)))  # Apply FC1 -> BN -> ReLU
            x_features_out = self.mlp1_relu2(self.mlp1_bn2(self.mlp1_fc2(x_features_out)))  # Apply FC2 -> BN -> ReLU
            x_features_out = x_features_out.unsqueeze(1)

            x_icd_out = self.mlp2_relu1(
                self.mlp2_bn1(self.mlp2_fc1(x_icd).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC1 -> BN -> ReLU
            x_icd_out = self.mlp2_relu2(
                self.mlp2_bn2(self.mlp2_fc2(x_icd_out).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC2 -> BN -> ReLU

            # Concatenate and pass through LSTM
            lstm_input = torch.cat((x_icd_out, x_features_out), dim=1)
            lstm_out, _ = self.lstm(lstm_input)
            # lstm_out, _ = self.lstm(x_combined)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128
        elif args.model =='Phenotypes and Disease':
            # MLP1 for x_features
            x_features_out = self.mlp1_relu1(self.mlp1_bn1(self.mlp1_fc1(x_features)))  # Apply FC1 -> BN -> ReLU
            x_features_out = self.mlp1_relu2(self.mlp1_bn2(self.mlp1_fc2(x_features_out)))  # Apply FC2 -> BN -> ReLU
            x_features_out = x_features_out.unsqueeze(1)

            x_disease_out = self.mlp2_relu1(
                self.mlp2_bn1(self.mlp2_fc1(x_disease).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC1 -> BN -> ReLU
            x_disease_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_disease_out).permute(0, 2, 1)).permute(0, 2,
                                                                                                                 1))  # Apply FC2 -> BN -> ReLU

            # Concatenate and pass through LSTM
            lstm_input = torch.cat((x_disease_out, x_features_out), dim=1)
            lstm_out, _ = self.lstm(lstm_input)
            # lstm_out, _ = self.lstm(x_combined)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128

        elif args.model =='Phenotypes and Drug':
            # MLP1 for x_features
            x_features_out = self.mlp1_relu1(self.mlp1_bn1(self.mlp1_fc1(x_features)))  # Apply FC1 -> BN -> ReLU
            x_features_out = self.mlp1_relu2(self.mlp1_bn2(self.mlp1_fc2(x_features_out)))  # Apply FC2 -> BN -> ReLU
            x_features_out = x_features_out.unsqueeze(1)

            x_drug_out = self.mlp2_relu1(self.mlp2_bn1(self.mlp2_fc1(x_drugs).permute(0, 2, 1)).permute(0, 2, 1))
            x_drug_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_drug_out).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC2 -> BN -> ReLU

            # Concatenate and pass through LSTM
            lstm_input = torch.cat((x_drug_out, x_features_out), dim=1)
            lstm_out, _ = self.lstm(lstm_input)
            # lstm_out, _ = self.lstm(x_combined)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128

        elif args.model =='ICD and Disease':
            # MLP2 for x_icd and x_disease
            x_icd_out = self.mlp2_relu1(
                self.mlp2_bn1(self.mlp2_fc1(x_icd).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC1 -> BN -> ReLU
            x_icd_out = self.mlp2_relu2(
                self.mlp2_bn2(self.mlp2_fc2(x_icd_out).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC2 -> BN -> ReLU

            x_disease_out = self.mlp2_relu1(
                self.mlp2_bn1(self.mlp2_fc1(x_disease).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC1 -> BN -> ReLU
            x_disease_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_disease_out).permute(0, 2, 1)).permute(0, 2,1))  # Apply FC2 -> BN -> ReLU
            # Concatenate and pass through LSTM
            lstm_input = torch.cat((x_icd_out, x_disease_out), dim=1)
            lstm_out, _ = self.lstm(lstm_input)
            # lstm_out, _ = self.lstm(x_combined)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128

        elif args.model =='ICD and Drug':
            # MLP2 for x_icd and x_disease
            x_icd_out = self.mlp2_relu1(
                self.mlp2_bn1(self.mlp2_fc1(x_icd).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC1 -> BN -> ReLU
            x_icd_out = self.mlp2_relu2(
                self.mlp2_bn2(self.mlp2_fc2(x_icd_out).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC2 -> BN -> ReLU

            x_drug_out = self.mlp2_relu1(self.mlp2_bn1(self.mlp2_fc1(x_drugs).permute(0, 2, 1)).permute(0, 2, 1))
            x_drug_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_drug_out).permute(0, 2, 1)).permute(0, 2, 1))
            # Concatenate and pass through LSTM
            lstm_input = torch.cat((x_icd_out, x_drug_out), dim=1)
            lstm_out, _ = self.lstm(lstm_input)
            # lstm_out, _ = self.lstm(x_combined)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128
        elif args.model =='Disease and Drug':
            # MLP2 for x_icd and x_disease
            x_disease_out = self.mlp2_relu1(
                self.mlp2_bn1(self.mlp2_fc1(x_disease).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC1 -> BN -> ReLU
            x_disease_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_disease_out).permute(0, 2, 1)).permute(0, 2,1))  # Apply FC2 -> BN -> ReL

            x_drug_out = self.mlp2_relu1(self.mlp2_bn1(self.mlp2_fc1(x_drugs).permute(0, 2, 1)).permute(0, 2, 1))
            x_drug_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_drug_out).permute(0, 2, 1)).permute(0, 2, 1))
            # Concatenate and pass through LSTM
            lstm_input = torch.cat((x_disease_out, x_drug_out), dim=1)
            lstm_out, _ = self.lstm(lstm_input)
            # lstm_out, _ = self.lstm(x_combined)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128

        elif args.model =='Phenotypes and ICD and Disease':
            ## MLP1 for x_features
            x_features_out = self.mlp1_relu1(self.mlp1_bn1(self.mlp1_fc1(x_features)))  # Apply FC1 -> BN -> ReLU
            x_features_out = self.mlp1_relu2(self.mlp1_bn2(self.mlp1_fc2(x_features_out)))  # Apply FC2 -> BN -> ReLU
            x_features_out = x_features_out.unsqueeze(1)

            x_icd_out = self.mlp2_relu1(
                self.mlp2_bn1(self.mlp2_fc1(x_icd).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC1 -> BN -> ReLU
            x_icd_out = self.mlp2_relu2(
                self.mlp2_bn2(self.mlp2_fc2(x_icd_out).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC2 -> BN -> ReLU

            x_disease_out = self.mlp2_relu1(
                self.mlp2_bn1(self.mlp2_fc1(x_disease).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC1 -> BN -> ReLU
            x_disease_out = self.mlp2_relu2(
                self.mlp2_bn2(self.mlp2_fc2(x_disease_out).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC2 -> BN -> ReL

            # Concatenate and pass through LSTM
            lstm_input = torch.cat((x_features_out, x_icd_out, x_disease_out), dim=1)
            lstm_out, _ = self.lstm(lstm_input)
            # lstm_out, _ = self.lstm(x_combined)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128

        elif args.model =='Phenotypes and ICD and Drug':
            # Forward pass through the model
            # MLP1 for x_features
            x_features_out = self.mlp1_relu1(self.mlp1_bn1(self.mlp1_fc1(x_features)))  # Apply FC1 -> BN -> ReLU
            x_features_out = self.mlp1_relu2(self.mlp1_bn2(self.mlp1_fc2(x_features_out)))  # Apply FC2 -> BN -> ReLU
            x_features_out = x_features_out.unsqueeze(1)

            # MLP2 for x_icd and x_disease
            x_icd_out = self.mlp2_relu1(self.mlp2_bn1(self.mlp2_fc1(x_icd).permute(0,2,1)).permute(0,2,1))  # Apply FC1 -> BN -> ReLU
            x_icd_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_icd_out).permute(0,2,1)).permute(0,2,1))  # Apply FC2 -> BN -> ReLU

            # x_disease_out = self.mlp2_relu1(self.mlp2_bn1(self.mlp2_fc1(x_disease).permute(0,2,1)).permute(0,2,1))  # Apply FC1 -> BN -> ReLU
            # x_disease_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_disease_out).permute(0,2,1)).permute(0,2,1))  # Apply FC2 -> BN -> ReLU

            x_drug_out = self.mlp2_relu1(self.mlp2_bn1(self.mlp2_fc1(x_drugs).permute(0,2,1)).permute(0,2,1))
            x_drug_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_drug_out).permute(0, 2, 1)).permute(0, 2, 1))

            # Combine MLP2 outputs (concatenate along sequence length)
            lstm_input = torch.cat((x_icd_out, x_features_out,x_drug_out), dim=1)

            lstm_out, _ = self.lstm(lstm_input)
            # lstm_out, _ = self.lstm(x_combined)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128

        elif args.model == 'Phenotypes and Disease and Drug':
            # Forward pass through the model
            # MLP1 for x_features
            x_features_out = self.mlp1_relu1(self.mlp1_bn1(self.mlp1_fc1(x_features)))  # Apply FC1 -> BN -> ReLU
            x_features_out = self.mlp1_relu2(self.mlp1_bn2(self.mlp1_fc2(x_features_out)))  # Apply FC2 -> BN -> ReLU
            x_features_out = x_features_out.unsqueeze(1)

            # MLP2 for x_icd and x_disease
            # x_icd_out = self.mlp2_relu1(
            #     self.mlp2_bn1(self.mlp2_fc1(x_icd).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC1 -> BN -> ReLU
            # x_icd_out = self.mlp2_relu2(
            #     self.mlp2_bn2(self.mlp2_fc2(x_icd_out).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC2 -> BN -> ReLU

            x_disease_out = self.mlp2_relu1(self.mlp2_bn1(self.mlp2_fc1(x_disease).permute(0,2,1)).permute(0,2,1))  # Apply FC1 -> BN -> ReLU
            x_disease_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_disease_out).permute(0,2,1)).permute(0,2,1))  # Apply FC2 -> BN -> ReLU

            x_drug_out = self.mlp2_relu1(self.mlp2_bn1(self.mlp2_fc1(x_drugs).permute(0, 2, 1)).permute(0, 2, 1))
            x_drug_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_drug_out).permute(0, 2, 1)).permute(0, 2, 1))

            # Combine MLP2 outputs (concatenate along sequence length)
            lstm_input = torch.cat((x_disease_out, x_features_out, x_drug_out), dim=1)

            lstm_out, _ = self.lstm(lstm_input)
            # lstm_out, _ = self.lstm(x_combined)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128

        elif args.model == 'ICD and Disease and Drug':
            # Forward pass through the model
            # # MLP1 for x_features
            # x_features_out = self.mlp1_relu1(self.mlp1_bn1(self.mlp1_fc1(x_features)))  # Apply FC1 -> BN -> ReLU
            # x_features_out = self.mlp1_relu2(self.mlp1_bn2(self.mlp1_fc2(x_features_out)))  # Apply FC2 -> BN -> ReLU
            # x_features_out = x_features_out.unsqueeze(1)

            # MLP2 for x_icd and x_disease
            x_icd_out = self.mlp2_relu1(
                self.mlp2_bn1(self.mlp2_fc1(x_icd).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC1 -> BN -> ReLU
            x_icd_out = self.mlp2_relu2(
                self.mlp2_bn2(self.mlp2_fc2(x_icd_out).permute(0, 2, 1)).permute(0, 2, 1))  # Apply FC2 -> BN -> ReLU

            x_disease_out = self.mlp2_relu1(self.mlp2_bn1(self.mlp2_fc1(x_disease).permute(0,2,1)).permute(0,2,1))  # Apply FC1 -> BN -> ReLU
            x_disease_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_disease_out).permute(0,2,1)).permute(0,2,1))  # Apply FC2 -> BN -> ReLU

            x_drug_out = self.mlp2_relu1(self.mlp2_bn1(self.mlp2_fc1(x_drugs).permute(0, 2, 1)).permute(0, 2, 1))
            x_drug_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_drug_out).permute(0, 2, 1)).permute(0, 2, 1))

            # Combine MLP2 outputs (concatenate along sequence length)
            lstm_input = torch.cat((x_disease_out, x_icd_out, x_drug_out), dim=1)

            lstm_out, _ = self.lstm(lstm_input)
            # lstm_out, _ = self.lstm(x_combined)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128

        elif args.model =='All':
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

            x_drug_out = self.mlp2_relu1(self.mlp2_bn1(self.mlp2_fc1(x_drugs).permute(0,2,1)).permute(0,2,1))
            x_drug_out = self.mlp2_relu2(self.mlp2_bn2(self.mlp2_fc2(x_drug_out).permute(0, 2, 1)).permute(0, 2, 1))

            # Combine MLP2 outputs (concatenate along sequence length)
            x_combined = torch.cat((x_icd_out, x_disease_out,x_drug_out), dim=1)

            # Concatenate and pass through LSTM
            lstm_input = torch.cat((x_combined, x_features_out), dim=1)
            lstm_out, _ = self.lstm(lstm_input)
            # lstm_out, _ = self.lstm(x_combined)
            combine_feature_out = lstm_out[:, -1, :]  # Take the last LSTM output 128

        # MLP3 for x_sex
        x_sex_out = self.mlp3_relu1(self.mlp3_bn1(self.mlp3_fc1(x_sex)))  # Apply FC1 -> BN -> ReLU
        x_sex_out = self.mlp3_relu2(self.mlp3_bn2(self.mlp3_fc2(x_sex_out)))  # Apply FC2 -> BN -> ReLU
        x_sex_out = x_sex_out.squeeze(1)  #16

        # Concatenate LSTM output with MLP3 output
        x_combined = torch.cat((combine_feature_out, x_sex_out), dim=1)

        outputs = {}
        system_names = list(self.output_layers_1.keys())

        for i, system in enumerate(system_names):
            distance_weight = self.distance_weights[system]

            mask_last_col = system_mask[:, -1]  # (20000,)

            base_output = self.output_layers_1[system](x_combined)  # (20000, output_dim)

            adjusted_output = torch.zeros_like(base_output)

            mask_2 = (mask_last_col == 2)  # (20000, 1)
            adjusted_output[mask_2] = base_output[mask_2] * (1 + distance_weight * 1)

            mask_0_1 = (mask_last_col == 0) | (mask_last_col == 1)  # (20000, 1)
            adjusted_output[mask_0_1] = base_output[mask_0_1] * (
                        1 + 2 * distance_weight * system_mask[mask_0_1, i].unsqueeze(1))

            if args.learnable_weight == 'Yes':

                outputs[system] = self.ReLU_1(adjusted_output)
            elif args.learnable_weight == 'No':
                outputs[system] = self.ReLU_1(base_output)
        # Compute final outputs
        final_outputs = {}
        for system, output_layer in self.output_layers_2.items():
            final_output = output_layer(outputs[system])
            final_outputs[system] = self.ReLU_2(final_output)

        return final_outputs