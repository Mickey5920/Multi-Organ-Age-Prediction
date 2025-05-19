# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.metrics import roc_auc_score
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  


def combined_loss(
        outputs,
        targets,
        system_mask,
        occurrence_next_time,
        penalty_factor,
        Loss,
        margin=1.0,
        alpha=0.5
):
    system_names = list(outputs.keys())

    preds = torch.stack([outputs[system].squeeze(1) for system in system_names], dim=1)

    if targets.dim() == 2 and targets.size(1) == 1:
        targets = targets.squeeze(1)
    elif targets.dim() != 1:
        raise ValueError(f"{targets.shape}")

    targets_expanded = targets.unsqueeze(1).expand_as(preds)
    mse_loss = nn.functional.mse_loss(preds, targets_expanded)
    system_mask = system_mask[:, :-1]
    disease_mask = system_mask == 1
    normal_mask = system_mask == 0

    preds_disease = preds.unsqueeze(2)
    preds_normal = preds.unsqueeze(1)
    diff = margin + preds_normal - preds_disease

    mask = disease_mask.unsqueeze(2) & normal_mask.unsqueeze(1)  # [batch_size, 7, 7]

    ranking_loss_matrix = torch.relu(diff) * mask.float()  # [batch_size, 7, 7]

    ranking_loss = ranking_loss_matrix.sum() * penalty_factor

    if Loss == 'RankingLoss':
        total_loss = ranking_loss
    elif Loss == 'MSELoss':
        total_loss = mse_loss
    elif Loss == 'All':
        total_loss = mse_loss* alpha  + ranking_loss

    return total_loss

def evaluate(model, data_loader, device,args, output_csv=None, is_test=False):
    model.eval()
    system_metrics = {
        'cardiovascular': {'preds': [], 'targets': []},
        'pulmonary': {'preds': [], 'targets': []},
        'musculoskeletal': {'preds': [], 'targets': []},
        'immune': {'preds': [], 'targets': []},
        'renal': {'preds': [], 'targets': []},
        'hepatic': {'preds': [], 'targets': []},
        'metabolic': {'preds': [], 'targets': []}
    }

    results_to_csv = [] if is_test else None

    diseased_count = 0
    diseased_correct = 0

    all_preds_for_auc = []
    all_diseases_for_auc = []

    with torch.no_grad():
        for batch in data_loader:
            (
                batch_Participant_ID_data,
                batch_features,
                batch_icd_vecs,
                batch_drug_vecs,
                batch_disease_vecs,
                batch_sex_data,
                batch_system_mask,
                occurrence_Next_Time,
                batch_Survival_Time,
                batch_labels
            ) = batch

            batch_features = batch_features.to(device, dtype=torch.float32, non_blocking=True)
            batch_icd_vecs = batch_icd_vecs.to(device, dtype=torch.float32, non_blocking=True)
            batch_disease_vecs = batch_disease_vecs.to(device, dtype=torch.float32, non_blocking=True)
            batch_sex_data = batch_sex_data.to(device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
            batch_system_mask = batch_system_mask.to(device, dtype=torch.float32, non_blocking=True)
            batch_system_mask = batch_system_mask.squeeze(0)
            batch_drug_vecs = batch_drug_vecs.to(device, dtype=torch.float32, non_blocking=True)

            outputs = model(batch_icd_vecs, batch_disease_vecs, batch_drug_vecs,batch_features, batch_sex_data, batch_system_mask,args)
            batch_labels = batch_labels.to(dtype=torch.float32, device=device).squeeze()
            batch_system_mask = batch_system_mask.squeeze(0)

            for i, system in enumerate(system_metrics.keys()):
                preds = outputs[system].squeeze()  # (N,)
                targets = batch_labels[:, i] if batch_labels.dim() > 1 else batch_labels  # (N,)

                system_metrics[system]['preds'].append(preds)
                system_metrics[system]['targets'].append(targets)

                for j in range(len(preds)):
                    predicted_age = preds[j].item()
                    actual_age = targets[j].item()
                    disease_status = batch_system_mask[j, i].item()
                    all_preds_for_auc.append(predicted_age)
                    all_diseases_for_auc.append(int(disease_status))
                    if disease_status == 1:
                        diseased_count += 1
                        if predicted_age > actual_age:
                            diseased_correct += 1

                if is_test:
                    batch_sex_data = batch_sex_data.squeeze()
                    for j in range(len(preds)):
                        actual_age = targets[j].item()
                        predicted_age = preds[j].item()
                        age_gap = predicted_age - actual_age

                        result_row = {
                            'Participant_ID': batch_Participant_ID_data[j],
                            'Sex': batch_sex_data[j].item(),
                            'System': system,
                            'Actual_Age': actual_age,
                            'Predicted_Age': predicted_age,
                            'Age Gap': age_gap,
                            'Has_Disease': batch_system_mask[j, i].item()
                        }
                        results_to_csv.append(result_row)

    metrics = {}
    for system, values in system_metrics.items():

        preds = torch.cat(values['preds']).to(device)
        targets = torch.cat(values['targets']).to(device)

        rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
        mae = torch.mean(torch.abs(preds - targets)).item()
        ss_total = torch.sum((targets - torch.mean(targets)) ** 2)
        ss_res = torch.sum((targets - preds) ** 2)
        r2 = 1 - (ss_res / ss_total).item() if ss_total != 0 else float('nan')

        metrics[system] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}

    if diseased_count > 0:
        c_index = diseased_correct / diseased_count
    else:
        c_index = float('nan')
    metrics['C-index'] = {'Value': c_index}

    unique_labels = set(all_diseases_for_auc)
    if len(unique_labels) > 1:
        try:
            auc = roc_auc_score(all_diseases_for_auc, all_preds_for_auc)
        except ValueError:
            auc = float('nan')
    else:
        auc = float('nan')
    metrics['AUC'] = {'Value': auc}


    for system, metric in metrics.items():
        if system in ['C-index', 'AUC']:
            print(f"{system}: {metric['Value']:.4f}")
        else:
            print(f"{system} - RMSE: {metric['RMSE']:.4f}, MAE: {metric['MAE']:.4f}, R2: {metric['R2']:.4f}")

    if is_test and output_csv:
        with open(output_csv, mode='w', newline='') as csvfile:
            fieldnames = ['Participant_ID', 'Sex', 'System', 'Actual_Age', 'Predicted_Age', 'Age Gap', 'Has_Disease']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_to_csv)

        with open(output_csv, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([])
            writer.writerow(['Metric', 'Value'])
            for system, metric in metrics.items():
                if system in ['C-index', 'AUC']:
                    writer.writerow([system, metric['Value']])
                else:
                    writer.writerow([f"{system} - RMSE", metric['RMSE']])
                    writer.writerow([f"{system} - MAE", metric['MAE']])
                    writer.writerow([f"{system} - R2", metric['R2']])

    return metrics


def get_penalty_factor(epoch, max_epochs, initial_factor, final_factor):

    return initial_factor - (initial_factor - final_factor) * (epoch / max_epochs)
