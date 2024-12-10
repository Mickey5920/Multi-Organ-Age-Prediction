# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
def combined_loss(outputs, targets, system_mask, penalty_factor, margin=1.0, alpha=0.5):
    """
    组合损失函数，包含MSE损失和排序约束。

    参数：
    - outputs: dict，模型的输出，键为系统名称，值为预测值张量，形状为 [batch_size, 1]
    - targets: torch.Tensor，真实标签，形状为 [batch_size]
    - system_mask: torch.Tensor，系统掩码，形状为 [batch_size, 7]
    - margin: float，排序损失的margin
    - alpha: float，MSE损失的权重
    - penalty_factor: float，疾病系统损失的惩罚因子
    """
    # 获取系统名称列表
    system_names = list(outputs.keys())
    num_systems = len(system_names)

    # 将输出堆叠成一个张量，形状为 [batch_size, 7]
    preds = torch.stack([outputs[system].squeeze(1) for system in system_names], dim=1)  # [batch_size, 7]

    # 确保targets的形状为 [batch_size]
    if targets.dim() == 2 and targets.size(1) == 1:
        targets = targets.squeeze(1)  # [batch_size]
    elif targets.dim() != 1:
        raise ValueError(f"Targets should be 1D tensor, but got shape {targets.shape}")

    # 扩展targets以匹配preds的形状 [batch_size, 7]
    targets_expanded = targets.unsqueeze(1).expand_as(preds)  # [batch_size, 7]

    # 计算MSE损失
    mse_loss = nn.functional.mse_loss(preds, targets_expanded)

    # 计算排序损失：有疾病的系统预测值 > 无疾病的系统预测值 + margin
    # 1. 创建有疾病和无疾病的掩码
    disease_mask = system_mask == 1  # [batch_size, 7]
    normal_mask = system_mask == 0   # [batch_size, 7]

    # 2. 扩展掩码以进行系统对的比较
    # preds_disease: [batch_size, 7, 1]
    preds_disease = preds.unsqueeze(2)  # [batch_size, 7, 1]

    # preds_normal: [batch_size, 1, 7]
    preds_normal = preds.unsqueeze(1)   # [batch_size, 1, 7]

    # 3. 计算差值，公式为 margin + preds_normal - preds_disease
    # diff: [batch_size, 7, 7]
    diff = margin + preds_normal - preds_disease  # [batch_size, 7, 7]

    # 4. 创建排序损失的掩码，仅在有疾病的系统i和无疾病的系统j时为True
    # mask: [batch_size, 7, 7]
    mask = disease_mask.unsqueeze(2) & normal_mask.unsqueeze(1)  # [batch_size, 7, 7]

    # 5. 计算排序损失，使用ReLU替代clamp以实现max(0, x)
    # ranking_loss_matrix: [batch_size, 7, 7]
    ranking_loss_matrix = torch.relu(diff) * mask.float()  # [batch_size, 7, 7]

    # 6. 计算总的排序损失
    ranking_loss = ranking_loss_matrix.sum() * penalty_factor

    # 总损失
    total_loss = mse_loss * alpha + ranking_loss
    # total_loss = mse_loss
    return total_loss



import torch
import csv
import numpy as np
from sklearn.metrics import roc_auc_score
import math

def evaluate(model, data_loader, device, output_csv=None, is_test=False):
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

    # 用于计算C-index和AUC的数据收集
    # C-index: 统计患病系统数及其中预测年龄 > 实际年龄的系统数
    diseased_count = 0
    diseased_correct = 0

    # AUC: 所有系统的预测值和疾病状态
    all_preds_for_auc = []
    all_diseases_for_auc = []

    with torch.no_grad():
        for batch in data_loader:
            (
                batch_Participant_ID_data,
                batch_features,
                batch_icd_vecs,
                batch_disease_vecs,
                batch_sex_data,
                batch_system_mask,
                batch_Survival_Time,
                batch_labels
            ) = batch

            # 数据传输到设备
            batch_features = batch_features.to(device, dtype=torch.float32, non_blocking=True)
            batch_icd_vecs = batch_icd_vecs.to(device, dtype=torch.float32, non_blocking=True)
            batch_disease_vecs = batch_disease_vecs.to(device, dtype=torch.float32, non_blocking=True)
            batch_sex_data = batch_sex_data.to(device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
            batch_system_mask = batch_system_mask.to(device, dtype=torch.float32, non_blocking=True)
            batch_system_mask = batch_system_mask.squeeze(0)

            # 前向传播
            outputs = model(batch_icd_vecs, batch_disease_vecs, batch_features, batch_sex_data, batch_system_mask)
            batch_labels = batch_labels.to(dtype=torch.float32, device=device).squeeze()
            batch_system_mask = batch_system_mask.squeeze(0)

            # 收集每个系统的预测值和目标值
            for i, system in enumerate(system_metrics.keys()):
                preds = outputs[system].squeeze()  # (N,)
                targets = batch_labels[:, i] if batch_labels.dim() > 1 else batch_labels  # (N,)

                system_metrics[system]['preds'].append(preds)
                system_metrics[system]['targets'].append(targets)

                # 收集AUC需要的数据
                # Has_Disease：batch_system_mask[j, i] -> 1表示患病, 0表示健康
                # predicted_age: preds[j]
                # actual_age: targets[j]
                # 针对C-index和AUC, 我们需要逐样本处理
                for j in range(len(preds)):
                    predicted_age = preds[j].item()
                    actual_age = targets[j].item()
                    disease_status = batch_system_mask[j, i].item()

                    # 收集AUC数据
                    all_preds_for_auc.append(predicted_age)
                    all_diseases_for_auc.append(int(disease_status))

                    # 计算C-index需要的数据：如果是患病系统且predicted_age > actual_age则计数
                    if disease_status == 1:
                        diseased_count += 1
                        if predicted_age > actual_age:
                            diseased_correct += 1

                # 如果是测试过程，将信息收集到结果列表中
                if is_test:
                    batch_sex_data = batch_sex_data.squeeze()
                    for j in range(len(preds)):
                        result_row = {
                            'Participant_ID': batch_Participant_ID_data[j],
                            'Sex': batch_sex_data[j].item(),
                            'System': system,
                            'Actual_Age': targets[j].item(),
                            'Predicted_Age': preds[j].item(),
                            'Has_Disease': batch_system_mask[j, i].item()  # 标注是否有疾病
                        }
                        results_to_csv.append(result_row)

        # 计算系统级别的 RMSE, MAE 和 R2
        metrics = {}
        for system, values in system_metrics.items():
            # 将所有批次的预测和目标值连接在一起
            preds = torch.cat(values['preds']).to(device)
            targets = torch.cat(values['targets']).to(device)

            # 计算 RMSE, MAE 和 R2
            rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
            mae = torch.mean(torch.abs(preds - targets)).item()
            ss_total = torch.sum((targets - torch.mean(targets)) ** 2)
            ss_res = torch.sum((targets - preds) ** 2)
            r2 = 1 - (ss_res / ss_total).item() if ss_total != 0 else float('nan')

            metrics[system] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}

        # 计算 C-index
        # 若没有患病系统，则C-index为 nan
        if diseased_count > 0:
            c_index = diseased_correct / diseased_count
        else:
            c_index = float('nan')
        metrics['C-index'] = {'Value': c_index}

        # 计算 AUC
        # 需要有至少一个患病和一个健康样本，才可计算AUC
        unique_labels = set(all_diseases_for_auc)
        if len(unique_labels) > 1:
            try:
                auc = roc_auc_score(all_diseases_for_auc, all_preds_for_auc)
            except ValueError:
                auc = float('nan')
        else:
            auc = float('nan')
        metrics['AUC'] = {'Value': auc}

        # 打印所有指标
        for system, metric in metrics.items():
            if system in ['C-index', 'AUC']:
                print(f"{system}: {metric['Value']:.4f}")
            else:
                print(f"{system} - RMSE: {metric['RMSE']:.4f}, MAE: {metric['MAE']:.4f}, R2: {metric['R2']:.4f}")

        # 仅在测试阶段写入 CSV 文件
        if is_test and output_csv:
            with open(output_csv, mode='w', newline='') as csvfile:
                fieldnames = ['Participant_ID', 'Sex', 'System', 'Actual_Age', 'Predicted_Age', 'Has_Disease']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_to_csv)

            # 在CSV文件后面追加C-index和AUC等指标
            with open(output_csv, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([])  # 空行作为分隔
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
    """
    动态调整惩罚因子。

    Args:
        epoch: 当前训练轮次。
        max_epochs: 最大训练轮次。
        initial_factor: 初始惩罚因子。
        final_factor: 最终惩罚因子。

    Returns:
        当前轮次的惩罚因子。
    """
    return initial_factor - (initial_factor - final_factor) * (epoch / max_epochs)
