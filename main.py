# main.py
import datetime
from lifelines import CoxPHFitter
from tqdm import tqdm
import argparse
import torch.nn as nn
# 导入sklearn库中的模块用于数据标准化和数据集划分
from sklearn.preprocessing import StandardScaler
# 导入open_clip库，用于处理ICD代码和疾病描述的文本编码
import open_clip
import json
import matplotlib.pyplot as plt
import seaborn as sns
# 导入tqdm库，用于显示训练进度条
from tqdm import tqdm
# 导入time库，用于记录时间
# 导入numpy，用于数值计算
import numpy as np
from torch.utils.data import DataLoader, Dataset,random_split,ConcatDataset,RandomSampler
from pandas.api.types import is_numeric_dtype
import csv
import gc
import os
import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd
import ast

import torch.optim as optim
import time
# 导入自定义模块
from Dataloader import CustomDataset,IterableCustomDataset,HDF5Dataset
from train import combined_loss, evaluate, get_penalty_factor
from model import AgePredictionModel
from Dataloader import encode_texts, process_batch, fit_pca_models,get_system_mask_code,BioAgeDataset
def parse_args():
    parser = argparse.ArgumentParser(description='Age Prediction Model Training')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for data files')
    parser.add_argument('--icd_file_path', type=str, default='./ICD_code_descriptions/X_icd_codes_vec_Instance_0_mask_system_Phenotypes_time_all_1205.h5', help='Path to ICD vectors HDF5 file')
    parser.add_argument('--desc_file_path', type=str, default='./ICD_code_descriptions/X_disease_desc_Instance_0_mask_system_Phenotypes_time_all_1205.h5', help='Path to disease descriptions HDF5 file')
    parser.add_argument('--dataset_file', type=str, default='./data/custom_dataset_mask_system_Phenotypes_time_all_1128.pt', help='Path to save/load the dataset')
    parser.add_argument('--batch_size', type=int, default=20000, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--initial_factor', type=float, default=5.0, help='Initial penalty factor')
    parser.add_argument('--final_factor', type=float, default=2.0, help='Final penalty factor')
    parser.add_argument('--npz_data_dir',type=str,default='./Phenotypes_time_npz_50000',help='npz data storage')
    parser.add_argument('--model_save_path', type=str, default='./checkpoint/model_checkpoint_Phenotypes_time_npz_all_1204.pt', help='Path to save the trained model')
    parser.add_argument('--output_dir', type=str, default='./test', help='Directory to save test outputs')
    # Add more arguments as needed
    return parser.parse_args()

def collate_fn(batch):
    # 因为每个样本已经是一个批次，所以直接返回第一个元素
    return batch[0]


import numpy as np
import h5py
import torch

def save_batches_to_npz(Participant_ID, features, icd_file_paths, desc_file_paths, sex, mask_system_data, Survival_Time,labels, batch_size=20000,npz_data_dir='data_npz'):
    """
    Saves data in batches to .npz files.

    Args:
        Participant_ID (array-like): Array of Participant IDs.
        features (array-like): Array of features.
        icd_file_paths (str): Path to the ICD data file.
        desc_file_paths (str): Path to the disease description data file.
        sex (array-like): Array of sex data.
        mask_system_data (array-like): Array of system mask data.
        labels (array-like): Array of labels.
        batch_size (int, optional): Number of samples per batch. Defaults to 20000.
    """
    total_samples = len(Participant_ID)
    num_batches = (total_samples + batch_size - 1) // batch_size  # Calculate number of batches

    # Ensure the output directory exists
    output_dir = npz_data_dir
    os.makedirs(output_dir, exist_ok=True)

    for batch_num in tqdm(range(num_batches), desc="Saving batches"):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_samples)

        # Get the batch data
        batch_Participant_ID = Participant_ID[start_idx:end_idx]
        batch_features = features[start_idx:end_idx]
        batch_Survival_Time = Survival_Time[start_idx:end_idx]
        batch_sex = sex[start_idx:end_idx]
        batch_mask_system_data = mask_system_data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        # Check if the batch is empty (should not be, but just in case)
        if len(batch_Participant_ID) == 0:
            print(f"Batch {batch_num} is empty, skipping.")
            continue

        # Replace '-' with '.' in Participant_ID and keep it as a string
        batch_Participant_ID = np.array([str(id).replace('-', '.') for id in batch_Participant_ID], dtype='S10')

        # Ensure sex is of float32 type
        batch_sex = np.array(batch_sex, dtype=np.float32)

        # Apply get_system_mask_code to batch_mask_system_data
        try:
            # Assuming get_system_mask_code is a function that processes each batch element
            # and returns a tensor or array-like object
            processed_masks = []
            for codes in batch_mask_system_data:
                processed_mask = get_system_mask_code(codes)
                # Ensure the processed_mask is a NumPy array
                if isinstance(processed_mask, torch.Tensor):
                    processed_mask = processed_mask.cpu().numpy()
                processed_masks.append(processed_mask)
            batch_mask_system_data = np.array(processed_masks, dtype=np.float32)
        except Exception as e:
            print(f"Error processing mask system data in batch {batch_num}: {e}")
            continue

        # Ensure features and labels are of float32 type
        try:
            batch_features = np.array(batch_features, dtype=np.float32)
            batch_Survival_Time = np.array(batch_Survival_Time,dtype=np.float32)
            batch_labels = np.array(batch_labels, dtype=np.float32)
        except Exception as e:
            print(f"Error converting features or labels to float32 in batch {batch_num}: {e}")
            continue

        # Read ICD and disease data from files
        try:
            with h5py.File(icd_file_paths, 'r') as icd_file:
                if 'data' not in icd_file:
                    raise KeyError(f"'data' key not found in ICD file {icd_file_paths}")
                icd_data = np.array(icd_file['data'][start_idx:end_idx], dtype=np.float32)
                batch_icd_vec = icd_data
        except Exception as e:
            print(f"Error reading ICD file in batch {batch_num}: {e}")
            continue

        try:
            with h5py.File(desc_file_paths, 'r') as desc_file:
                if 'data' not in desc_file:
                    raise KeyError(f"'data' key not found in description file {desc_file_paths}")
                disease_data = np.array(desc_file['data'][start_idx:end_idx], dtype=np.float32)
                batch_disease_vec = disease_data
        except Exception as e:
            print(f"Error reading description file in batch {batch_num}: {e}")
            continue

        # Verify that all arrays have the same number of samples
        batch_size_actual = len(batch_Participant_ID)
        if not (len(batch_features) == len(batch_Survival_Time) == len(batch_icd_vec) == len(batch_disease_vec) == len(batch_sex) == len(batch_mask_system_data) == len(batch_labels) == batch_size_actual):
            print(f"Batch {batch_num} has inconsistent sizes, skipping.")
            continue

        # Save the batch to .npz file
        npz_path = os.path.join(output_dir, f'batch_{batch_num}.npz')
        try:
            np.savez(npz_path,
                     Participant_ID=batch_Participant_ID,
                     features=batch_features,
                     icd_vec=batch_icd_vec,
                     disease_vec=batch_disease_vec,
                     sex_data=batch_sex,
                     system_mask=batch_mask_system_data,
                     Survival_Time = batch_Survival_Time,
                     label=batch_labels)
            print(f"Batch {batch_num} saved to {npz_path}.")
        except Exception as e:
            print(f"Error saving batch {batch_num} to .npz: {e}")
# Function to create data loaders
def get_data_loaders(data_dir, batch_size=1, val_size=0.1, test_size=0.1):
    # Create dataset from .npz files in data_dir
    dataset = BioAgeDataset(data_dir)

    # Calculate the sizes for each split
    total_size = len(dataset)
    val_size = int(total_size * val_size)
    test_size = int(total_size * test_size)
    train_size = total_size - val_size - test_size

    # Split the dataset into train, validation, and test subsets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders for each subset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create a data loader for the entire dataset (all_loader)
    all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, all_loader

def main():
    args = parse_args()
    # 加载csv文件中的数据
    print("-----------------读取csv数据-----------------")
    df1 = pd.read_csv(r'E:\MPU\Project\Dataset-0911\Process\data\3-combined_all_phenotypes_icd_time_age_all_1203.csv')
    # participant_id = df1['Participant ID']
    df_ICD_Date = pd.read_csv(r'E:\MPU\Project\Dataset-0911\Date\Date_ICD_10.csv', low_memory=False)  # ICD代码、疾病记录的时间
    df_Phenotype = pd.read_csv(r'E:\MPU\Project\Dataset-0911\Process\Phenotypes-Date.csv')  # 表型代码测量时间
    df_ICD_Date_time = pd.merge(df_ICD_Date, df_Phenotype, on='Participant ID', how='left')
    del df_ICD_Date
    del df_Phenotype
    '''
    读取ICD的记录时间
    '''
    # 定义基准时间列
    baseline_col = 'Alanine aminotransferase assay date | Instance 0'
    # 将基准时间列转换为 datetime 格式
    df_ICD_Date_time[baseline_col] = pd.to_datetime(df_ICD_Date_time[baseline_col], errors='coerce')

    # 初始化一个新的 DataFrame 用于存储时间差结果
    df_ICD_Time_Residual = pd.DataFrame()
    df_ICD_Time_Residual['Participant ID'] = df_ICD_Date_time['Participant ID']
    # 创建一个字典来存储所有时间差结果
    time_diff_dict = {}

    # 遍历所有其他列，计算时间差
    print("-----------------计算时间差-----------------")
    for column in tqdm(df_ICD_Date_time.columns):
        if column != 'Participant ID' and column != baseline_col:
            # 将当前列的日期转换为 datetime 格式
            df_ICD_Date_time[column] = pd.to_datetime(df_ICD_Date_time[column], format='%Y/%m/%d', errors='coerce')
            # 计算时间差，结果为天数
            time_diff = ((df_ICD_Date_time[column] - df_ICD_Date_time[baseline_col]).dt.days) / 365.25
            # 将时间差存入字典
            time_diff_dict[column] = time_diff

    # 将字典转换为 DataFrame
    time_diff_df = pd.DataFrame(time_diff_dict)

    # 使用 pd.concat 一次性将新列添加到 df_ICD_Time_Residual
    df_ICD_Time_Residual = pd.concat([df_ICD_Time_Residual, time_diff_df], axis=1)
    del time_diff_df
    del time_diff_dict
    del df_ICD_Date_time
    # 创建副本以去碎片化
    df_ICD_Time_Residual = df_ICD_Time_Residual.copy()
    # print('df_ICD_Time_Residual_count', df_ICD_Time_Residual.count())

    '''
    将小于等于20的数据进行补全，使用表型列的平均值进行补全
    '''
    print('-----------------读取缺失数据，进行填补-----------------')
    # 文件名
    csv_file_name = './missing_data/processed_missing_Phenotypes_time_all_1204.csv'
    # 检查 CSV 文件是否存在
    if os.path.exists(csv_file_name):
        # 如果存在，直接读取
        df_merge = pd.read_csv(csv_file_name)
    else:
        # 如果不存在，处理数据
        # 计算每个数值列的平均值
        mean_values = df1.iloc[:, 10:88].mean()

        # 定义一个填补缺失值的函数
        def fill_missing(row):
            missing_count = row[10:88].isnull().sum()
            if 0 < missing_count <= 20:
                numeric_cols = [col for col in row.index[10:88] if is_numeric_dtype(df1[col])]
                for col in numeric_cols:
                    if pd.isnull(row[col]):
                        row[col] = mean_values[col]  # 用预计算的平均值填补
                # row['Missing data'] = 1
            return row

        # 使用 tqdm 显示进度
        tqdm.pandas(desc="Processing rows")
        df_merge = df1.progress_apply(fill_missing, axis=1)

        # 保存处理后的数据到 CSV 文件
        df_merge.to_csv(csv_file_name, index=False)
    del df1
    # 2. 在 df_merge 中创建一个新的列 'Base Participant ID'，提取基准 ID（去除 '-' 及其后的部分）
    df_merge['Base Participant ID'] = df_merge['Participant ID'].str.split('-').str[0]

    # 3. 将 'Base Participant ID' 转换为整数类型
    # 首先，确保所有基准 ID 都是有效的数字
    # 如果有无效的值（例如空值或非数字），可以选择填充或删除
    df_merge['Base Participant ID'] = pd.to_numeric(df_merge['Base Participant ID'], errors='coerce')

    # 检查是否有转换失败的值
    if df_merge['Base Participant ID'].isnull().any():
        print("警告：存在无法转换为整数的基准 Participant ID。将这些行删除或填充。")
        # 例如，删除这些行
        df_merge = df_merge.dropna(subset=['Base Participant ID'])
        # 或者填充默认值，例如 0
        # df_merge['Base Participant ID'] = df_merge['Base Participant ID'].fillna(0).astype(int)

    # 将 'Base Participant ID' 转换为整数类型
    df_merge['Base Participant ID'] = df_merge['Base Participant ID'].astype(int)

    # 4. 确保 df_ICD_Time_Residual 中的 'Participant ID' 也是整数类型
    # 如果 'Participant ID' 已经是整数类型，可以跳过此步骤
    # 否则，进行转换
    df_ICD_Time_Residual['Participant ID'] = pd.to_numeric(df_ICD_Time_Residual['Participant ID'], errors='coerce')

    # 检查是否有转换失败的值
    if df_ICD_Time_Residual['Participant ID'].isnull().any():
        print("警告：df_ICD_Time_Residual 中存在无法转换为整数的 Participant ID。将这些行删除或填充。")
        # 例如，删除这些行
        df_ICD_Time_Residual = df_ICD_Time_Residual.dropna(subset=['Participant ID'])
        # 或者填充默认值，例如 0
        # df_ICD_Time_Residual['Participant ID'] = df_ICD_Time_Residual['Participant ID'].fillna(0).astype(int)

    # 将 'Participant ID' 转换为整数类型
    df_ICD_Time_Residual['Participant ID'] = df_ICD_Time_Residual['Participant ID'].astype(int)

    # 5. 执行合并操作
    df_all = pd.merge(
        df_merge,
        df_ICD_Time_Residual,
        left_on='Base Participant ID',
        right_on='Participant ID',
        how='inner',  # 根据需要选择 'left', 'right', 'outer'
        suffixes=('_merge', '_ICD')
    )
    del df_ICD_Time_Residual
    # 6. 清理合并后的数据
    # 删除不需要的列，例如 'Base Participant ID' 和重复的 'Participant ID_ICD'
    df_all = df_all.drop(columns=['Base Participant ID', 'Participant ID_ICD'])

    # 如果需要，可以重命名 'Participant ID_merge' 回 'Participant ID'
    df_all = df_all.rename(columns={'Participant ID_merge': 'Participant ID'})
    print("-----------------删除空值行-----------------")
    # # 删除前83列中包含空值的行
    df_all = df_all.dropna(subset=df_all.columns[6:88])
    # 保留Participant_ID中含有'-1'的行
    # df_all = df_all[df_all['Participant ID'].str.contains('-1')]

    del df_merge
    # 选择前100行
    # df_all = df_all.head(5000)

    # 选择特征列并将其赋值给X_features
    X_features = df_all[['Sex'] + [
        "Participant ID", "Hand grip strength (average) | Instance 0", "Waist circumference | Instance 0",
        "Hip circumference | Instance 0", "Standing height | Instance 0",
        "Pulse rate, automated reading | Instance 0 | Array 0",
        "Forced vital capacity (FVC) | Instance 0 | Array 0",
        "Forced expiratory volume in 1-second (FEV1) | Instance 0 | Array 0",
        "Peak expiratory flow (PEF) | Instance 0 | Array 0",
        "Diastolic blood pressure, automated reading | Instance 0 | Array 0",
        "Systolic blood pressure, automated reading | Instance 0 | Array 0",
        "Body mass index (BMI) | Instance 0", "Weight | Instance 0(participant - p21002_i0)",
        "White blood cell (leukocyte) count | Instance 0",
        "Red blood cell (erythrocyte) count | Instance 0",
        "Haemoglobin concentration | Instance 0", "Haematocrit percentage | Instance 0",
        "Mean corpuscular volume | Instance 0", "Mean corpuscular haemoglobin | Instance 0",
        "Mean corpuscular haemoglobin concentration | Instance 0",
        "Red blood cell (erythrocyte) distribution width | Instance 0",
        "Platelet count | Instance 0", "Platelet crit | Instance 0",
        "Mean platelet (thrombocyte) volume | Instance 0",
        "Platelet distribution width | Instance 0", "Lymphocyte count | Instance 0",
        "Monocyte count | Instance 0", "Neutrophill count | Instance 0",
        "Eosinophill count | Instance 0", "Basophill count | Instance 0",
        "Nucleated red blood cell count | Instance 0", "Lymphocyte percentage | Instance 0",
        "Monocyte percentage | Instance 0", "Neutrophill percentage | Instance 0",
        "Eosinophill percentage | Instance 0", "Basophill percentage | Instance 0",
        "Nucleated red blood cell percentage | Instance 0", "Reticulocyte percentage | Instance 0",
        "Reticulocyte count | Instance 0", "Mean reticulocyte volume | Instance 0",
        "Mean sphered cell volume | Instance 0", "Immature reticulocyte fraction | Instance 0",
        "High light scatter reticulocyte percentage | Instance 0",
        "High light scatter reticulocyte count | Instance 0",
        "Creatinine (enzymatic) in urine | Instance 0", "Potassium in urine | Instance 0",
        "Sodium in urine | Instance 0", "Albumin | Instance 0(participant - p30600_i0)",
        "Alkaline phosphatase | Instance 0", "Alanine aminotransferase | Instance 0",
        "Apolipoprotein A | Instance 0", "Apolipoprotein B | Instance 0",
        "Aspartate aminotransferase | Instance 0", "Direct bilirubin | Instance 0",
        "Urea | Instance 0", "Calcium | Instance 0", "Cholesterol | Instance 0",
        "Creatinine | Instance 0", "C-reactive protein | Instance 0",
        "Cystatin C | Instance 0", "Gamma glutamyltransferase | Instance 0",
        "Glucose | Instance 0", "Glycated haemoglobin (HbA1c) | Instance 0",
        "HDL cholesterol | Instance 0", "IGF-1 | Instance 0", "LDL direct | Instance 0",
        "Lipoprotein A | Instance 0", "Phosphate | Instance 0", "SHBG | Instance 0",
        "Total bilirubin | Instance 0", "Testosterone | Instance 0", "Total protein | Instance 0",
        "Triglycerides | Instance 0", "Urate | Instance 0", "Vitamin D | Instance 0",
        "FEV1-FVC ratio | Instance 0", "Waist-hip circumference ratio | Instance 0",
        "Ankle spacing width | Instance 0", "Heel bone mineral density (BMD) | Instance 0",
        "Age of Instance 0",  # 不需要instance1共计81个
        "Ischemic_Heart_Disease", "Hypertension", "Stroke", "COPD", "CKD",  # 相关10类疾病
        "Diabetes", "Cirrhosis", "Osteoarthritis", "Osteoporosis", "Cancer"
    ]].copy()
    # 选择ICD10代码列并复制
    X_icd_codes = df_all['ICD'].copy()  # 替换为正确的列名
    ICD_Time = df_all.iloc[:, 177:-1]
    Survival_Time = df_all.iloc[:, 121:177]
    # 使用df_ICD_Time_Residual与X
    # 选择疾病描述列并复制
    X_disease_desc = df_all['Disease_Description_Feature_x'].copy()  # 替换为正确的列名

    # 目标变量为年龄
    y = df_all['Actual_Age'].copy()
    mask_system = df_all['ICD'].copy()
    # 标准化数值型特征
    print("-----------------标准化Phenotypes特征-----------------")
    # scaler = StandardScaler()
    # X_features.iloc[:, 2:80] = scaler.fit_transform(X_features.iloc[:, 2:80])
    Phenotypes = X_features.iloc[:, 2:80]
    Participant_IDs = df_all['Participant ID'].copy()
    # 将性别转换为数值表示，Female为0，Male为1
    Sex = X_features['Sex'].map({'Female': 0, 'Male': 1})
    # Assuming your dataframe is named df
    if Sex.isna().any().any():
        print("Sex contains NaN values")
    if Phenotypes.isna().any().any():
        print("Phenotypes contains NaN values")
    if mask_system.isna().any().any():
        print("mask_system contains NaN values")
    if X_icd_codes.isna().any().any():
        print("X_icd_codes contains NaN values")
    if y.isna().any().any():
        print("y contains NaN values")
    if X_disease_desc.isna().any().any():
        print("X_disease_desc contains NaN values")
    if Participant_IDs.isna().any().any():
        print("Participant_IDs contains NaN values")
    if df_all.isna().any().any():
        print("df_all contains NaN values")
    # 这里放置您的主要代码，包括数据加载、预处理、模型训练和评估
    # 请确保调整文件路径和变量名以匹配您的实际情况
    # 获取嵌入维度
    with torch.no_grad():
        sample_text = ["sample"]
        sample_features = encode_texts(sample_text)
        embedding_dim = sample_features.shape[-1]
        print(f"初始嵌入维度: {embedding_dim}")  # 应该是 512 或其他

    # 临时文件路径
    temp_icd_file_path = './ICD_code_descriptions/temp_X_icd_codes_1125.h5'
    temp_desc_file_path = './ICD_code_descriptions/temp_X_disease_desc_1125.h5'

    # 检查 HDF5 文件是否存在
    if os.path.exists(args.icd_file_path) and os.path.exists(args.desc_file_path):
        print("-----------------加载已保存的ICD代码和疾病描述向量数据-----------------")

        # 加载或创建 CustomDataset
        if os.path.exists(args.dataset_file):
            print("-----------------加载已保存的数据集-----------------")
            dataset = torch.load(args.dataset_file)
        else:
            print("-----------------没有找到保存的文件，开始计算并保存-----------------")
            # 数据预处理：将 mask_system 从字符串转换为列表
            parsed_mask_system_data = []
            for idx, codes_str in enumerate(mask_system):
                if isinstance(codes_str, str):
                    # 去掉方括号
                    codes_str = codes_str.strip("[]")
                    # 按逗号分割
                    codes = codes_str.split(',')
                    # 去除多余的引号和空格
                    codes_list = [code.strip().strip("'").strip('"') for code in codes]
                else:
                    codes_list = codes_str
                parsed_mask_system_data.append(codes_list)

            # 创建 BatchedDataset 对象


            # Check if we should save batches
            if len([f for f in os.listdir(args.npz_data_dir) if f.endswith('.npz')]) < int(1325056/args.batch_size):
                # If there are fewer than 133 files, call save_batches_to_npz
                print("Less than 133 .npz files found, saving new batches...")
                save_batches_to_npz(Participant_IDs.values,
                                    Phenotypes.values,
                                    args.icd_file_path,
                                    args.desc_file_path,
                                    Sex.values,
                                    pd.Series(parsed_mask_system_data),
                                    Survival_Time.values,
                                    labels=y.values,
                                    batch_size=args.batch_size,
                                    npz_data_dir = args.npz_data_dir)
            else:
                # Otherwise, skip saving the batches
                print(f"Already {len(os.listdir(args.npz_data_dir))} .npz files found, skipping batch saving.")

            # 保存 dataset
            # torch.save(train_dataset, dataset_file)
            print("数据集已保存到", args.dataset_file)
    else:
        print("-----------------未找到保存的向量数据，开始计算编码-----------------")

        # 创建临时 HDF5 文件用于编码数据
        with h5py.File(temp_icd_file_path, 'w') as f_icd_temp, h5py.File(temp_desc_file_path, 'w') as f_desc_temp:
            # 创建空的数据集，特征维度为 embedding_dim + 2（时间向量）
            f_icd_temp.create_dataset('data', shape=(0, 15, embedding_dim + 2),
                                      maxshape=(None, 15, embedding_dim + 2),
                                      dtype='float32', chunks=True)
            f_desc_temp.create_dataset('data', shape=(0, 15, embedding_dim + 2),
                                       maxshape=(None, 15, embedding_dim + 2),
                                       dtype='float32', chunks=True)
            print(f"初始化临时 HDF5 文件，形状: (0, 15, {embedding_dim + 2})")

        # 进行编码并保存到临时 HDF5 文件
        batch_size = 100000  # 根据显存大小调整
        total_samples = len(X_icd_codes)
        num_batches = (total_samples + batch_size - 1) // batch_size

        print("-----------------开始编码数据并保存到临时 HDF5 文件-----------------")
        with h5py.File(temp_icd_file_path, 'a') as f_icd_temp, h5py.File(temp_desc_file_path, 'a') as f_desc_temp:
            dset_icd_temp = f_icd_temp['data']
            dset_desc_temp = f_desc_temp['data']

            for batch_num in tqdm(range(num_batches), desc='编码保存数据'):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, total_samples)
                batch_icd_codes = X_icd_codes.iloc[start_idx:end_idx].tolist()
                batch_disease_desc = X_disease_desc.iloc[start_idx:end_idx].tolist()
                batch_icd_time = ICD_Time.iloc[start_idx:end_idx].values

                # 批量处理
                icd_features, desc_features = process_batch(batch_icd_codes, batch_disease_desc, batch_icd_time,max_len=15)

                # 打印批次信息
                print(
                    f"Batch {batch_num + 1}/{num_batches} - ICD: {icd_features.shape}, Description: {desc_features.shape}")

                # 保存到临时 HDF5 文件
                try:
                    # 保存 ICD 向量
                    dset_icd_temp.resize(dset_icd_temp.shape[0] + icd_features.shape[0], axis=0)
                    dset_icd_temp[-icd_features.shape[0]:] = icd_features

                    # 保存疾病描述向量
                    dset_desc_temp.resize(dset_desc_temp.shape[0] + desc_features.shape[0], axis=0)
                    dset_desc_temp[-desc_features.shape[0]:] = desc_features

                    print(f"Batch {batch_num + 1} 已保存到临时文件")
                except Exception as e:
                    print(f"保存Batch {batch_num + 1}到临时文件时出错: {e}")

                # **释放内存**
                del icd_features, desc_features
                gc.collect()
                torch.cuda.empty_cache()

        print("-----------------所有数据已成功编码并保存到临时 HDF5 文件中-----------------")

        # 拟合 PCA 模型
        print("-----------------开始拟合 PCA 模型-----------------")
        pca_icd, pca_desc = fit_pca_models(temp_icd_file_path, temp_desc_file_path, target_dim=514, batch_size=1000)
        args.desc_file_path
        # 初始化最终 HDF5 文件（清空之前的数据）
        with h5py.File(args.icd_file_path, 'w') as f_icd, h5py.File(args.desc_file_path, 'w') as f_desc:
            # 创建数据集，初始为空，形状为 (0, 15, 514)
            f_icd.create_dataset('data', shape=(0, 15, 514), maxshape=(None, 15, 514),
                                 dtype='float32', chunks=True)
            f_desc.create_dataset('data', shape=(0, 15, 514), maxshape=(None, 15, 514),
                                  dtype='float32', chunks=True)
            print(f"初始化最终 HDF5 文件，形状: (0, 15, 514)")

        # 重新遍历临时文件，应用 PCA 并保存到最终 HDF5 文件
        print("-----------------开始应用 PCA 并保存数据到最终 HDF5 文件-----------------")
        with h5py.File(temp_icd_file_path, 'r') as f_icd_temp, h5py.File(temp_desc_file_path, 'r') as f_desc_temp, \
                h5py.File(args.icd_file_path, 'a') as f_icd_final, h5py.File(args.desc_file_path, 'a') as f_desc_final:

            data_icd_temp = f_icd_temp['data']
            data_desc_temp = f_desc_temp['data']

            dset_icd_final = f_icd_final['data']
            dset_desc_final = f_desc_final['data']

            for start_idx in tqdm(range(0, data_icd_temp.shape[0], batch_size), desc="Applying PCA and Saving"):
                end_idx = min(start_idx + batch_size, data_icd_temp.shape[0])
                batch_icd = data_icd_temp[start_idx:end_idx, :, :]  # (batch_size, 15, feature_dim + 2)
                batch_desc = data_desc_temp[start_idx:end_idx, :, :]  # (batch_size, 15, feature_dim + 2)

                # Reshape to (batch_size*15, feature_dim + 2) for PCA
                batch_icd_flat = batch_icd.reshape(-1, batch_icd.shape[2])
                batch_desc_flat = batch_desc.reshape(-1, batch_desc.shape[2])

                # 应用 PCA
                icd_pca = pca_icd.transform(batch_icd_flat)
                desc_pca = pca_desc.transform(batch_desc_flat)

                # 重新组织为 (batch_size, 15, 514)
                icd_pca = icd_pca.reshape(-1, 15, 514)
                desc_pca = desc_pca.reshape(-1, 15, 514)

                # 打印批次信息
                print(f"Batch {start_idx // batch_size + 1} - ICD: {icd_pca.shape}, Description: {desc_pca.shape}")

                # 保存到最终 HDF5 文件
                try:
                    # 保存 ICD 向量
                    dset_icd_final.resize(dset_icd_final.shape[0] + icd_pca.shape[0], axis=0)
                    dset_icd_final[-icd_pca.shape[0]:] = icd_pca

                    # 保存疾病描述向量
                    dset_desc_final.resize(dset_desc_final.shape[0] + desc_pca.shape[0], axis=0)
                    dset_desc_final[-desc_pca.shape[0]:] = desc_pca

                    print(f"Batch {start_idx // batch_size + 1} 已保存到最终文件")
                except Exception as e:
                    print(f"保存Batch {start_idx // batch_size + 1}到最终文件时出错: {e}")

                # **释放内存**
                del batch_icd, batch_desc, batch_icd_flat, batch_desc_flat, icd_pca, desc_pca
                gc.collect()
                torch.cuda.empty_cache()

        print("-----------------所有数据已成功应用 PCA 并保存到最终 HDF5 文件中-----------------")

        # 删除临时文件
        os.remove(temp_icd_file_path)
        os.remove(temp_desc_file_path)
        print("临时文件已删除")

        # 创建并保存 CustomDataset
        print("-----------------开始创建并保存 CustomDataset-----------------")
        # 数据预处理：将 mask_system 从字符串转换为列表
        parsed_mask_system_data = []
        for idx, codes_str in enumerate(mask_system):
            if isinstance(codes_str, str):
                # 去掉方括号
                codes_str = codes_str.strip("[]")
                # 按逗号分割
                codes = codes_str.split(',')
                # 去除多余的引号和空格
                codes_list = [code.strip().strip("'").strip('"') for code in codes]
            else:
                codes_list = codes_str
            parsed_mask_system_data.append(codes_list)

        # 创建 BatchedDataset 对象
        # Check if we should save batches
        if len([f for f in os.listdir(args.npz_data_dir) if f.endswith('.npz')]) < int(5000 / args.batch_size):
            # If there are fewer than 133 files, call save_batches_to_npz
            print("Less than 133 .npz files found, saving new batches...")
            save_batches_to_npz(df_all['Participant ID'].values,
                                Phenotypes.values,
                                args.icd_file_path,
                                args.desc_file_path,
                                Sex.values,
                                pd.Series(parsed_mask_system_data),
                                Survival_Time.values,
                                labels=y.values,
                                batch_size=args.batch_size,
                                npz_data_dir=args.npz_data_dir)

        batch_size = 10000
        hdf5_file_path = 'batches_data.h5'

        dataset = HDF5Dataset(hdf5_file_path, batch_size=batch_size)
        # 保存 dataset
        # torch.save(train_dataset, dataset_file)
        print("数据集已保存到", args.dataset_file)

    # 将数据划分为训练集、验证集和测试集
    print("-----------------划分数据集-----------------")
    train_loader, val_loader, test_loader, all_loader = get_data_loaders(args.npz_data_dir, batch_size=1)

    # 创建数据加载器
    print("-----------------创建数据加载器-----------------")

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AgePredictionModel().to(device)

    # 定义损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    initial_factor = args.initial_factor
    final_factor = args.final_factor
    num_epochs = args.num_epochs

    # 训练循环
    print('-----------------Training-----------------')
    for epoch in tqdm(range(num_epochs), desc='Training'):
        model.train()
        start_time = time.time()
        system_loss_total = 0.0
        batch_1 = 0
        for batch in train_loader:
            batch_Participant_ID_data, batch_features, batch_icd_vecs, batch_disease_vecs, batch_sex_data, batch_system_mask, batch_Survival_Time, batch_labels = batch

            # 强制转换为 float32 并移动到设备
            batch_features = batch_features.to(dtype=torch.float32, device=device)
            batch_icd_vecs = batch_icd_vecs.to(dtype=torch.float32, device=device)
            batch_disease_vecs = batch_disease_vecs.to(dtype=torch.float32, device=device)
            batch_sex_data = batch_sex_data.to(dtype=torch.float32, device=device)
            batch_system_mask = batch_system_mask.to(dtype=torch.float32, device=device)
            optimizer.zero_grad()

            # 前向传播
            final_outputs = model(batch_icd_vecs, batch_disease_vecs, batch_features, batch_sex_data, batch_system_mask)
            # print('finished forward propagate')
            batch_labels = batch_labels.to(dtype=torch.float32, device=device).squeeze(0).squeeze(0).unsqueeze(1)
            batch_system_mask = batch_system_mask.squeeze(0)
            # 计算损失
            penalty_factor = get_penalty_factor(epoch, num_epochs, initial_factor, final_factor)
            loss = combined_loss(final_outputs, batch_labels, batch_system_mask, penalty_factor, margin=1.0, alpha=0.5)

            # print('currently batch:',batch_1)
            # print('batch time:',time.time() - start_time)
            batch_1 += 1
            # 反向传播和优化
            # print('loss:',loss)
            loss.backward()
            # Check for exploding gradients
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    if grad_norm > 1e6:  # Threshold for detecting exploding gradients
                        print(f"Exploding gradients detected in {param}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            system_loss_total += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = system_loss_total / len(batch_labels)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.8f}, Time: {epoch_time:.2f} seconds")

        # 调度器步进
        scheduler.step(avg_loss)

        # 在验证集上评估模型
        metrics = evaluate(model, val_loader, device)
        # print(f"Validation Metrics for Epoch {epoch + 1}:")
        # for system, metric_vals in metrics.items():
            # print(f"  {system.capitalize()} - RMSE: {metric_vals['RMSE']:.8f}, MAE: {metric_vals['MAE']:.8f}, R2: {metric_vals['R2']:.8f}")

    # 保存训练好的模型
    torch.save(model.state_dict(), args.model_save_path)
    print(f"模型已保存到 {args.model_save_path}")

    # 测试过程
    print(f"-----------------Test Metrics:-----------------")
    formatted_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, f'test_data_{formatted_date}.csv')
    # metrics = evaluate(model, test_loader, device, output_csv=output_csv, is_test=True)
    metrics = evaluate(model, test_loader, device, output_csv=output_csv, is_test=True)


    # for system, metric_vals in metrics.items():
        # print(f"  {system.capitalize()} - RMSE: {metric_vals['RMSE']:.8f}, MAE: {metric_vals['MAE']:.8f}, R2: {metric_vals['R2']:.8f}")

    print(f"-----------------All data Test Metrics:-----------------")
    # 使用全部的数据进行验证
    all_data_output_csv = os.path.join(args.output_dir, f'All_data_{formatted_date}.csv')
    metrics_all = evaluate(model, all_loader, device, output_csv=all_data_output_csv, is_test=True)


    # for system, metric_vals in metrics_all.items():
        # print(f"  {system.capitalize()} - RMSE: {metric_vals['RMSE']:.8f}, MAE: {metric_vals['MAE']:.8f}, R2: {metric_vals['R2']:.8f}")

if __name__ == '__main__':
    main()
