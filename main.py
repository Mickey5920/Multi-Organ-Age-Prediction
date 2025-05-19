# main.py
from datetime import datetime
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
from train import combined_loss, evaluate, get_penalty_factor
from model import AgePredictionModel
from Dataloader import encode_texts, process_batch, fit_pca_models,get_system_mask_code,BioAgeDataset
import numpy as np
import h5py
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Age Prediction Model Training')
    parser.add_argument('--All_use_data',type=str,default='./All_use_data/All_use_data-0412.csv',help='All Use Data Path')
    parser.add_argument('--learnable_weight',type=str, default='Yes',help='learnable_weight:Yes,No')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for data files')
    parser.add_argument('--Loss',type=str,default='All',help='Loss Function: Ranking、All、MESLoss')
    parser.add_argument('--alpha',type=float,default=2,help='alpha')
    parser.add_argument('--model',type=str,default='All',help='All,Drug,Disease,ICD,Phenotypes')
    parser.add_argument('--icd_file', type=str, default='./ICD_code_descriptions/X_icd_codes_vec_Instance_0_mask_system_Phenotypes_time_No_Data_Augmentation_0412.h5', help='Path to ICD vectors HDF5 file')
    parser.add_argument('--desc_file', type=str, default='./ICD_code_descriptions/X_disease_desc_Instance_0_mask_system_Phenotypes_time_No_Data_Augmentation_0412.h5', help='Path to disease descriptions HDF5 file')
    parser.add_argument('--drug_file', type=str,default='./ICD_code_descriptions/Drug_data_No_Data_Augmentation_0411.h5',help='Path to disease descriptions HDF5 file')
    parser.add_argument('--dataset_file', type=str, default='./data/custom_dataset_mask_system_Phenotypes_time_all_0411.pt', help='Path to save/load the dataset')
    parser.add_argument('--batch_size', type=int, default=20000, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--initial_factor', type=float, default=5.0, help='Initial penalty factor')
    parser.add_argument('--final_factor', type=float, default=2.0, help='Final penalty factor')
    parser.add_argument('--npz_data_dir',type=str,default='./npz_data_No_Data_Augmentation-0412',help='npz data storage')
    parser.add_argument('--model_save_path', type=str, default='./checkpoint/model_checkpoint_Phenotypes_time_npz_All_0412.pt', help='Path to save the trained model')
    parser.add_argument('--output_dir', type=str, default='./test', help='Directory to save test outputs')
    # Add more arguments as needed
    return parser.parse_args()

def convert_string_list_to_array(data):
    parsed_data = []
    for item in data:
        try:
            parsed_item = ast.literal_eval(item)
            if isinstance(parsed_item, list):
                parsed_data.append(parsed_item)
            else:
                parsed_data.append([0.0] * 7)
        except:
            parsed_data.append([0.0] * 7)
    return np.array(parsed_data, dtype=np.float32)
def save_batches_to_npz(Participant_ID, features, icd_file_paths, desc_file_paths,durg_file_paths, sex, mask_system_data, occurrence_Next_Time,Survival_Time,labels, batch_size=20000,npz_data_dir='data_npz'):
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
        batch_occurrence_Next_Time = occurrence_Next_Time[start_idx:end_idx]
        # Check if the batch is empty (should not be, but just in case)
        if len(batch_Participant_ID) == 0:
            print(f"Batch {batch_num} is empty, skipping.")
            continue

        # Replace '-' with '.' in Participant_ID and keep it as a string
        batch_Participant_ID = np.array([str(id).replace('-', '.') for id in batch_Participant_ID], dtype='S10')

        # Ensure sex is of float32 type
        batch_sex = np.array(batch_sex, dtype=np.float32)


        try:

            processed_masks = []
            for codes in batch_mask_system_data:
                processed_mask = get_system_mask_code(codes)

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
            batch_occurrence_Next_Time = convert_string_list_to_array(batch_occurrence_Next_Time)
            batch_labels = np.array(batch_labels, dtype=np.float32)
        except Exception as e:
            print(f"Error converting features or labels to float32 in batch {batch_num}: {e}")
            continue

        # Read ICD and disease data from files
        try:
            with h5py.File(durg_file_paths, 'r') as icd_file:
                if 'data' not in icd_file:
                    raise KeyError(f"'data' key not found in ICD file {durg_file_paths}")
                drug_data = np.array(icd_file['data'][start_idx:end_idx], dtype=np.float32)
                batch_drug_vec = drug_data
        except Exception as e:
            print(f"Error reading ICD file in batch {batch_num}: {e}")
            continue
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
        if not (len(batch_features) == len(batch_Survival_Time) == len(batch_icd_vec) == len(batch_disease_vec) ==len(batch_drug_vec)== len(batch_sex) == len(batch_mask_system_data) == len(batch_labels) == batch_size_actual):
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
                     drug_vec=batch_drug_vec,
                     sex_data=batch_sex,
                     occurrence_Next_Time=batch_occurrence_Next_Time,
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
    df_all = pd.read_csv(args.All_use_data)
    df_all = df_all[df_all['Participant ID'].astype(str).str.endswith('-1')]
    sample_length = len(df_all)
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
        "Age of Instance 0",
        "Ischemic_Heart_Disease", "Hypertension", "Stroke", "COPD", "CKD",
        "Diabetes", "Cirrhosis", "Osteoarthritis", "Osteoporosis", "Cancer"
    ]].copy()

    X_icd_codes = df_all['ICD'].copy()
    ICD_Time = df_all.iloc[:, 177:-1]
    Survival_Time = df_all.iloc[:, 121:177]

    X_disease_desc = df_all['Disease_Description_Feature_x'].copy()


    y = df_all['Actual_Age'].copy()
    mask_system = df_all['ICD'].copy()

    Phenotypes = X_features.iloc[:, 2:80]
    Participant_IDs = df_all['Participant ID'].copy()

    Sex = X_features['Sex'].map({'Female': 0, 'Male': 1})
    Drug_Names = df_all['New_Drugs_Name']
    Drug_Times = df_all['New_time_residual']
    occurrence_Next_Time = df_all['occurrence_Next_Time']

    with torch.no_grad():
        sample_text = ["sample"]
        sample_features = encode_texts(sample_text)
        embedding_dim = sample_features.shape[-1]
        print(f": {embedding_dim}")


    temp_icd_file_path = './ICD_code_descriptions/temp_X_icd_codes_0226.h5'
    temp_desc_file_path = './ICD_code_descriptions/temp_X_disease_desc_0226.h5'
    temp_drug_file_path = './ICD_code_descriptions/temp_X_drug_desc_0226.h5'

    if os.path.exists(args.icd_file) and os.path.exists(args.desc_file) and os.path.exists(args.drug_file):


        # 加载或创建 CustomDataset
        if os.path.exists(args.dataset_file):

            dataset = torch.load(args.dataset_file)
        else:


            parsed_mask_system_data = []
            for idx, codes_str in enumerate(mask_system):
                if isinstance(codes_str, str):

                    codes_str = codes_str.strip("[]")

                    codes = codes_str.split(',')

                    codes_list = [code.strip().strip("'").strip('"') for code in codes]
                else:
                    codes_list = codes_str
                parsed_mask_system_data.append(codes_list)

            # Check if we should save batches
            if len([f for f in os.listdir(args.npz_data_dir) if f.endswith('.npz')]) < int(sample_length / args.batch_size):
                # If there are fewer than 133 files, call save_batches_to_npz
                print("Less than 133 .npz files found, saving new batches...")
                save_batches_to_npz(Participant_IDs.values,
                                    Phenotypes.values,
                                    args.icd_file,
                                    args.desc_file,
                                    args.desc_file,
                                    Sex.values,
                                    pd.Series(parsed_mask_system_data),
                                    occurrence_Next_Time,
                                    Survival_Time.values,
                                    labels=y.values,
                                    batch_size=args.batch_size,
                                    npz_data_dir=args.npz_data_dir)
            else:
                # Otherwise, skip saving the batches
                print(f"Already {len(os.listdir(args.npz_data_dir))} .npz files found, skipping batch saving.")

            print("dataset", args.dataset_file)
    else:
        with h5py.File(temp_icd_file_path, 'w') as f_icd_temp, h5py.File(temp_desc_file_path, 'w') as f_desc_temp, h5py.File(temp_drug_file_path, 'w') as f_drug_temp:

            f_icd_temp.create_dataset('data', shape=(0, 15, embedding_dim + 2),
                                      maxshape=(None, 15, embedding_dim + 2),
                                      dtype='float32', chunks=True)
            f_desc_temp.create_dataset('data', shape=(0, 15, embedding_dim + 2),
                                       maxshape=(None, 15, embedding_dim + 2),
                                       dtype='float32', chunks=True)
            f_drug_temp.create_dataset('data', shape=(0, 15, embedding_dim + 2),
                                       maxshape=(None, 15, embedding_dim + 2),
                                       dtype='float32', chunks=True)
            print(f": (0, 15, {embedding_dim + 2})")


        batch_size = 100000

        total_samples = len(X_icd_codes)
        num_batches = (total_samples + batch_size - 1) // batch_size


        with h5py.File(temp_icd_file_path, 'a') as f_icd_temp, h5py.File(temp_desc_file_path, 'a') as f_desc_temp,h5py.File(temp_drug_file_path, 'a') as f_drug_temp:
            dset_icd_temp = f_icd_temp['data']
            dset_desc_temp = f_desc_temp['data']
            dset_drug_temp = f_drug_temp['data']
            for batch_num in tqdm(range(num_batches), desc='encoding batches'):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, total_samples)
                batch_icd_codes = X_icd_codes.iloc[start_idx:end_idx].tolist()
                batch_disease_desc = X_disease_desc.iloc[start_idx:end_idx].tolist()
                batch_icd_time = ICD_Time.iloc[start_idx:end_idx].values

                batch_drug_name = Drug_Names.iloc[start_idx:end_idx].tolist()
                batch_drug_time = Drug_Times.iloc[start_idx:end_idx].values


                icd_features, desc_features,drug_features = process_batch(batch_icd_codes, batch_disease_desc, batch_icd_time,batch_drug_name,batch_drug_time,
                                                            max_len=15)

                print(f"Batch {batch_num + 1}/{num_batches} - "
                      f"ICD: {icd_features.shape}, "
                      f"Desc: {desc_features.shape}, "
                      f"Drug: {drug_features.shape}")
                try:

                    new_size = dset_icd_temp.shape[0] + icd_features.shape[0]


                    dset_icd_temp.resize(new_size, axis=0)
                    dset_desc_temp.resize(new_size, axis=0)
                    dset_drug_temp.resize(new_size, axis=0)


                    dset_icd_temp[-icd_features.shape[0]:] = icd_features.astype('float32')
                    dset_desc_temp[-desc_features.shape[0]:] = desc_features.astype('float32')
                    dset_drug_temp[-drug_features.shape[0]:] = drug_features.astype('float32')

                    print(f"Batch {batch_num + 1} ")
                except Exception as e:
                    print(f" {batch_num + 1} : {e}")
                    raise


                del icd_features, desc_features, drug_features
                gc.collect()
                torch.cuda.empty_cache()

        pca_icd, pca_desc,pca_drug = fit_pca_models(temp_icd_file_path, temp_desc_file_path,temp_drug_file_path, target_dim=514, batch_size=1000)

        with h5py.File(args.icd_file, 'w') as f_icd, h5py.File(args.desc_file, 'w') as f_desc, h5py.File(args.drug_file, 'w') as f_drug:

            f_icd.create_dataset('data', shape=(0, 15, 514), maxshape=(None, 15, 514),
                                 dtype='float32', chunks=True)
            f_desc.create_dataset('data', shape=(0, 15, 514), maxshape=(None, 15, 514),
                                  dtype='float32', chunks=True)
            f_drug.create_dataset('data', shape=(0, 15, 514), maxshape=(None, 15, 514),
                                  dtype='float32', chunks=True)


        with h5py.File(temp_icd_file_path, 'r') as f_icd_temp, \
                h5py.File(temp_desc_file_path, 'r') as f_desc_temp, \
                h5py.File(temp_drug_file_path, 'r') as f_drug_temp, \
                h5py.File(args.icd_file, 'a') as f_icd_final, \
                h5py.File(args.desc_file, 'a') as f_desc_final, \
                h5py.File(args.drug_file, 'a') as f_drug_final:

            data_icd_temp = f_icd_temp['data']
            data_desc_temp = f_desc_temp['data']
            data_drug_temp = f_drug_temp['data']

            dset_icd_final = f_icd_final['data']
            dset_desc_final = f_desc_final['data']
            dset_drug_final = f_drug_final['data']

            for start_idx in tqdm(range(0, data_icd_temp.shape[0], batch_size), desc="Applying PCA and Saving"):
                end_idx = min(start_idx + batch_size, data_icd_temp.shape[0])
                batch_icd = data_icd_temp[start_idx:end_idx, :, :]  # (batch_size, 15, feature_dim + 2)
                batch_desc = data_desc_temp[start_idx:end_idx, :, :]  # (batch_size, 15, feature_dim + 2)
                batch_drug = data_drug_temp[start_idx:end_idx, :, :]

                # Reshape 并处理 NaN
                batch_icd_flat = batch_icd.reshape(-1, batch_icd.shape[2])
                batch_desc_flat = batch_desc.reshape(-1, batch_desc.shape[2])
                batch_drug_flat = batch_drug.reshape(-1, batch_drug.shape[2])

                # 将 NaN 替换为 0
                batch_icd_flat = np.nan_to_num(batch_icd_flat, nan=0.0)
                batch_desc_flat = np.nan_to_num(batch_desc_flat, nan=0.0)
                batch_drug_flat = np.nan_to_num(batch_drug_flat, nan=0.0)

                # 应用 PCA
                icd_pca = pca_icd.transform(batch_icd_flat)
                desc_pca = pca_desc.transform(batch_desc_flat)
                drug_pca = pca_drug.transform(batch_drug_flat)


                icd_pca = icd_pca.reshape(-1, 15, 514)
                desc_pca = desc_pca.reshape(-1, 15, 514)
                drug_pca = drug_pca.reshape(-1, 15, 514)

                print(
                    f"Batch {start_idx // batch_size + 1} - ICD: {icd_pca.shape}, Description: {desc_pca.shape}, Drug: {drug_pca.shape}")


                try:

                    for data, dset in zip([icd_pca, desc_pca, drug_pca],
                                          [dset_icd_final, dset_desc_final, dset_drug_final]):
                        dset.resize(dset.shape[0] + data.shape[0], axis=0)
                        dset[-data.shape[0]:] = data

                    print(f"Batch {start_idx // batch_size + 1} ")
                except Exception as e:
                    print(f" {start_idx // batch_size + 1} : {e}")


                del batch_icd, batch_desc, batch_drug, batch_icd_flat, batch_desc_flat, batch_drug_flat, icd_pca, desc_pca, drug_pca
                gc.collect()
                torch.cuda.empty_cache()
        os.remove(temp_icd_file_path)
        os.remove(temp_desc_file_path)

        parsed_mask_system_data = []
        for idx, codes_str in enumerate(mask_system):
            if isinstance(codes_str, str):

                codes_str = codes_str.strip("[]")

                codes = codes_str.split(',')

                codes_list = [code.strip().strip("'").strip('"') for code in codes]
            else:
                codes_list = codes_str
            parsed_mask_system_data.append(codes_list)

        # Check if we should save batches
        if len([f for f in os.listdir(args.npz_data_dir) if f.endswith('.npz')]) < int(1325056 / args.batch_size):
            # If there are fewer than 133 files, call save_batches_to_npz
            print("Less than 133 .npz files found, saving new batches...")
            save_batches_to_npz(Participant_IDs.values,
                                Phenotypes.values,
                                args.icd_file,
                                args.desc_file,
                                args.desc_file,
                                Sex.values,
                                pd.Series(parsed_mask_system_data),
                                occurrence_Next_Time,
                                Survival_Time.values,
                                labels=y.values,
                                batch_size=args.batch_size,
                                npz_data_dir=args.npz_data_dir)

    train_loader, val_loader, test_loader, all_loader = get_data_loaders(args.npz_data_dir, batch_size=1)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AgePredictionModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    for epoch in tqdm(range(args.num_epochs), desc='Training'):
        model.train()
        start_time = time.time()
        system_loss_total = 0.0
        batch_1 = 0
        for batch in train_loader:
            batch_Participant_ID_data, batch_features, batch_icd_vecs, batch_disease_vecs,batch_drug_vecs, batch_sex_data, batch_system_mask, batch_occurrence_Next_Time,batch_Survival_Time, batch_labels = batch

            # 强制转换为 float32 并移动到设备
            batch_features = batch_features.to(dtype=torch.float32, device=device)
            batch_icd_vecs = batch_icd_vecs.to(dtype=torch.float32, device=device)
            batch_disease_vecs = batch_disease_vecs.to(dtype=torch.float32, device=device)
            batch_sex_data = batch_sex_data.to(dtype=torch.float32, device=device)
            batch_system_mask = batch_system_mask.to(dtype=torch.float32, device=device)
            batch_drug_vecs = batch_drug_vecs.to(dtype=torch.float32, device=device)
            occurrence_Next_Time = batch_occurrence_Next_Time.to(dtype=torch.float32, device=device)
            optimizer.zero_grad()

            # 前向传播
            final_outputs = model(batch_icd_vecs, batch_disease_vecs, batch_drug_vecs, batch_features, batch_sex_data, batch_system_mask,args)

            batch_labels = batch_labels.to(dtype=torch.float32, device=device).squeeze(0).squeeze(0).unsqueeze(1)
            batch_system_mask = batch_system_mask.squeeze(0)
            # 计算损失
            penalty_factor = get_penalty_factor(epoch, args.num_epochs, args.initial_factor, args.final_factor)
            loss = combined_loss(final_outputs, batch_labels, batch_system_mask, occurrence_Next_Time,penalty_factor,args.Loss, margin=2.0, alpha=args.alpha)
            batch_1 += 1

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
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {avg_loss:.8f}, Time: {epoch_time:.2f} seconds")
        scheduler.step(avg_loss)
        metrics = evaluate(model, val_loader, device,args)

    # 保存训练好的模型
    torch.save(model.state_dict(), args.model_save_path)
    print(f"{args.model_save_path}")

    # 测试过程
    print(f"-----------------Test Metrics:-----------------")
    formatted_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, f'test_data_{formatted_date}-Input Data_{args.model}-Loss_{args.Loss}-Learnable_{args.learnable_weight}-alpha_{args.alpha}.csv')
    metrics = evaluate(model, test_loader, device,args, output_csv=output_csv, is_test=True)
    print(f"-----------------All data Test Metrics:-----------------")
    all_data_output_csv = os.path.join(args.output_dir, f'All_data_{formatted_date}-Input Data_{args.model}-Loss_{args.Loss}-Learnable_{args.learnable_weight}-alpha_{args.alpha}.csv')
    metrics_all = evaluate(model, all_loader, device, args, output_csv=all_data_output_csv, is_test=True)
if __name__ == '__main__':
    main()
