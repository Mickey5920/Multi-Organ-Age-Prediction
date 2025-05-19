# Dataloader.py

import torch
import  random
mask = [0] * 7
from torch.utils.data import Dataset,IterableDataset
import h5py
import pandas as pd
import ast
import open_clip
from sklearn.decomposition import IncrementalPCA
import numpy as np
import time
import threading
import queue
import os
from tqdm import tqdm

chronic_diseases = {
    'cardiovascular': [
        #Ischemic heart disease
        "I20", "I20.0", "I20.1", "I20.8", "I20.9",
        "I21", "I21.0", "I21.1", "I21.2", "I21.3", "I21.4", "I21.9",
        "I22", "I22.0", "I22.1", "I22.8", "I22.9",
        "I23", "I23.0", "I23.1", "I23.2", "I23.3", "I23.4", "I23.5", "I23.6", "I23.8",
        "I24", "I24.0", "I24.1", "I24.8", "I24.9",
        "I25", "I25.0", "I25.1", "I25.2", "I25.3", "I25.4", "I25.5", "I25.6", "I25.8", "I25.9",
        #Hypertensive diseases
        "I10",
        "I11", "I11.0", "I11.9",
        "I12", "I12.0", "I12.9",
        "I13", "I13.0", "I13.1", "I13.2", "I13.9",
        "I50", "I50.0", "I50.1", "I50.9",
        #Cancer
        # "C37", "C38", "C39"

    ],
    'pulmonary': [
        #COPD
        "J43", "J43.0", "J43.1", "J43.2", "J43.8", "J43.9", "J44", "J44.0", "J44.1", "J44.8", "J44.9",
        #Cancer
        # "C30", "C31", "C32", "C33", "C34"
    ],
    'musculoskeletal': [
        #Osteoarthritis
        "M15", "M15.0", "M15.00", "M15.1", "M15.2", "M15.3", "M15.4", "M15.8", "M15.9", "M15.99",
        "M16", "M16.0", "M16.1", "M16.2", "M16.3", "M16.4", "M16.5", "M16.6", "M16.7", "M16.9",
        "M17", "M17.0", "M17.1", "M17.2", "M17.3", "M17.4", "M17.5", "M17.9",
        "M18", "M18.0", "M18.1", "M18.2", "M18.3", "M18.4", "M18.5", "M18.9",
        "M19", "M19.0", "M19.00", "M19.01", "M19.02", "M19.03", "M19.04", "M19.05", "M19.06", "M19.07", "M19.08", "M19.09",
        "M19.1", "M19.10", "M19.11", "M19.12", "M19.13", "M19.14", "M19.15", "M19.16", "M19.17", "M19.18", "M19.19",
        "M19.2", "M19.20", "M19.21", "M19.22", "M19.23", "M19.24", "M19.25", "M19.26", "M19.27", "M19.28", "M19.29",
        "M19.8", "M19.80", "M19.81", "M19.82", "M19.83", "M19.84", "M19.85", "M19.86", "M19.87", "M19.88", "M19.89",
        "M19.9", "M19.90", "M19.91", "M19.92", "M19.93", "M19.94", "M19.95", "M19.96", "M19.97", "M19.98", "M19.99"
        #Osteoporosis
        "M80.00", "M80.01", "M80.02", "M80.03", "M80.04", "M80.05", "M80.06", "M80.07", "M80.08", "M80.09",
        "M80.9", "M80.90", "M80.91", "M80.92", "M80.93", "M80.94", "M80.95", "M80.96", "M80.97", "M80.98", "M80.99",
        #Cancer
        # "C40", "C41", "C49"

    ],
    'immune': [
        "D80", "D81", "D82", "D83", "D84", "D85", "D86", "D89", "M05", "M06", "M08", "M30", "M31", "M32", "M33", "M35", "M36", "L93.0", "L93.1", "L93.2",
       # Cancer
       #  "C81", "C82", "C83", "C84", "C85", "C86", "C88", "C90"
    ],
    'renal': [
        #CKD
        "N18", "N18.0", "N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.8", "N18.9",
        #Cancer
        # "C64", "C65", "C66", "C67", "C68"
    ],
    'hepatic': [
        #Cirrhosis
        "K70.2", "K70.3", "K70.4", "K72", "K72.1", "K72.9", "K74", "K74.0", "K74.1", "K74.2", "K74.3", "K74.4",
        "K74.5", "K74.6",
        #Cancer
        # "C22", "C22.0", "C22.1"
    ],
    'metabolic': [
        # Diabetes
        "E11", "E11.0", "E11.1", "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7", "E11.8", "E11.9",
        #Cancer
        # "C43", "C43.0", "C43.1", "C43.2", "C43.3", "C43.4", "C43.5", "C43.6", "C43.7", "C43.8", "C43.9", "C69", "C70", "C71", "C72", "C73", "C74", "C75"
    ]
}

import numpy as np
from sklearn.decomposition import IncrementalPCA
import h5py


def fit_pca_models(file_path_icd, file_path_desc, file_path_drug, target_dim=514, batch_size=1000):
    """
    - file_path_icd: str，
    - file_path_desc: str，
    - file_path_drug: str，
    - target_dim: int，
    - batch_size: int，

    - pca_icd: sklearn.decomposition.IncrementalPCA
    - pca_desc: sklearn.decomposition.IncrementalPCA
    - pca_drug: sklearn.decomposition.IncrementalPCA
    """

    pca_icd = IncrementalPCA(n_components=target_dim, batch_size=batch_size)
    pca_desc = IncrementalPCA(n_components=target_dim, batch_size=batch_size)
    pca_drug = IncrementalPCA(n_components=target_dim, batch_size=batch_size)


    with h5py.File(file_path_icd, 'r') as f_icd:
        data_icd = f_icd['data']
        total_samples_icd = data_icd.shape[0]

        for start_idx in range(0, total_samples_icd, batch_size):
            end_idx = min(start_idx + batch_size, total_samples_icd)
            batch_icd = data_icd[start_idx:end_idx, :, :]  # (batch_size, 15, feature_dim + 2)
            batch_icd_flat = batch_icd.reshape(-1, batch_icd.shape[2])  # (batch_size*15, feature_dim + 2)


            valid_rows = ~np.isnan(batch_icd_flat).any(axis=1)
            cleaned_batch = batch_icd_flat[valid_rows]

            if cleaned_batch.shape[0] > 0:
                pca_icd.partial_fit(cleaned_batch)


    with h5py.File(file_path_desc, 'r') as f_desc:
        data_desc = f_desc['data']
        total_samples_desc = data_desc.shape[0]

        for start_idx in range(0, total_samples_desc, batch_size):
            end_idx = min(start_idx + batch_size, total_samples_desc)
            batch_desc = data_desc[start_idx:end_idx, :, :]
            batch_desc_flat = batch_desc.reshape(-1, batch_desc.shape[2])


            valid_rows = ~np.isnan(batch_desc_flat).any(axis=1)
            cleaned_batch = batch_desc_flat[valid_rows]

            if cleaned_batch.shape[0] > 0:
                pca_desc.partial_fit(cleaned_batch)


    with h5py.File(file_path_drug, 'r') as f_drug:
        data_drug = f_drug['data']
        total_samples = data_drug.shape[0]

        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch = data_drug[start_idx:end_idx, :, :]
            batch_flat = batch.reshape(-1, batch.shape[2])


            valid_rows = ~np.isnan(batch_flat).any(axis=1)
            cleaned_batch = batch_flat[valid_rows]

            if cleaned_batch.shape[0] > 0:
                pca_drug.partial_fit(cleaned_batch)
    return pca_icd, pca_desc, pca_drug


def process_batch(batch_icd_codes, batch_disease_descs, batch_icd_times, batch_drug_names, batch_drug_times,
                  max_len=15):
    batch_size = len(batch_icd_codes)
    print(f"Processing batch_size: {batch_size}")

    icd_codes_list = []
    disease_descs_list = []
    icd_time_list = []
    drug_name_list = []
    drug_time_list = []

    for icd_codes, disease_descs, icd_time_row, drug_name, drug_time_row in tqdm(zip(
            batch_icd_codes, batch_disease_descs, batch_icd_times, batch_drug_names, batch_drug_times
    ), total=batch_size):

        icd_codes = str(icd_codes) if not pd.isna(icd_codes) else ''
        disease_descs = str(disease_descs) if not pd.isna(disease_descs) else ''
        icd_codes_split = icd_codes.strip('[]').replace("'", "").split(', ')
        disease_descs_split = disease_descs.strip('[]').replace("'", "").split(', ')


        icd_time_row = np.nan_to_num(icd_time_row, nan=0.0)
        icd_time_row = icd_time_row[:max_len]
        if len(icd_time_row) < max_len:
            icd_time_row = np.pad(icd_time_row, (0, max_len - len(icd_time_row)), 'constant', constant_values=0.0)


        icd_codes_split = icd_codes_split[:max_len] + [''] * (max_len - len(icd_codes_split))
        disease_descs_split = disease_descs_split[:max_len] + [''] * (max_len - len(disease_descs_split))


        drug_name = str(drug_name) if not pd.isna(drug_name) else "['']"
        drug_name_split = drug_name.strip('[]').replace("'", "").split(', ')
        # drug_name_split = drug_name.strip('[]').replace("'", "").split(', ')
        drug_name_split = drug_name_split[:max_len] + [''] * (max_len - len(drug_name_split))


        if pd.isna(drug_time_row) or drug_time_row == '[]':
            drug_time_parsed = []
        else:
            drug_time_parsed = ast.literal_eval(str(drug_time_row))
        drug_time_arr = np.array([float(t) for t in drug_time_parsed], dtype=np.float32)
        drug_time_arr = drug_time_arr[:max_len]  # 截断
        if len(drug_time_arr) < max_len:
            drug_time_arr = np.pad(drug_time_arr, (0, max_len - len(drug_time_arr)), 'constant', constant_values=0.0)


        drug_name_list.append(drug_name_split)
        drug_time_list.append(drug_time_arr)


        icd_codes_list.append(icd_codes_split)
        disease_descs_list.append(disease_descs_split)
        icd_time_list.append(icd_time_row)

    all_icd_texts = [code if code != '' else ' ' for sample in icd_codes_list for code in sample]
    all_desc_texts = [desc if desc != '' else ' ' for sample in disease_descs_list for desc in sample]
    all_drug_texts = [drug if drug != '' else ' ' for sample in drug_name_list for drug in sample]

    all_icd_features = encode_texts(all_icd_texts)
    all_desc_features = encode_texts(all_desc_texts)
    all_drug_features = encode_texts(all_drug_texts)

    all_icd_features_np = np.nan_to_num(all_icd_features.numpy(), nan=0.0)
    all_desc_features_np = np.nan_to_num(all_desc_features.numpy(), nan=0.0)
    all_drug_features_np = np.nan_to_num(all_drug_features.numpy(), nan=0.0)

    batch_icd_times_processed = np.array(icd_time_list)  # (batch_size, max_len)
    batch_drug_times_processed = np.array(drug_time_list)  # (batch_size, max_len)

    drug_time_tensors = np.hstack([
        np.ones((batch_size * max_len, 1)),
        batch_drug_times_processed.reshape(-1, 1)
    ])
    drug_vectors = np.hstack([drug_time_tensors, all_drug_features_np])

    drug_features_list = drug_vectors.reshape(batch_size, max_len, -1)

    icd_time_tensors = np.hstack([np.zeros((batch_size * max_len, 1)), batch_icd_times_processed.reshape(-1, 1)])
    desc_time_tensors = np.hstack([np.ones((batch_size * max_len, 1)), batch_icd_times_processed.reshape(-1, 1)])
    icd_vectors = np.hstack([icd_time_tensors, all_icd_features_np])
    desc_vectors = np.hstack([desc_time_tensors, all_desc_features_np])
    icd_features_list = icd_vectors.reshape(batch_size, max_len, -1)
    desc_features_list = desc_vectors.reshape(batch_size, max_len, -1)

    assert drug_features_list.shape == icd_features_list.shape, \
        f"Shape mismatch: drug {drug_features_list.shape} vs icd {icd_features_list.shape}"

    return icd_features_list, desc_features_list, drug_features_list

def encode_texts(texts, batch_size=128):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    model_clip = model_clip.to(device)
    model_clip.eval()

    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    all_features = []
    num_texts = len(texts)
    num_batches = (num_texts + batch_size - 1) // batch_size
    for i in range(num_batches):
        batch_texts = texts[i * batch_size: (i + 1) * batch_size]
        tokens = tokenizer(batch_texts)
        tokens = tokens.to(device)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                text_features = model_clip.encode_text(tokens)
        text_features = torch.nan_to_num(text_features, nan=0.0)
        # Move features to CPU and append
        all_features.append(text_features.cpu())

        del tokens, text_features
        torch.cuda.empty_cache()
    all_features = torch.cat(all_features, dim=0)

    del model_clip, tokenizer
    return all_features

def get_system_mask_code(icd_codes):

    mask = [0] * 8
    found_system = False

    systems_order = [
        ('cardiovascular', 0),
        ('pulmonary', 1),
        ('musculoskeletal', 2),
        ('immune', 3),
        ('renal', 4),
        ('hepatic', 5),
        ('metabolic', 6)
    ]

    for code in icd_codes:
        if isinstance(code, list):
            if len(code) == 0 or (len(code) == 1 and str(code[0]).strip() == ''):
                found_system = True
                continue
            current_code = str(code[0]).strip()
        else:
            current_code = str(code).strip()
            if current_code == '':
                found_system = True
                continue

        code_matched = False
        for system_name, mask_idx in systems_order:
            for prefix in chronic_diseases[system_name]:
                if current_code.startswith(prefix):
                    mask[mask_idx] = 1
                    found_system = True
                    code_matched = True
                    break
            if code_matched:
                break

    if not found_system:
        mask[7] = 2

    return torch.tensor(mask, dtype=torch.float32)


class BioAgeDataset(Dataset):
    def __init__(self, data_dir):
        # List all the .npz files in the data directory
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.data_dir = data_dir

    def __len__(self):
        # Number of .npz files
        return len(self.data_files)

    def __getitem__(self, idx):
        # Load a specific .npz file
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        data = np.load(file_path)

        # Extract data from the npz file
        batch_Participant_ID = data['Participant_ID']
        batch_features = data['features']
        batch_icd_vec = data['icd_vec']
        batch_disease_vec = data['disease_vec']
        batch_drug_vec = data['drug_vec']
        batch_sex = data['sex_data']
        batch_system_mask = data['system_mask']
        batch_occurrence_Next_Time = data['occurrence_Next_Time']
        batch_Survival_Time = data['Survival_Time']
        batch_labels = data['label']

        # Convert data to torch tensors
        batch_Participant_ID = batch_Participant_ID.tolist()  # Convert to a list of strings
        batch_features = torch.tensor(batch_features, dtype=torch.float32)
        batch_icd_vec = torch.tensor(batch_icd_vec, dtype=torch.float32)
        batch_disease_vec = torch.tensor(batch_disease_vec, dtype=torch.float32)
        batch_drug_vec = torch.tensor(batch_drug_vec, dtype=torch.float32)
        batch_sex = torch.tensor(batch_sex, dtype=torch.float32)
        batch_system_mask = torch.tensor(batch_system_mask, dtype=torch.float32)
        batch_occurrence_Next_Time = torch.tensor(batch_occurrence_Next_Time, dtype=torch.float32)
        batch_Survival_Time = torch.tensor(batch_Survival_Time, dtype=torch.float32)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float32)

        return (batch_Participant_ID, batch_features, batch_icd_vec, batch_disease_vec,batch_drug_vec,
                batch_sex, batch_system_mask, batch_occurrence_Next_Time,batch_Survival_Time, batch_labels)
