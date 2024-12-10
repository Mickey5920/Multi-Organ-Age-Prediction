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
# 定义一个包含不同系统慢性病ICD代码的字典
chronic_diseases = {
    'cardiovascular': [ # 心血管系统相关疾病代码
        "I20", "I20.0", "I20.1", "I20.8", "I20.9", "I21", "I21.0", "I21.1", "I21.2", "I21.3", "I21.4", "I21.9",
        "I22", "I22.0", "I22.1", "I22.8", "I22.9", "I23", "I23.0", "I23.1", "I23.2", "I23.3", "I23.4", "I23.5",
        "I23.6", "I23.8", "I24", "I24.0", "I24.1", "I24.8", "I24.9", "I25", "I25.0", "I25.1", "I25.2", "I25.3",
        "I25.4", "I25.5", "I25.6", "I25.8", "I25.9", "I10", "I11", "I11.0", "I11.9", "I12", "I12.0", "I12.9",
        "I13", "I13.0", "I13.1", "I13.2", "I13.9", "I50", "I50.0", "I50.1", "I50.9", "I60", "I60.0", "I60.1",
        "I60.2", "I60.3", "I60.4", "I60.5", "I60.6", "I60.7", "I60.8", "I60.9", "I61", "I61.0", "I61.1", "I61.2",
        "I61.3", "I61.4", "I61.5", "I61.6", "I61.8", "I61.9", "I63", "I63.0", "I63.1", "I63.2", "I63.3", "I63.4",
        "I63.5", "I63.6", "I63.8", "I63.9", "I64"
    ],
    'pulmonary': [ # 肺系统相关疾病代码
        "J43", "J43.0", "J43.1", "J43.2", "J43.8", "J43.9", "J44", "J44.0", "J44.1", "J44.8", "J44.9"
    ],
    'musculoskeletal': [ # 肌肉骨骼系统相关疾病代码
        "M15", "M15.0", "M15.00", "M15.1", "M15.2", "M15.3", "M15.4", "M15.8", "M15.9", "M15.99", "M16", "M16.0",
        "M16.1", "M16.2", "M16.3", "M16.4", "M16.5", "M16.6", "M16.7", "M16.9", "M17", "M17.0", "M17.1", "M17.2",
        "M17.3", "M17.4", "M17.5", "M17.9", "M18", "M18.0", "M18.1", "M18.2", "M18.3", "M18.4", "M18.5", "M18.9",
        "M19", "M19.0", "M19.00", "M19.01", "M19.02", "M19.03", "M19.04", "M19.05", "M19.06", "M19.07", "M19.08",
        "M19.09", "M19.1", "M19.10", "M19.11", "M19.12", "M19.13", "M19.14", "M19.15", "M19.16", "M19.17", "M19.18",
        "M19.19", "M19.2", "M19.20", "M19.21", "M19.22", "M19.23", "M19.24", "M19.25", "M19.26", "M19.27", "M19.28",
        "M19.29", "M19.8", "M19.80", "M19.81", "M19.82", "M19.83", "M19.84", "M19.85", "M19.86", "M19.87", "M19.88",
        "M19.89", "M19.9", "M19.90", "M19.91", "M19.92", "M19.93", "M19.94", "M19.95", "M19.96", "M19.97", "M19.98",
        "M19.99", "M80.00", "M80.01", "M80.02", "M80.03", "M80.04", "M80.05", "M80.06", "M80.07", "M80.08", "M80.09",
        "M80.9", "M80.90", "M80.91", "M80.92", "M80.93", "M80.94", "M80.95", "M80.96", "M80.97", "M80.98", "M80.99"
    ],
    'immune': [ # 免疫系统相关疾病代码
        "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14",
        "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C22.0", "C22.1", "C23", "C24", "C25", "C26", "C30",
        "C31", "C32", "C33", "C34", "C37", "C38", "C39", "C40", "C41", "C43", "C43.0", "C43.1", "C43.2", "C43.3",
        "C43.4", "C43.5", "C43.6", "C43.7", "C43.8", "C43.9", "C44", "C45", "C46", "C47", "C48", "C49", "C50.0",
        "C50.1", "C50.2", "C50.3", "C50.4", "C50.5", "C50.6", "C50.8", "C50.9", "C51", "C52", "C53", "C54", "C55",
        "C56", "C57", "C58", "C64", "C65", "C66", "C67", "C68", "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76",
        "C77", "C78", "C79", "C80", "C81", "C82", "C83", "C84", "C85", "C86", "C88", "C90", "C91", "C92", "C93",
        "C94", "C95", "C96", "C97"
    ],
    'renal': [ # 肾系统相关疾病代码
        "N18", "N18.0", "N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.8", "N18.9"
    ],
    'hepatic': [ # 肝脏系统相关疾病代码
        "K70.2", "K70.3", "K70.4", "K72", "K72.1", "K72.9", "K74", "K74.0", "K74.1", "K74.2", "K74.3", "K74.4",
        "K74.5", "K74.6"
    ],
    'metabolic': [ # 代谢系统相关疾病代码
        "E11", "E11.0", "E11.1", "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7", "E11.8", "E11.9"
    ]
}

def fit_pca_models(file_path_icd, file_path_desc, target_dim=514, batch_size=1000):
    """
    使用 IncrementalPCA 拟合ICD和疾病描述的PCA模型。

    参数：
    - file_path_icd: str，ICD 向量的 HDF5 文件路径
    - file_path_desc: str，疾病描述向量的 HDF5 文件路径
    - target_dim: int，PCA目标维度
    - batch_size: int，批次大小

    返回：
    - pca_icd: sklearn.decomposition.IncrementalPCA
    - pca_desc: sklearn.decomposition.IncrementalPCA
    """
    pca_icd = IncrementalPCA(n_components=target_dim, batch_size=batch_size)
    pca_desc = IncrementalPCA(n_components=target_dim, batch_size=batch_size)

    # 处理ICD向量
    with h5py.File(file_path_icd, 'r') as f_icd:
        data_icd = f_icd['data']
        total_samples_icd = data_icd.shape[0]
        for start_idx in range(0, total_samples_icd, batch_size):
            end_idx = min(start_idx + batch_size, total_samples_icd)
            batch_icd = data_icd[start_idx:end_idx, :, :]  # (batch_size, 15, feature_dim + 2)
            batch_icd_flat = batch_icd.reshape(-1, batch_icd.shape[2])  # (batch_size*15, feature_dim + 2)
            pca_icd.partial_fit(batch_icd_flat)

    # 处理疾病描述向量
    with h5py.File(file_path_desc, 'r') as f_desc:
        data_desc = f_desc['data']
        total_samples_desc = data_desc.shape[0]
        for start_idx in range(0, total_samples_desc, batch_size):
            end_idx = min(start_idx + batch_size, total_samples_desc)
            batch_desc = data_desc[start_idx:end_idx, :, :]  # (batch_size, 15, feature_dim + 2)
            batch_desc_flat = batch_desc.reshape(-1, batch_desc.shape[2])  # (batch_size*15, feature_dim + 2)
            pca_desc.partial_fit(batch_desc_flat)

    print("PCA 模型拟合完成")
    return pca_icd, pca_desc
# 数据处理和保存函数定义（示例）
def process_batch(batch_icd_codes, batch_disease_descs, batch_icd_times, max_len=15):
    batch_size = len(batch_icd_codes)
    print(f"Processing batch_size: {batch_size}")

    # 解析并展开所有 ICD 代码和疾病描述
    icd_codes_list = []
    disease_descs_list = []
    icd_time_list = []

    for icd_codes, disease_descs, icd_time_row in zip(batch_icd_codes, batch_disease_descs, batch_icd_times):
        icd_codes_split = icd_codes.strip('[]').replace("'", "").split(', ')
        disease_descs_split = disease_descs.strip('[]').replace("'", "").split(', ')

        # 处理 icd_time_row，替换 NaN 为 0.0
        icd_time_row = np.nan_to_num(icd_time_row, nan=0.0)

        # 对 ICD 代码和疾病描述进行填充或截断
        icd_codes_split = icd_codes_split[:max_len] + [''] * (max_len - len(icd_codes_split))
        disease_descs_split = disease_descs_split[:max_len] + [''] * (max_len - len(disease_descs_split))
        icd_time_row = icd_time_row[:max_len]
        if len(icd_time_row) < max_len:
            icd_time_row = np.pad(icd_time_row, (0, max_len - len(icd_time_row)), 'constant', constant_values=0.0)

        icd_codes_list.append(icd_codes_split)
        disease_descs_list.append(disease_descs_split)
        icd_time_list.append(icd_time_row)

    # 将所有文本展开为一个列表，方便批量编码
    all_icd_texts = [code if code != '' else ' ' for sample in icd_codes_list for code in sample]
    all_desc_texts = [desc if desc != '' else ' ' for sample in disease_descs_list for desc in sample]

    # 批量编码所有文本
    all_icd_features = encode_texts(all_icd_texts)
    all_desc_features = encode_texts(all_desc_texts)

    # 准备时间向量并与特征向量拼接
    batch_icd_times_processed = np.array(icd_time_list)
    print(f"batch_icd_times_processed shape: {batch_icd_times_processed.shape}")  # 应该是 (batch_size, max_len)

    if batch_icd_times_processed.shape != (batch_size, max_len):
        raise ValueError(f"batch_icd_times shape {batch_icd_times_processed.shape} does not match expected {(batch_size, max_len)}")

    icd_time_tensors = np.hstack([np.zeros((batch_size * max_len, 1)), batch_icd_times_processed.reshape(-1, 1)])  # (batch_size*15, 2)
    desc_time_tensors = np.hstack([np.ones((batch_size * max_len, 1)), batch_icd_times_processed.reshape(-1, 1)])  # (batch_size*15, 2)

    # 转换为NumPy数组
    all_icd_features_np = all_icd_features.numpy()
    all_desc_features_np = all_desc_features.numpy()

    # 打印拼接前的形状
    # print(f"icd_time_tensors shape: {icd_time_tensors.shape}")  # (batch_size*15, 2)
    # print(f"all_icd_features_np shape: {all_icd_features_np.shape}")  # (batch_size*15, feature_dim)
    # print(f"all_desc_features_np shape: {all_desc_features_np.shape}")  # (batch_size*15, feature_dim)

    # 拼接时间向量和特征向量
    icd_vectors = np.hstack([icd_time_tensors, all_icd_features_np])  # (batch_size*15, feature_dim + 2)
    desc_vectors = np.hstack([desc_time_tensors, all_desc_features_np])  # (batch_size*15, feature_dim + 2)

    # 替换 NaN 值
    icd_vectors = np.nan_to_num(icd_vectors, nan=0.0)
    desc_vectors = np.nan_to_num(desc_vectors, nan=0.0)

    # 重新组织编码后的特征，按样本分割
    icd_features_list = icd_vectors.reshape(batch_size, max_len, -1)  # (batch_size, 15, feature_dim + 2)
    desc_features_list = desc_vectors.reshape(batch_size, max_len, -1)  # (batch_size, 15, feature_dim + 2)

    return icd_features_list, desc_features_list
# 定义文本编码函数，接收文本列表并返回其向量表示
def encode_texts(texts, batch_size=128):
    # 预处理模型
    print("ICD代码和疾病描述预处理")

    # 设置设备为CUDA或CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载OpenCLIP模型用于文本特征编码
    model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    model_clip = model_clip.to(device)
    model_clip.eval()

    # 获取OpenCLIP的分词器
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    all_features = []
    num_texts = len(texts)
    num_batches = (num_texts + batch_size - 1) // batch_size
    for i in range(num_batches):
        batch_texts = texts[i * batch_size: (i + 1) * batch_size]
        tokens = tokenizer(batch_texts)
        tokens = tokens.to(device)
        with torch.cuda.amp.autocast():  # 使用混合精度
            with torch.no_grad():
                text_features = model_clip.encode_text(tokens)
        text_features = torch.nan_to_num(text_features, nan=0.0)
        # Move features to CPU and append
        all_features.append(text_features.cpu())
        # 释放GPU内存
        del tokens, text_features
        torch.cuda.empty_cache()
    all_features = torch.cat(all_features, dim=0)

    del model_clip, tokenizer
    return all_features
def get_system_mask_code(icd_codes):
    """
    将ICD代码映射为系统掩码。

    Args:
        icd_codes (list): ICD代码列表。

    Returns:
        torch.Tensor: 系统掩码张量。
    """
    mask = [0] * 7
    for code in icd_codes:
        if code in chronic_diseases['cardiovascular']:
            mask[0] = 1
        elif code in chronic_diseases['pulmonary']:
            mask[1] = 1
        elif code in chronic_diseases['musculoskeletal']:
            mask[2] = 1
        elif code in chronic_diseases['immune']:
            mask[3] = 1
        elif code in chronic_diseases['renal']:
            mask[4] = 1
        elif code in chronic_diseases['hepatic']:
            mask[5] = 1
        elif code in chronic_diseases['metabolic']:
            mask[6] = 1
    return torch.tensor(mask, dtype=torch.float32)





class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file_path):
        self.hdf5_file_path = hdf5_file_path
        self.samples = []  # 存储所有单个样本的索引

        # 打开HDF5文件并加载数据
        with h5py.File(hdf5_file_path, 'r') as f:
            # 获取所有批次的名称
            batch_names = list(f.keys())

            # 遍历每个批次
            for batch_name in batch_names:
                batch_data = f[batch_name]
                num_samples = len(batch_data['Participant_ID'])

                # 将批次中的所有样本加入samples列表
                for i in range(num_samples):
                    self.samples.append((batch_name, i))

    def __len__(self):
        # 返回数据集的总样本数量
        return len(self.samples)

    def __getitem__(self, idx):
        # 获取指定索引的样本
        batch_name, sample_idx = self.samples[idx]

        # 打开 HDF5 文件并读取数据
        with h5py.File(self.hdf5_file_path, 'r') as f:
            batch_data = f[batch_name]
            Participant_ID = batch_data['Participant_ID'][sample_idx]
            features = batch_data['features'][sample_idx]
            icd_vec = batch_data['icd_vec'][sample_idx]
            disease_vec = batch_data['disease_vec'][sample_idx]
            sex_data = batch_data['sex_data'][sample_idx]
            system_mask = batch_data['system_mask'][sample_idx]
            Survival_Time = batch_data['Survival_Time'][sample_idx]
            label = batch_data['label'][sample_idx]

        # 返回一个元组（可以根据需要转换为 tensor 类型）
        return {
            'Participant_ID': str(Participant_ID),  # 需要确保是字符串
            'features': torch.tensor(features, dtype=torch.float32),
            'icd_vec': torch.tensor(icd_vec, dtype=torch.float32),
            'disease_vec': torch.tensor(disease_vec, dtype=torch.float32),
            'sex_data': torch.tensor(sex_data, dtype=torch.float32),
            'system_mask': torch.tensor(system_mask, dtype=torch.float32),
            'Survival_Time': torch.tensor(Survival_Time, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }
# class BioAgeDataset(Dataset):
#     def __init__(self, data_dir):
#         # List all the .npz files in the data directory
#         self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
#         self.data_dir = data_dir
#
#     def __len__(self):
#         # Number of .npz files
#         return len(self.data_files)
#
#     def __getitem__(self, idx):
#         # Load a specific .npz file
#         file_path = os.path.join(self.data_dir, self.data_files[idx])
#         data = np.load(file_path)
#
#         # Extract data from the npz file
#         batch_Participant_ID = data['Participant_ID']
#         batch_features = data['features']
#         batch_icd_vec = data['icd_vec']
#         batch_disease_vec = data['disease_vec']
#         batch_sex = data['sex_data']
#         batch_system_mask = data['system_mask']
#         batch_labels = data['label']
#
#         # Convert data to torch tensors and move to the device
#         # batch_Participant_ID = torch.tensor(batch_Participant_ID, dtype=torch.string)
#         batch_Participant_ID = batch_Participant_ID.tolist()  # Convert to a list of strings
#         batch_features = torch.tensor(batch_features, dtype=torch.float32)
#         batch_icd_vec = torch.tensor(batch_icd_vec, dtype=torch.float32)
#         batch_disease_vec = torch.tensor(batch_disease_vec, dtype=torch.float32)
#         batch_sex = torch.tensor(batch_sex, dtype=torch.float32)
#         batch_system_mask = torch.tensor(batch_system_mask, dtype=torch.float32)
#         batch_labels = torch.tensor(batch_labels, dtype=torch.float32)
#
#         return (batch_Participant_ID, batch_features, batch_icd_vec, batch_disease_vec,
#                 batch_sex, batch_system_mask, batch_labels)
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
        batch_sex = data['sex_data']
        batch_system_mask = data['system_mask']
        batch_Survival_Time = data['Survival_Time']
        batch_labels = data['label']

        # Convert data to torch tensors
        batch_Participant_ID = batch_Participant_ID.tolist()  # Convert to a list of strings
        batch_features = torch.tensor(batch_features, dtype=torch.float32)
        batch_icd_vec = torch.tensor(batch_icd_vec, dtype=torch.float32)
        batch_disease_vec = torch.tensor(batch_disease_vec, dtype=torch.float32)
        batch_sex = torch.tensor(batch_sex, dtype=torch.float32)
        batch_system_mask = torch.tensor(batch_system_mask, dtype=torch.float32)
        batch_Survival_Time = torch.tensor(batch_Survival_Time, dtype=torch.float32)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float32)

        return (batch_Participant_ID, batch_features, batch_icd_vec, batch_disease_vec,
                batch_sex, batch_system_mask, batch_Survival_Time, batch_labels)


class IterableCustomDataset(IterableDataset):
    def __init__(self, Participant_ID, features, icd_file_path, desc_file_path, sex, mask_system_data, labels,
                 split='train', val_split=0.1, test_split=0.1, dataset_size=None, shuffle=False, batch_size=20000):
        self.Participant_ID = Participant_ID
        self.features = features
        self.icd_file_path = icd_file_path
        self.desc_file_path = desc_file_path
        self.sex = sex
        self.mask_system_data = mask_system_data.reset_index(drop=True)
        self.labels = labels
        self.split = split
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.batch_size = batch_size

        # 如果未提供 dataset_size，则使用 len(self.labels)
        self.dataset_size = dataset_size or len(self.labels)

        # 计算训练、验证和测试集的索引
        self.train_idx = int(self.dataset_size * (1 - val_split - test_split))
        self.val_idx = int(self.dataset_size * (1 - test_split))

        # 根据 split 类型选择索引
        if split == 'train':
            self.indices = list(range(0, self.train_idx))
        elif split == 'val':
            self.indices = list(range(self.train_idx, self.val_idx))
        elif split == 'test':
            self.indices = list(range(self.val_idx, self.dataset_size))
        else:
            raise ValueError("Invalid split type. Choose from ['train', 'val', 'test']")

        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        # 在每个工作进程中打开 HDF5 文件
        with h5py.File(self.icd_file_path, 'r') as icd_file, h5py.File(self.desc_file_path, 'r') as desc_file:
            for i in range(0, len(self.indices), self.batch_size):
                batch_indices = self.indices[i:i + self.batch_size]
                Participant_ID_batch = []
                features_batch = []
                icd_vec_batch = []
                disease_vec_batch = []
                sex_data_batch = []
                system_mask_batch = []
                label_batch = []

                # 收集批次数据
                for idx in batch_indices:
                    Participant_ID_data = self.Participant_ID[idx]
                    features = self.features[idx].astype(float)  # 假设是 NumPy 数组
                    icd_vec = icd_file['data'][idx].astype(float)
                    disease_vec = desc_file['data'][idx].astype(float)
                    sex = self.sex[idx]
                    label = self.labels[idx].astype(float)
                    codes_list = self.mask_system_data[idx]
                    system_mask = get_system_mask_code(codes_list).numpy()

                    Participant_ID_batch.append(Participant_ID_data)
                    features_batch.append(features)
                    icd_vec_batch.append(icd_vec)
                    disease_vec_batch.append(disease_vec)
                    sex_data_batch.append(sex)
                    system_mask_batch.append(system_mask)
                    label_batch.append(label)

                # 将列表转换为张量
                # 假设 Participant_ID_data 是整数
                Participant_ID_tensor = Participant_ID_batch

                features_tensor = torch.tensor(features_batch, dtype=torch.float32)
                icd_vec_tensor = torch.tensor(icd_vec_batch, dtype=torch.float32)
                disease_vec_tensor = torch.tensor(disease_vec_batch, dtype=torch.float32)
                sex_data_tensor = torch.tensor(sex_data_batch, dtype=torch.float32).unsqueeze(1)  # 假设单个值
                system_mask_tensor = torch.tensor(system_mask_batch, dtype=torch.float32)
                label_tensor = torch.tensor(label_batch, dtype=torch.float32)

                yield (
                    Participant_ID_tensor,
                    features_tensor,
                    icd_vec_tensor,
                    disease_vec_tensor,
                    sex_data_tensor,
                    system_mask_tensor,
                    label_tensor
                )

# class IterableCustomDataset(IterableDataset):
#     def __init__(self, Participant_ID, features, icd_file_path, desc_file_path, sex, mask_system_data, labels,
#                  split='train', val_split=0.1, test_split=0.1, dataset_size=None, shuffle=False):
#         self.Participant_ID = Participant_ID
#         self.features = features
#         self.icd_file_path = icd_file_path
#         self.desc_file_path = desc_file_path
#         self.sex = sex
#         self.mask_system_data = mask_system_data.reset_index(drop=True)
#         self.labels = labels
#         self.split = split
#         self.val_split = val_split
#         self.test_split = test_split
#         self.shuffle = shuffle
#
#         # If dataset_size is passed, use it, otherwise fallback to len(self.labels)
#         self.dataset_size = dataset_size or len(self.labels)
#
#         # Calculate train, validation, and test indices
#         self.train_idx = int(self.dataset_size * (1 - val_split - test_split))
#         self.val_idx = int(self.dataset_size * (1 - test_split))
#
#         # Determine the subset of indices based on the split type
#         if split == 'train':
#             self.indices = list(range(0, self.train_idx))
#         elif split == 'val':
#             self.indices = list(range(self.train_idx, self.val_idx))
#         elif split == 'test':
#             self.indices = list(range(self.val_idx, self.dataset_size))
#         else:
#             raise ValueError("Invalid split type. Choose from ['train', 'val', 'test']")
#
#         if self.shuffle:
#             random.shuffle(self.indices)
#
#         # Initialize file references to None
#         self.icd_file = None
#         self.desc_file = None
#
#     def _open_files(self):
#         """Open the HDF5 files once to avoid opening them repeatedly."""
#         if self.icd_file is None:
#             self.icd_file = h5py.File(self.icd_file_path, 'r')
#         if self.desc_file is None:
#             self.desc_file = h5py.File(self.desc_file_path, 'r')
#
#     def _close_files(self):
#         """Close the HDF5 files after all data is processed."""
#         if self.icd_file is not None:
#             self.icd_file.close()
#         if self.desc_file is not None:
#             self.desc_file.close()
#
#     def _get_data(self, idx):
#         """Helper function to fetch data for a given index."""
#         Participant_ID_data = self.Participant_ID[idx]
#         features = torch.tensor(self.features[idx], dtype=torch.float32)
#         icd_vec = torch.tensor(self.icd_file['data'][idx], dtype=torch.float32)
#         disease_vec = torch.tensor(self.desc_file['data'][idx], dtype=torch.float32)
#         sex_data = torch.tensor([self.sex[idx]], dtype=torch.float32)
#         label = torch.tensor(self.labels[idx], dtype=torch.float32)
#         codes_list = self.mask_system_data[idx]
#         system_mask = get_system_mask_code(codes_list)
#
#         return (
#             Participant_ID_data,
#             features,
#             icd_vec,
#             disease_vec,
#             sex_data,
#             system_mask,
#             label
#         )
#
#     def __iter__(self):
#         """Override the __iter__ method to yield one data sample at a time."""
#         self._open_files()
#
#         # Yield data for the current split (train/val/test)
#         for idx in self.indices:
#             yield self._get_data(idx)
#
#         self._close_files()
# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, Participant_ID, features, icd_file_path, desc_file_path, sex, mask_system_data, labels):
        self.Participant_ID = Participant_ID
        self.features = features
        self.icd_file_path = icd_file_path
        self.desc_file_path = desc_file_path
        self.sex = sex
        self.mask_system_data = mask_system_data.reset_index(drop=True)
        self.labels = labels

        # 初始化文件引用为None，将在__getitem__中打开
        self.icd_file = None
        self.desc_file = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        start_time = time.time()
        # 如果尚未打开HDF5文件，则打开
        if self.icd_file is None:
            self.icd_file = h5py.File(self.icd_file_path, 'r')
        if self.desc_file is None:
            self.desc_file = h5py.File(self.desc_file_path, 'r')

        # print(idx)
        # print('打开HDF5文件:', time.time() - start_time)
        Participant_ID_data = self.Participant_ID[idx]
        # print('Participant_ID_data:', time.time() - start_time)
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        # print('features:', time.time() - start_time)
        icd_vec = torch.tensor(self.icd_file['data'][idx], dtype=torch.float32)
        # print('icd_vec:', time.time() - start_time)
        disease_vec = torch.tensor(self.desc_file['data'][idx], dtype=torch.float32)
        # print('disease_vec:', time.time() - start_time)
        sex_data = torch.tensor([self.sex[idx]], dtype=torch.float32)
        # print('sex_data:', time.time() - start_time)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        # print('label:', time.time() - start_time)
        codes_list = self.mask_system_data[idx]
        system_mask = get_system_mask_code(codes_list)

        # print('system_mask:', time.time() - start_time)
        return (
            Participant_ID_data,
            features,
            icd_vec,
            disease_vec,
            sex_data,
            system_mask,
            label
        )


def clean_mask_system_data(mask_system_series):
    """
    清洗mask_system_data，确保所有条目都是有效的列表。
    无效条目将被替换为一个空列表或其他默认值。

    Args:
        mask_system_series (pd.Series): 原始mask_system数据。

    Returns:
        pd.Series: 清洗后的mask_system数据。
    """
    cleaned_data = []
    error_indices = []
    error_entries = []
    for idx, codes in enumerate(mask_system_series):
        if isinstance(codes, list):
            cleaned_list = [code for code in codes if isinstance(code, str)]
            if len(cleaned_list) != len(codes):
                print(f"Warning: mask_system_data at index {idx} contains non-string entries. They have been removed.")
                error_indices.append(idx)
                error_entries.append(codes)
            cleaned_data.append(cleaned_list)
        elif isinstance(codes, str):
            try:
                codes_list = ast.literal_eval(codes)
                if not isinstance(codes_list, list):
                    print(f"Warning: mask_system_data at index {idx} is not a list after literal_eval. Replacing with empty list.")
                    codes_list = []
                else:
                    codes_list = [code for code in codes_list if isinstance(code, str)]
            except Exception as e:
                print(f"Error parsing mask_system_data at index {idx}: {e}")
                print(f"Invalid data: {codes}")
                codes_list = []
                error_indices.append(idx)
                error_entries.append(codes)
            cleaned_data.append(codes_list)
        else:
            print(f"Warning: mask_system_data at index {idx} is neither string nor list. Replacing with empty list.")
            codes_list = []
            cleaned_data.append(codes_list)
            error_indices.append(idx)
            error_entries.append(codes)

    if error_indices:
        with open('mask_system_data_errors.log', 'w') as f:
            for idx, entry in zip(error_indices, error_entries):
                f.write(f"Index {idx}: {entry}\n")
        print(f"Found and logged {len(error_indices)} malformed mask_system_data entries.")

    return pd.Series(cleaned_data)

def preprocess_mask_system_data(mask_system_series):
    """
    预处理mask_system_data，确保每个条目都是有效的列表格式。
    无效条目将被替换为一个空列表。

    Args:
        mask_system_series (pd.Series): 原始mask_system数据。

    Returns:
        pd.Series: 预处理后的mask_system数据。
    """
    processed_mask_system = []
    for idx, codes_str in enumerate(mask_system_series):
        if isinstance(codes_str, str):
            try:
                codes_list = ast.literal_eval(codes_str)
                if not isinstance(codes_list, list):
                    print(f"Warning: mask_system_data at index {idx} is not a list. Replacing with empty list.")
                    codes_list = []
            except Exception as e:
                print(f"Error parsing mask_system_data at index {idx}: {e}")
                print(f"Invalid data: {codes_str}")
                codes_list = []
        elif isinstance(codes_str, list):
            codes_list = codes_str
        else:
            print(f"Warning: mask_system_data at index {idx} is neither string nor list. Replacing with empty list.")
            codes_list = []
        processed_mask_system.append(codes_list)
    return pd.Series(processed_mask_system)
