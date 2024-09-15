import pandas as pd
import numpy as np 
import os 
import json 
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from get_data import normalize_and_fill_missing
import pdb


def create_dataloaders(x1, surv_time, surv_status, cv_splits, rep, fold, batch_size, device):
    """
    Create DataLoader for each set using indices from cv_splits
    :param x1: tensor of first modality data 
    :param x2: tensor of second modality data 
    :param surv_time: tensor of survival times
    :param surv_status: tensor of censor statuses
    :param cv_splits: dictionary of cross-validation indices
    :param rep: repetition 
    :param fold: fold 
    :param batch_size: batch size for DataLoader
    """
    train_indices = cv_splits[rep][fold]['train']
    val_indices = cv_splits[rep][fold]['val']
    test_indices = cv_splits[rep][fold]['test']

    # normalize the datasets
    x1_train, x1_val, x1_test = normalize_and_fill_missing(x1[train_indices], x1[val_indices], x1[test_indices])
    
    train_dataset = UnimodalDataset(torch.FloatTensor(x1_train).to(device), 
                                  torch.FloatTensor(surv_time[train_indices]).to(device), 
                                  torch.FloatTensor(surv_status[train_indices]).to(device))
    val_dataset = UnimodalDataset(torch.FloatTensor(x1_val).to(device), 
                                torch.FloatTensor(surv_time[val_indices]).to(device), 
                                torch.FloatTensor(surv_status[val_indices]).to(device))
    test_dataset = UnimodalDataset(torch.FloatTensor(x1_test).to(device), 
                                 torch.FloatTensor(surv_time[test_indices]).to(device), 
                                 torch.FloatTensor(surv_status[test_indices]).to(device))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class UnimodalDataset(Dataset):
    def __init__(self, x1, surv_time, censor_status):
        """
        Initialize the dataset with subsets of data based on indices provided.
        Args: 
            x1 (numpy.ndarray): data from modality 1 
        """
        self.x1 = x1
        self.surv_time = surv_time
        self.censor_status = censor_status

    def __len__(self):
        return len(self.surv_time)

    def __getitem__(self, idx):
        return (self.x1[idx], self.surv_time[idx], self.censor_status[idx])


# ========= Functions to load individual dataset =========

def get_data(data_dir_path, dataset, modality): 
    """Loads features and outcomes 
    Args: 
        data_dir_path (str): path to a folder containing all datasets 
        dataset (str): dataset name
    """
    if dataset == 'lung_radiopathomic':
        return load_lung_radiopathomic(os.path.join(data_dir_path, dataset), modality)
    elif dataset == 'lung_radiogenomic':
        return load_lung_radiogenomic(os.path.join(data_dir_path, dataset), modality)
    elif dataset == 'brain_radiogenomic': 
        return load_brain_radiogenomic(os.path.join(data_dir_path, dataset), modality)
    elif dataset == 'prostate_t2w_adc': 
        return load_prostate_t2w_adc(os.path.join(data_dir_path, 'prostate_t2w_adc'), modality)
    else: 
        raise NotImplementedError('Double check dataset name')

def load_prostate_t2w_adc(data_dir_path, modality): 
    # read status and time 
    outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)
    overall_time_to_progression = outcome_df['time'].values
    overall_progression_status = outcome_df['status'].values

    # read data 
    if modality == 't2w':
        t2w_df = pd.read_csv(os.path.join(data_dir_path, 'T2_radiomic_features.csv'), index_col=0)
        data = t2w_df.drop(columns=['coded_mrn', 'Lesion_ID'], errors='ignore').values 
    elif modality == 'adc':
        adc_df = pd.read_csv(os.path.join(data_dir_path, 'ADC_radiomic_features.csv'), index_col=0)
        data = adc_df.drop(columns=['coded_mrn', 'Lesion_ID'], errors='ignore').values 
    elif modality == 'capras': 
        capras_df = pd.read_csv(os.path.join(data_dir_path, 'CAPRAS_data.csv'), index_col=0)
        data = capras_df[['CAPRAS_Gleason', 'CAPRAS_PSA', 'CAPRAS_SM', 'CAPRAS_SVI', 'CAPRAS_ECE', 'CAPRAS_LNI']].values

    # load repeated cv splits 
    with open(os.path.join(data_dir_path, 'cv_splits.json'), 'r') as f: 
        cv_splits = json.load(f)

    return data, overall_progression_status, overall_time_to_progression, cv_splits
    
def load_brain_radiogenomic(data_dir_path, modality): 
    # load data from csv files 
    if modality == 'radiomic':
        radiomic_df = pd.read_csv(os.path.join(data_dir_path, 'radiomic.csv'), index_col=0)
        data = radiomic_df.values 
    elif modality == 'genomic':
        genomic_df = pd.read_csv(os.path.join(data_dir_path, 'genomic.csv'), index_col=0)
        data = genomic_df.values 

    outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)
    overall_time_to_progression = outcome_df['days'].values
    overall_progression_status = outcome_df['status'].values

    with open(os.path.join(data_dir_path, 'cv_splits.json'), 'r') as f: 
        cv_splits = json.load(f)
    return data, overall_progression_status, overall_time_to_progression, cv_splits

def load_lung_radiopathomic(data_dir_path, modality):
    # Load data from CSV files
    if modality == 'radiomic':
        radiomics_df = pd.read_csv(os.path.join(data_dir_path, 'all_radiomics.csv'), index_col=0)
        radiomics_df = radiomics_df.drop(columns='PatientID', errors='ignore')
        data = radiomics_df.values 
    elif modality == 'pathomic':
        pathomics_df = pd.read_csv(os.path.join(data_dir_path, 'all_pathomics.csv'), index_col=0)
        pathomics_df = pathomics_df.dropna(axis=1, how='all') # Drop columns with all NA values
        pathomics_df = pathomics_df.drop(columns='PatientID', errors='ignore')
        data = pathomics_df.values
    
    # read outcome 
    outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)
    overall_time_to_progression = outcome_df['time'].values
    overall_progression_status = outcome_df['status'].values

    # read splits 
    with open(os.path.join(data_dir_path, 'cv_splits.json'), 'r') as f: 
        cv_splits = json.load(f)
    return data, overall_progression_status, overall_time_to_progression, cv_splits

def load_lung_radiogenomic(data_dir_path, modality):
    if modality == 'radiomic':
        radiomic_df = pd.read_csv(os.path.join(data_dir_path, 'radiomic.csv'), index_col=0)
        data = radiomic_df.values
    elif modality == 'genomic':
        genomic_df = pd.read_csv(os.path.join(data_dir_path, 'genomic.csv'), index_col=0)
        data = genomic_df.values
    
    # read outcome 
    outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)
    time = outcome_df['time'].values
    status = outcome_df['status'].values

    # read splits 
    with open(os.path.join(data_dir_path, 'cv_splits.json'), 'r') as f: 
        cv_splits = json.load(f)
    return data, status, time, cv_splits 