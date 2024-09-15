import pandas as pd
import numpy as np 
import os 
import json 
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import pdb

def normalize_and_fill_missing(train_data, val_data, test_data):
    """
    Fills missing values with the mean from the training data and normalizes the datasets.
    
    Args:
        train_data (numpy.ndarray): Training dataset with potential missing values.
        val_data (numpy.ndarray): Validation dataset to be filled and normalized.
        test_data (numpy.ndarray): Test dataset to be filled and normalized.
    
    Returns:
        train_norm (numpy.ndarray): Normalized training dataset with missing values filled.
        val_norm (numpy.ndarray): Normalized validation dataset with missing values filled.
        test_norm (numpy.ndarray): Normalized test dataset with missing values filled.
    """
    # fill missing values with mean on the training set 
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(train_data)
    train_data = imp.transform(train_data)
    val_data = imp.transform(val_data)
    test_data = imp.transform(test_data)
    
    # normalize data 
    scaler = StandardScaler()
    scaler.fit(train_data)  
    train_norm = scaler.transform(train_data)
    val_norm = scaler.transform(val_data)
    test_norm = scaler.transform(test_data)

    return train_norm, val_norm, test_norm

def create_dataloaders(x1, x2, surv_time, surv_status, cv_splits, rep, fold, batch_size, device, x3=None):
    """
    Creates DataLoaders for training, validation, and testing datasets using indices from cv_splits.
    
    Args:
        x1 (torch.Tensor): Tensor of the first modality data.
        x2 (torch.Tensor): Tensor of the second modality data.
        surv_time (torch.Tensor): Tensor of survival times for each sample.
        surv_status (torch.Tensor): Tensor of event/censor statuses for each sample.
        cv_splits (dict): Dictionary containing cross-validation indices for train, validation, and test sets.
        rep (int): Repetition number for cross-validation.
        fold (int): Fold number for cross-validation.
        batch_size (int): Batch size for DataLoader.
        device (torch.device): Device (e.g., 'cuda' or 'cpu') to which the data will be transferred.
        x3 (torch.Tensor, optional): Tensor of the third modality data, if applicable. Default is None.
    
    Returns:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
    """
    train_indices = cv_splits[rep][fold]['train']
    val_indices = cv_splits[rep][fold]['val']
    test_indices = cv_splits[rep][fold]['test']

    # normalize the datasets
    x1_train, x1_val, x1_test = normalize_and_fill_missing(x1[train_indices], x1[val_indices], x1[test_indices])
    x2_train, x2_val, x2_test = normalize_and_fill_missing(x2[train_indices], x2[val_indices], x2[test_indices])
    if not x3 is None: 
        x3_train, x3_val, x3_test = normalize_and_fill_missing(x3[train_indices], x3[val_indices], x3[test_indices])
        train_dataset = Dataset3Mod(torch.FloatTensor(x1_train).to(device), 
                                    torch.FloatTensor(x2_train).to(device), 
                                    torch.FloatTensor(x3_train).to(device), 
                                    torch.FloatTensor(surv_time[train_indices]).to(device), 
                                    torch.FloatTensor(surv_status[train_indices]).to(device))
        val_dataset = Dataset3Mod(torch.FloatTensor(x1_val).to(device), 
                                    torch.FloatTensor(x2_val).to(device), 
                                    torch.FloatTensor(x3_val).to(device), 
                                    torch.FloatTensor(surv_time[val_indices]).to(device), 
                                    torch.FloatTensor(surv_status[val_indices]).to(device))
        test_dataset = Dataset3Mod(torch.FloatTensor(x1_test).to(device), 
                                    torch.FloatTensor(x2_test).to(device), 
                                    torch.FloatTensor(x3_test).to(device), 
                                    torch.FloatTensor(surv_time[test_indices]).to(device), 
                                    torch.FloatTensor(surv_status[test_indices]).to(device))
    else: 
        train_dataset = Dataset2Mod(torch.FloatTensor(x1_train).to(device), 
                                    torch.FloatTensor(x2_train).to(device), 
                                    torch.FloatTensor(surv_time[train_indices]).to(device), 
                                    torch.FloatTensor(surv_status[train_indices]).to(device))
        val_dataset = Dataset2Mod(torch.FloatTensor(x1_val).to(device), 
                                    torch.FloatTensor(x2_val).to(device), 
                                    torch.FloatTensor(surv_time[val_indices]).to(device), 
                                    torch.FloatTensor(surv_status[val_indices]).to(device))
        test_dataset = Dataset2Mod(torch.FloatTensor(x1_test).to(device), 
                                    torch.FloatTensor(x2_test).to(device), 
                                    torch.FloatTensor(surv_time[test_indices]).to(device), 
                                    torch.FloatTensor(surv_status[test_indices]).to(device))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

class Dataset2Mod(Dataset):
    def __init__(self, x1, x2, surv_time, censor_status):
        """
        Initialize the dataset with subsets of data based on indices provided.
        Args: 
            x1 (numpy.ndarray): data from modality 1 
            x2 (numpy.ndarray): data from modality 2 
        """
        self.x1 = x1
        self.x2 = x2
        self.surv_time = surv_time
        self.censor_status = censor_status

    def __len__(self):
        return len(self.surv_time)

    def __getitem__(self, idx):
        return (self.x1[idx], self.x2[idx], self.surv_time[idx], self.censor_status[idx])

class Dataset3Mod(Dataset):
    def __init__(self, x1, x2, x3, surv_time, censor_status):
        """
        Initialize the dataset with subsets of data based on indices provided.
        Args: 
            x1 (numpy.ndarray): data from modality 1 
            x2 (numpy.ndarray): data from modality 2 
        """
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3 
        self.surv_time = surv_time
        self.censor_status = censor_status

    def __len__(self):
        return len(self.surv_time)

    def __getitem__(self, idx):
        return (self.x1[idx], self.x2[idx], self.x3[idx], self.surv_time[idx], self.censor_status[idx])


# ========= Functions to load individual dataset =========

def get_data(data_dir_path, dataset): 
    """Loads features and outcomes 
    Args: 
        data_dir_path (str): path to a folder containing all datasets 
        dataset (str): dataset name
    Returns: 
        numpy.ndarray: Array of features from modality 1 
        numpy.ndarray: Array of features from modality 2 
        numpy.ndarray: Binary status indicating event occurrence 
        numpy.ndarray: Array of time-to-event data.
        cv_splits (dict): Dictionary containing predefined cross-validation splits for training/validation.
    """
    if dataset == 'lung_radiopathomic':
        return load_lung_radiopathomic(os.path.join(data_dir_path, dataset))
    elif dataset == 'lung_radiogenomic':
        return load_lung_radiogenomic(os.path.join(data_dir_path, dataset))
    elif dataset == 'brain_radiogenomic': 
        return load_brain_radiogenomic(os.path.join(data_dir_path, dataset))
    elif dataset == 'prostate_capras_t2w':
        return load_prostate_capras_t2w(os.path.join(data_dir_path, 'prostate_t2w_adc'))
    elif dataset == 'prostate_capras_adc': 
        return load_prostate_capras_adc(os.path.join(data_dir_path, 'prostate_t2w_adc'))
    elif dataset == 'prostate_t2w_adc': 
        return load_prostate_t2w_adc(os.path.join(data_dir_path, 'prostate_t2w_adc'))
    elif dataset == 'prostate_capras_t2w_adc': 
        return prostate_capras_t2w_adc(os.path.join(data_dir_path, 'prostate_t2w_adc'))
    elif dataset == 'prostate_capras_imaging': 
        return prostate_capras_imaging(os.path.join(data_dir_path, 'prostate_t2w_adc'))
    else: 
        raise NotImplementedError('Double check dataset name')

def prostate_capras_t2w_adc(data_dir_path): 
    """
    Loads the prostate CAPRAS-imaging dataset, including CAPRAS features, 
    radiomic features from both T2W and ADC modalities respectively, clinical outcomes, and CV splits.
    
    Args:
    data_dir_path (str): Path to the directory where the dataset files are stored. 
    
    Returns:
        t2w (numpy.ndarray): Array of T2W radiomic feature 
        adc (numpy.ndarray): Array of ADC radiomic feature 
        capras (numpy.ndarray): Array of CAPRAS feature 
        overall_progression_status (numpy.ndarray): Binary status indicating event occurrence 
        overall_time_to_progression (numpy.ndarray): Array of time-to-event data.
        cv_splits (dict): Dictionary containing predefined cross-validation splits for training/validation.
    """
    t2w_df = pd.read_csv(os.path.join(data_dir_path, 'T2_radiomic_features.csv'), index_col=0)
    adc_df = pd.read_csv(os.path.join(data_dir_path, 'ADC_radiomic_features.csv'), index_col=0)
    capras_df = pd.read_csv(os.path.join(data_dir_path, 'CAPRAS_data.csv'), index_col=0)
    outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)

    t2w = t2w_df.drop(columns=['coded_mrn', 'Lesion_ID'], errors='ignore').values 
    adc = adc_df.drop(columns=['coded_mrn', 'Lesion_ID'], errors='ignore').values 
    capras = capras_df[['CAPRAS_Gleason', 'CAPRAS_PSA', 'CAPRAS_SM', 'CAPRAS_SVI', 'CAPRAS_ECE', 'CAPRAS_LNI']].values

    overall_time_to_progression = outcome_df['time'].values
    overall_progression_status = outcome_df['status'].values

    with open(os.path.join(data_dir_path, 'cv_splits.json'), 'r') as f: 
        cv_splits = json.load(f)

    return t2w, adc, capras, overall_progression_status, overall_time_to_progression, cv_splits

def prostate_capras_imaging(data_dir_path): 
    """
    Loads the prostate CAPRAS-imaging dataset, including CAPRAS features, 
    radiomic features from both T2W and ADC modalities, clinical outcomes, and CV splits.
    
    Args:
    data_dir_path (str): Path to the directory where the dataset files are stored. 
    
    Returns:
        imaging (numpy.ndarray): Array of concatenated T2W and ADC radiomic feature 
        capras (numpy.ndarray): Array of CAPRAS feature 
        overall_progression_status (numpy.ndarray): Binary status indicating event occurrence 
        overall_time_to_progression (numpy.ndarray): Array of time-to-event data.
        cv_splits (dict): Dictionary containing predefined cross-validation splits for training/validation.
    """
    t2w_df = pd.read_csv(os.path.join(data_dir_path, 'T2_radiomic_features.csv'), index_col=0)
    adc_df = pd.read_csv(os.path.join(data_dir_path, 'ADC_radiomic_features.csv'), index_col=0)
    capras_df = pd.read_csv(os.path.join(data_dir_path, 'CAPRAS_data.csv'), index_col=0)
    outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)

    t2w = t2w_df.drop(columns=['coded_mrn', 'Lesion_ID'], errors='ignore').values 
    adc = adc_df.drop(columns=['coded_mrn', 'Lesion_ID'], errors='ignore').values 
    imaging = np.concatenate((t2w, adc), axis=1)
    capras = capras_df[['CAPRAS_Gleason', 'CAPRAS_PSA', 'CAPRAS_SM', 'CAPRAS_SVI', 'CAPRAS_ECE', 'CAPRAS_LNI']].values

    overall_time_to_progression = outcome_df['time'].values
    overall_progression_status = outcome_df['status'].values

    with open(os.path.join(data_dir_path, 'cv_splits.json'), 'r') as f: 
        cv_splits = json.load(f)

    return imaging, capras, overall_progression_status, overall_time_to_progression, cv_splits

def load_prostate_t2w_adc(data_dir_path): 
    """
    Loads the prostate CAPRAS-imaging dataset, including T2W radiomic features, 
    ADC radiomic features, clinical outcomes, and CV splits.
    
    Args:
    data_dir_path (str): Path to the directory where the dataset files are stored. 
    
    Returns:
        t2w (numpy.ndarray): Array of T2W feature 
        adc (numpy.ndarray): Array of ADC radiomic feature
        overall_progression_status (numpy.ndarray): Binary status indicating event occurrence 
        overall_time_to_progression (numpy.ndarray): Array of time-to-event data.
        cv_splits (dict): Dictionary containing predefined cross-validation splits for training/validation.
    """
    t2w_df = pd.read_csv(os.path.join(data_dir_path, 'T2_radiomic_features.csv'), index_col=0)
    adc_df = pd.read_csv(os.path.join(data_dir_path, 'ADC_radiomic_features.csv'), index_col=0)
    outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)

    t2w = t2w_df.drop(columns=['coded_mrn', 'Lesion_ID'], errors='ignore').values 
    adc = adc_df.drop(columns=['coded_mrn', 'Lesion_ID'], errors='ignore').values 

    overall_time_to_progression = outcome_df['time'].values
    overall_progression_status = outcome_df['status'].values

    with open(os.path.join(data_dir_path, 'cv_splits.json'), 'r') as f: 
        cv_splits = json.load(f)

    return t2w, adc, overall_progression_status, overall_time_to_progression, cv_splits


def load_prostate_capras_adc(data_dir_path):
    """
    Loads the prostate CAPRAS-imaging dataset, including CAPRAS features, 
    radiomic features from ADC modality, clinical outcomes, and CV splits.
    
    Args:
    data_dir_path (str): Path to the directory where the dataset files are stored. 
    
    Returns:
        capras (numpy.ndarray): Array of CAPRAS feature 
        adc (numpy.ndarray): Array of ADC radiomic feature
        overall_progression_status (numpy.ndarray): Binary status indicating event occurrence 
        overall_time_to_progression (numpy.ndarray): Array of time-to-event data.
        cv_splits (dict): Dictionary containing predefined cross-validation splits for training/validation.
    """
    capras_df = pd.read_csv(os.path.join(data_dir_path, 'CAPRAS_data.csv'), index_col=0)
    adc_df = pd.read_csv(os.path.join(data_dir_path, 'ADC_radiomic_features.csv'), index_col=0)
    outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)

    capras = capras_df[['CAPRAS_Gleason', 'CAPRAS_PSA', 'CAPRAS_SM', 'CAPRAS_SVI', 'CAPRAS_ECE', 'CAPRAS_LNI']].values
    adc = adc_df.drop(columns=['coded_mrn', 'Lesion_ID'], errors='ignore').values 
    overall_time_to_progression = outcome_df['time'].values
    overall_progression_status = outcome_df['status'].values
    
    with open(os.path.join(data_dir_path, 'cv_splits.json'), 'r') as f: 
        cv_splits = json.load(f)

    return capras, adc, overall_progression_status, overall_time_to_progression, cv_splits
    
def load_prostate_capras_t2w(data_dir_path):
    """
    Loads the prostate CAPRAS-imaging dataset, including CAPRAS features, 
    radiomic features from T2W modality, clinical outcomes, and CV splits.
    
    Args:
    data_dir_path (str): Path to the directory where the dataset files are stored. 
    
    Returns:
        capras (numpy.ndarray): Array of CAPRAS feature 
        t2w (numpy.ndarray): Array of t2w radiomic feature
        overall_progression_status (numpy.ndarray): Binary status indicating event occurrence 
        overall_time_to_progression (numpy.ndarray): Array of time-to-event data.
        cv_splits (dict): Dictionary containing predefined cross-validation splits for training/validation.
    """
    capras_df = pd.read_csv(os.path.join(data_dir_path, 'CAPRAS_data.csv'), index_col=0)
    t2w_df = pd.read_csv(os.path.join(data_dir_path, 'T2_radiomic_features.csv'), index_col=0)
    outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)

    capras = capras_df[['CAPRAS_Gleason', 'CAPRAS_PSA', 'CAPRAS_SM', 'CAPRAS_SVI', 'CAPRAS_ECE', 'CAPRAS_LNI']].values
    t2w = t2w_df.drop(columns=['coded_mrn', 'Lesion_ID'], errors='ignore').values 
    overall_time_to_progression = outcome_df['time'].values
    overall_progression_status = outcome_df['status'].values
    
    with open(os.path.join(data_dir_path, 'cv_splits.json'), 'r') as f: 
        cv_splits = json.load(f)

    return capras, t2w, overall_progression_status, overall_time_to_progression, cv_splits
    
def load_brain_radiogenomic(data_dir_path): 
    """
    Loads the brain radiogenomic dataset, including radiomic features, 
    genomic features, clinical outcomes, and CV splits.
    
    Args:
    data_dir_path (str): Path to the directory where the dataset files are stored. 
    
    Returns:
        radiomic (numpy.ndarray): Array of radiomic feature 
        genomic (numpy.ndarray): Array of genomic feature
        overall_progression_status (numpy.ndarray): Binary status indicating event occurrence 
        overall_time_to_progression (numpy.ndarray): Array of time-to-event data.
        cv_splits (dict): Dictionary containing predefined cross-validation splits for training/validation.
    """
    # load data from csv files 
    radiomic_df = pd.read_csv(os.path.join(data_dir_path, 'radiomic.csv'), index_col=0)
    genomic_df = pd.read_csv(os.path.join(data_dir_path, 'genomic.csv'), index_col=0)
    outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)

    radiomic = radiomic_df.values 
    genomic = genomic_df.values 

    overall_time_to_progression = outcome_df['days'].values
    overall_progression_status = outcome_df['status'].values

    with open(os.path.join(data_dir_path, 'cv_splits.json'), 'r') as f: 
        cv_splits = json.load(f)
    return radiomic, genomic, overall_progression_status, overall_time_to_progression, cv_splits

def load_lung_radiopathomic(data_dir_path):
    """
    Loads the lung radiopathomic dataset, including radiomic features, 
    pathomic features, clinical outcomes, and CV splits.
    
    Args:
    data_dir_path (str): Path to the directory where the dataset files are stored. 
    
    Returns:
        radiomics (numpy.ndarray): Array of radiomic feature 
        pathomics (numpy.ndarray): Array of pathomic feature
        overall_progression_status (numpy.ndarray): Binary status indicating event occurrence 
        overall_time_to_progression (numpy.ndarray): Array of time-to-event data.
        cv_splits (dict): Dictionary containing predefined cross-validation splits for training/validation.
    """
    # Load data from CSV files
    radiomics_df = pd.read_csv(os.path.join(data_dir_path, 'all_radiomics.csv'), index_col=0)
    pathomics_df = pd.read_csv(os.path.join(data_dir_path, 'all_pathomics.csv'), index_col=0)
    outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)

    # Drop columns with all NA values
    pathomics_df = pathomics_df.dropna(axis=1, how='all')

    # Extract outcome data after checking for NaN and inf
    overall_time_to_progression = outcome_df['time'].values
    overall_progression_status = outcome_df['status'].values

    # Drop 'PatientID' column if it exists
    radiomics_df = radiomics_df.drop(columns='PatientID', errors='ignore')
    pathomics_df = pathomics_df.drop(columns='PatientID', errors='ignore')

    # Check for NaN and inf values in radiomics and pathomics dataframes
    for df in [radiomics_df, pathomics_df]:
        if df.isna().any().any() or df.isin([float('inf'), -float('inf')]).any().any():
            raise ValueError(f"Warning: NaN or inf found in {df.name} data")

    radiomics = radiomics_df.values 
    pathomics = pathomics_df.values 

    with open(os.path.join(data_dir_path, 'cv_splits.json'), 'r') as f: 
        cv_splits = json.load(f)
    return radiomics, pathomics, overall_progression_status, overall_time_to_progression, cv_splits

def load_lung_radiogenomic(data_dir_path):
    """
    Loads the lung radiogenomic dataset, including radiomic features, 
    genomic features, clinical outcomes, and CV splits.
    
    Args:
    data_dir_path (str): Path to the directory where the dataset files are stored. 
    
    Returns:
        radiomic_features (numpy.ndarray): Array of radiomic feature 
        genomic_features (numpy.ndarray): Array of genomic feature
        status (numpy.ndarray): Binary status indicating event occurrence 
        time (numpy.ndarray): Array of time-to-event data.
        cv_splits (dict): Dictionary containing predefined cross-validation splits for training/validation.
    """
    radiomic_df = pd.read_csv(os.path.join(data_dir_path, 'radiomic.csv'), index_col=0)
    genomic_df = pd.read_csv(os.path.join(data_dir_path, 'genomic.csv'), index_col=0)
    outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)

    # Extract outcome data after checking for NaN and inf
    time = outcome_df['time'].values
    status = outcome_df['status'].values

    radiomic_features = radiomic_df.values
    genomic_features = genomic_df.values

    with open(os.path.join(data_dir_path, 'cv_splits.json'), 'r') as f: 
        cv_splits = json.load(f)
    return radiomic_features, genomic_features, status, time, cv_splits 