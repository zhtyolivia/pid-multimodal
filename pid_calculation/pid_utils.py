import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

def clustering(X, pca=False, n_clusters=20, n_components=5, random_state=None):
    X = np.nan_to_num(X)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0],-1)
    if pca:
        # print(np.any(np.isnan(X)), np.all(np.isfinite(X)))
        X = normalize(X)
        X = PCA(n_components=n_components).fit_transform(X)
    if not random_state is None: 
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    else: 
        kmeans = KMeans(n_clusters=n_clusters).fit(X)
    return kmeans.labels_, X

def histogram_map_to_bin(number_of_bins, time): 
    """
    Convert a continuous variable into discrete. 
    Params: 
        - number_of_bins: int. the number of evenly spaced bins. 
          Should be set to cubed-root of the number of samples 
          in the context of PID metrics calculation. 
        - time: numpy.array. values of continuous variable. 
    Returns: 
        - time_in_bins
    """
    # Specify the number of bins you want
    # cube root of the number of samples 

    # Use numpy.histogram to get bins based on the histogram
    hist, bin_edges = np.histogram(time, bins=number_of_bins)

    # Use numpy.digitize to assign each time value to a bin
    time_in_bins = np.digitize(time, bin_edges, right=False).reshape(-1, 1)
    
    return time_in_bins

def parse_pca(customize, pca_components, dataset): 
    if customize: 
        if dataset == "lung_radiopathomic":
            x1_pca = 5 
            x2_pca = 30 
        elif dataset == "lung_radiogenomic":
            x1_pca = 2 
            x2_pca = 80 
        elif dataset == "brain_radiogenomic":
            x1_pca = 9 
            x2_pca = 45
        elif dataset == "prostate_capras_t2w":
            print(f"No customized PCA for prostate_capras_t2w dataset.")
            sys.exit(1)
        elif dataset == "prostate_capras_adc":
            print(f"No customized PCA for prostate_capras_adc dataset.")
            sys.exit(1)
        elif dataset == "prostate_t2w_adc":
            print(f"No customized PCA for prostate_t2w_adc dataset.")
            sys.exit(1)
        else :
            print("Dataset not recognized.")
            sys.exit(1)
    else: # not customizing PCA components 
        x1_pca = pca_components 
        x2_pca = pca_components 
    return x1_pca, x2_pca 

def get_num_bins(dataset):
    if dataset == "lung_radiopathomic":
        return 5
    elif dataset == "lung_radiogenomic":
        return 7 # cubed-root of 310 
    elif dataset == "brain_radiogenomic":
        return 5 # cubed-root of 140 
    elif dataset == "prostate_capras_t2w":
        return 7 
    elif dataset == "prostate_capras_adc":
        return 7 
    elif dataset == "prostate_t2w_adc":
        return 7 
    elif dataset == 'prostate_capras_t2w_adc': 
        return 7 
    else :
        print("Dataset not recognized.")
        sys.exit(1)

def load_data(data_path, dataset):
    data_dir_path = os.path.join(data_path, dataset)
    if dataset == 'lung_radiopathomic': 
        return load_lung_radiopathomic(data_dir_path)
    if dataset == 'lung_radiogenomic': 
        radiomic_df = pd.read_csv(os.path.join(data_dir_path, 'radiomic.csv'), index_col=0)
        genomic_df = pd.read_csv(os.path.join(data_dir_path, 'genomic.csv'), index_col=0)
        outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)
        time = outcome_df['time'].values
        status = outcome_df['status'].values
        radiomic_features = radiomic_df.values
        genomic_features = genomic_df.values
        return radiomic_features, genomic_features, time, status 
    if dataset == 'brain_radiogenomic': 
        radiomic_df = pd.read_csv(os.path.join(data_dir_path, 'radiomic.csv'), index_col=0)
        genomic_df = pd.read_csv(os.path.join(data_dir_path, 'genomic.csv'), index_col=0)
        outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)
        radiomic = radiomic_df.values 
        genomic = genomic_df.values 
        overall_time_to_progression = outcome_df['days'].values
        overall_progression_status = outcome_df['status'].values
        return radiomic, genomic, overall_time_to_progression, overall_progression_status
    data_dir_path = os.path.join(data_path, 'prostate_t2w_adc')
    t2w, adc, capras, time, status = load_all_prostate_data(data_dir_path)
    if dataset == 'prostate_capras_t2w_adc':
        fused_t2w_adc_features = np.concatenate([t2w, adc], axis=1)
        return capras, fused_t2w_adc_features, time, status 
    if dataset == 'prostate_t2w_adc':
        return t2w, adc, time, status 
    if dataset == 'prostate_capras_adc':
        return capras, adc, time, status 
    if dataset == 'prostate_capras_t2w': 
        return capras, t2w, time, status
    print("Dataset not recognized.")
    sys.exit(1)

def load_all_prostate_data(data_dir_path): 
    t2w_df = pd.read_csv(os.path.join(data_dir_path, 'T2_radiomic_features.csv'), index_col=0)
    adc_df = pd.read_csv(os.path.join(data_dir_path, 'ADC_radiomic_features.csv'), index_col=0)
    capras_df = pd.read_csv(os.path.join(data_dir_path, 'CAPRAS_data.csv'), index_col=0)
    clinical_df = pd.read_csv(os.path.join(data_dir_path, 'clinical_variables.csv'), index_col=0)
    outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)

    # Extract outcome data after checking for NaN and inf
    prostate_time = outcome_df['time'].values
    prostate_status = outcome_df['status'].values

    # Drop coded_mrn', 'Lesion_ID' columns if it exists
    t2w_features = t2w_df.drop(columns=['coded_mrn', 'Lesion_ID'], errors='ignore').values
    adc_features = adc_df.drop(columns=['coded_mrn', 'Lesion_ID'], errors='ignore').values
    capras_features = capras_df[['CAPRAS_Gleason', 'CAPRAS_PSA', 'CAPRAS_SM', 'CAPRAS_SVI', 'CAPRAS_ECE', 'CAPRAS_LNI']].values

    return t2w_features, adc_features, capras_features, prostate_time, prostate_status 

def load_lung_radiopathomic(data_dir_path):
    # Load data from CSV files
    radiomics_df = pd.read_csv(os.path.join(data_dir_path, 'all_radiomics.csv'), index_col=0)
    pathomics_df = pd.read_csv(os.path.join(data_dir_path, 'all_pathomics.csv'), index_col=0)
    outcome_df = pd.read_csv(os.path.join(data_dir_path, 'outcome.csv'), index_col=0)

    # Drop columns with all NA values
    pathomics_df = pathomics_df.dropna(axis=1, how='all')

    # Check for NaN and inf values in outcome_df before converting to numpy arrays
    if outcome_df['time'].isna().any() or \
        outcome_df['time'].isin([float('inf'), -float('inf')]).any():
        raise ValueError("NaN or inf found in overall time to progression data")

    if outcome_df['status'].isna().any() or \
      outcome_df['status'].isin([float('inf'), -float('inf')]).any():
        raise ValueError("NaN or inf found in overall progression status data")

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
    return radiomics, pathomics, overall_time_to_progression, overall_progression_status 