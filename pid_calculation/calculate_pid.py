import sys
import os
import argparse 
cwd = os.getcwd()
print("Current working directory:", cwd)
sys.path.append(os.getcwd())
sys.path.append('../PID')
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from synthetic.rus import *
from pid_utils import * 
from sklearn.utils import resample
from tqdm import tqdm
import random 
# set seed for reproducibility 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="../datasets/", type=str, help="Path to here all datasets are located")
parser.add_argument("--dataset", default="lung_radiopathomic", type=str, 
                    help="Name of the dataset")
parser.add_argument("--num-bins", default=5, type=int, help="Number of bins for histogram. Should set to cube root of the number of samples")
parser.add_argument("--pca-components", default=3, type=int, help="Number of PCA components")
parser.add_argument("--customize-pca", action='store_true') 
parser.add_argument("--k", default=10, type=int, help="Number of clusters")
parser.add_argument("--bootstrap", action='store_true', help="Perform bootstrapping to get 95% CI for each metric")
parser.add_argument("--n-bootstrap", default=1000, type=int, help="Number of bootstrap samples")

args = parser.parse_args()

# Define PID hyperparameters 
number_of_bins = get_num_bins(args.dataset)
n_clusters = args.k
x1_pca, x2_pca = parse_pca(args.customize_pca, args.pca_components, args.dataset)
   
# Load dataset 
x1, x2, time, status = load_data(args.data_path, args.dataset)


# Convert time to bins using historgramming 
time_bins = histogram_map_to_bin(number_of_bins, time)

# Dataset clustering 
data_cluster = dict()
data_cluster = dict()
dataset = {'0': x1, '1': x2, 'label': time_bins}

kmeans_0, data_0 = clustering(dataset['0'], pca=True, n_components=x1_pca, n_clusters=n_clusters, random_state=SEED)
data_cluster['0'] = kmeans_0.reshape(-1,1)
kmeans_1, data_1 = clustering(dataset['1'], pca=True, n_components=x2_pca, n_clusters=n_clusters, random_state=SEED)
data_cluster['1'] = kmeans_1.reshape(-1,1)
data_cluster['label'] = dataset['label']

## PID measures 
data = (data_cluster['0'], data_cluster['1'], data_cluster['label'])
P, maps = convert_data_to_distribution(*data)
result = get_measure(P)
print(result)


if args.bootstrap: 
    print("Calculating bootstrapped 95% CIs")
    bootstrap_results = []
    n_samples = len(x1)  
    metrics = ['redundancy', 'unique1', 'unique2', 'synergy']
    for i in tqdm(range(args.n_bootstrap), desc="Bootstrapping", unit="iteration"):
        # resample x1, x2, time, and status with replacement
        resampled_idx = resample(np.arange(n_samples), replace=True, random_state=SEED + i)
        resampled_x1 = x1[resampled_idx]
        resampled_x2 = x2[resampled_idx]
        resampled_time = time[resampled_idx]
        resampled_status = status[resampled_idx]  # If needed

        # convert resampled time to bins using histogramming
        resampled_time_bins = histogram_map_to_bin(number_of_bins, resampled_time)

        # clustering
        resampled_data_cluster = dict()
        resampled_dataset = {'0': resampled_x1, '1': resampled_x2, 'label': resampled_time_bins}

        kmeans_0, data_0 = clustering(resampled_dataset['0'], pca=True, n_components=x1_pca, n_clusters=n_clusters, random_state=SEED)
        resampled_data_cluster['0'] = kmeans_0.reshape(-1, 1)
        kmeans_1, data_1 = clustering(resampled_dataset['1'], pca=True, n_components=x2_pca, n_clusters=n_clusters, random_state=SEED)
        resampled_data_cluster['1'] = kmeans_1.reshape(-1, 1)
        resampled_data_cluster['label'] = resampled_dataset['label']

        # Calculate PID measures for resampled data
        resampled_data = (resampled_data_cluster['0'], resampled_data_cluster['1'], resampled_data_cluster['label'])
        P, maps = convert_data_to_distribution(*resampled_data)
        bootstrap_result = get_measure(P)
        bootstrap_results.append(bootstrap_result)

    
    bootstrap_results = {metric: [r[metric] for r in bootstrap_results] for metric in metrics}

    # Calculate 95% confidence intervals
    ci = {}
    for metric in metrics:
        lower_bound = np.percentile(bootstrap_results[metric], 2.5)
        upper_bound = np.percentile(bootstrap_results[metric], 97.5)
        ci[metric] = (lower_bound, upper_bound)

    print("Bootstrapped 95% Confidence Intervals:")
    for metric, (low, high) in ci.items():
        print(f"{metric}: {result[metric]:.4f} ({low:.4f}, {high:.4f})")