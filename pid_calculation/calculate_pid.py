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

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="../datasets/", type=str, help="Path to here all datasets are located")
parser.add_argument("--dataset", default="lung_radiopathomic", type=str, 
                    help="Name of the dataset")
parser.add_argument("--num-bins", default=5, type=int, help="Number of bins for histogram. Should set to cube root of the number of samples")
parser.add_argument("--pca-components", default=3, type=int, help="Number of PCA components")
parser.add_argument("--customize-pca", action='store_true') 
parser.add_argument("--k", default=10, type=int, help="Number of clusters")
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

kmeans_0, data_0 = clustering(dataset['0'], pca=True, n_components=x1_pca, n_clusters=n_clusters)
data_cluster['0'] = kmeans_0.reshape(-1,1)
kmeans_1, data_1 = clustering(dataset['1'], pca=True, n_components=x2_pca, n_clusters=n_clusters)
data_cluster['1'] = kmeans_1.reshape(-1,1)
data_cluster['label'] = dataset['label']

## PID measures 
data = (data_cluster['0'], data_cluster['1'], data_cluster['label'])
P, maps = convert_data_to_distribution(*data)
result = get_measure(P)
print(result)