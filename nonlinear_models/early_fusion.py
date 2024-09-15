import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import logging
import gzip
from datetime import datetime
import os, sys
sys.path.append(os.getcwd())
from get_data import get_data, create_dataloaders
from objective import cox_loss
from common_models import * 
from fusions import * 
from train import MMDL, test, train 
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import pdb 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# set seed for reproducibility 
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="../datasets/", type=str, help="Path to here all datasets are located")
parser.add_argument("--dataset", default="lung_radiopathomic", type=str, help="Name the target dataset")
parser.add_argument("--bs", default=32, type=int)
parser.add_argument("--hidden-dim", default=512, type=int)
parser.add_argument("--out-dim", default=512, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--lr", default=1.5e-4, type=float)
parser.add_argument("--weight-decay", default=1e-5, type=float)
parser.add_argument("--weight-regu", default=1e-3, type=float)
parser.add_argument("--no-early-stop", action="store_false", dest="early_stop", help="Disable early stopping (default is enabled)")
parser.add_argument("--no-dropout", action="store_false", dest="dropout", help="Disable dropout (default is enabled)")
parser.add_argument("--dropout-rate", default=0.1, type=float)
parser.add_argument("--patience", default=6, type=int)
parser.add_argument("--save-figs", action='store_true') # default is False 
args = parser.parse_args()

exp = os.path.join('exp', datetime.now().strftime('%Y%m%d_%H%M%S'))
if args.save_figs and not os.path.exists(exp):
    os.makedirs(exp)
    print(f"'{exp}' has been created.")

num_rep = 10
num_fold = 5 

# load data 
x1, x2, surv_status, surv_time, cv_splits = get_data(args.data_path, args.dataset)

## Repeated CV 
rep_cv_c_indices_train = []
rep_cv_c_indices_val = [] 
rep_cv_c_indices_test = [] 
for rep in range(num_rep): 
    for fold in range(num_fold): 
        print(f"rep {rep+1} - fold {fold+1}")
        train_loader, val_loader, test_loader = create_dataloaders(
            x1, x2, surv_time, surv_status, cv_splits, rep=rep, fold=fold, batch_size=args.bs, device=device)

        # Specify the encoder, fusion, and head 
        input_dims = [x1.shape[1], x2.shape[1]]
        encoders = [Linear(indim=input_dim, outdim=args.out_dim) for input_dim in input_dims]
        fusion = Concat().to(device)
        head = MLP(args.out_dim*2, hiddim=args.hidden_dim, outdim=1, dropout=args.dropout, dropoutp=args.dropout_rate).to(device) 

        train_c_index, val_c_index, test_c_index = train(
            exp_folder=exp, rep=rep, fold=fold, encoders=encoders, fusion=fusion, head=head, 
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            total_epochs=args.epochs, device=device, is_packed=False, early_stop=args.early_stop, max_patience=args.patience, 
            optimtype=torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay, lambda_reg=args.weight_regu,
            objective=cox_loss(device=device), save_model=False, save_figs=args.save_figs
        )
        
        # Track metrics 
        rep_cv_c_indices_train.append(train_c_index)
        rep_cv_c_indices_val.append(val_c_index)
        rep_cv_c_indices_test.append(test_c_index)
        
print(f"Training C-index: {np.mean(rep_cv_c_indices_train):.4f} ({np.std(rep_cv_c_indices_train):.4f})")
# print(f"Validation C-index: {np.mean(rep_cv_c_indices_val):.4f} ({np.std(rep_cv_c_indices_val):.4f})")
print(f"Test C-index: {np.mean(rep_cv_c_indices_test):.4f} ({np.std(rep_cv_c_indices_test):.4f})")
