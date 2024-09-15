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
from get_data_unimodal import get_data, create_dataloaders
from objective import cox_loss
from common_models import * 
from util import regularize_weights
from lifelines.utils import concordance_index # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pdb 
from sklearn.decomposition import PCA


print(f'Using torch version {torch.__version__}')
device = torch.device('cuda')
print(f"Using device: {device}")

def train(exp_folder, rep, fold, pca, encoder, head, train_loader, val_loader, test_loader,
          total_epochs, early_stop, max_patience, optimtype, lr, weight_decay, weight_regu, objective):
    
    model = nn.Sequential(encoder, head)
    
    if pca is not None:
        # fit PCA on training data 
        train_features = torch.cat([j[0].cpu() for j in train_loader], dim=0).numpy()
        # import pdb; pdb.set_trace()
        pca.fit(train_features)

    best_val_loss = float('inf')
    patience = 0 
    op = optimtype([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(op, step_size=10, gamma=0.1)

    trainlosses = [] 
    vallosses = []

    for epoch in range(total_epochs):
        # train 
        model.train()
        totalloss = 0.0
        totals = 0
        for j in train_loader:
            inputs, survtime, censor = j[0].cpu().numpy(), j[1], j[2]
            if pca is not None: 
                inputs = pca.transform(inputs)
            inputs = torch.from_numpy(inputs).to(device)
            op.zero_grad()
            # out = model([i.to(device) for i in j[:-2]])
            out = model(inputs)
            loss = weight_regu * regularize_weights(model) + objective.loss(survtime, censor, out) # (survtime, censor, hazard_pred) 
            totalloss += loss * len(j[-1])
            totals += len(j[-1])
            loss.backward()
            op.step()
            scheduler.step()
        trainloss = (totalloss/totals).cpu().detach().numpy()
        # print("Epoch "+str(epoch)+" train loss: "+str(trainloss))
        trainlosses.append(trainloss)

        # val 
        model.eval()
        totalloss = 0.0
        totals = 0
        pred = []
        true_censor = []
        true_time = []
        with torch.no_grad():
            for j in val_loader:
                inputs, survtime, censor = j[0].cpu().numpy(), j[1], j[2]
                if pca is not None: 
                    inputs = pca.transform(inputs)
                inputs = torch.from_numpy(inputs).to(device)
                op.zero_grad()
                # out = model([i.to(device) for i in j[:-2]])
                out = model(inputs)
                loss = weight_regu * regularize_weights(model) + objective.loss(survtime, censor, out) # (survtime, censor, hazard_pred) 
                totalloss += loss*len(j[-1])
                totals += len(j[-1])
                pred.append(out)
                true_time.append(j[-2])
                true_censor.append(j[-1])
        pred = torch.cat(pred, 0)
        true_time = torch.cat(true_time, 0)
        true_censor = torch.cat(true_censor, 0)
        totals = true_time.shape[0]
        valloss = (totalloss/totals).cpu().detach().numpy()
        vallosses.append(valloss)
        epoch_val_c_index = concordance_index(true_time.cpu().detach().numpy(),  -pred[:,0].cpu().detach().numpy(), true_censor.cpu().detach().numpy())
        # if epoch_val_c_index < best_val_c_index: 
        if valloss < best_val_loss: 
            patience = 0
            # best_val_c_index = epoch_val_c_index
            best_val_loss = valloss
        else: 
            patience+=1
        if early_stop and patience > max_patience:
            break
    # ====== End of training loop ======
   
    # plot curves 
    plt.figure(figsize=(10, 5))
    plt.plot(trainlosses, label='Training Loss')
    plt.plot(vallosses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{exp_folder}/loss_rep{rep}_fold{fold}.png') 
    plt.close() 

    train_c_index = test(model, train_loader, pca=pca)
    val_c_index = test(model, val_loader, pca=pca)
    test_c_index = test(model, test_loader, pca=pca)

    return train_c_index, val_c_index, test_c_index

def test(model, dataloader, pca=None):
    """Calculates C-index on data in dataloader 
    """
    model.eval() 

    with torch.no_grad():
        pred = []
        true_censor = []
        true_time = []
        for j in dataloader:
            inputs = j[0].cpu().numpy()
            if pca is not None:
                inputs = pca.transform(inputs)
            inputs = torch.from_numpy(inputs).to(device)
            out = model(inputs)
            pred.append(out)
            true_time.append(j[-2])
            true_censor.append(j[-1])
    true_time = torch.cat(true_time, 0)
    true_censor = torch.cat(true_censor, 0)
    pred = torch.cat(pred, 0)
    c_index = concordance_index(true_time.cpu().detach().numpy(), -pred[:,0].cpu().detach().numpy(),  true_censor.cpu().detach().numpy())
    return c_index

# set seed for reproducibility 
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="../datasets/", type=str, help="Path to here all datasets are located")
parser.add_argument("--dataset", default="lung_radiopathomic", type=str, 
                    help="lung_radiopathomic, lung_radiogenomic, brain_radiogenomic, prostate_capras_t2w, prostate_capras_adc, prostate_t2w_adc")
parser.add_argument("--modality", default="radiomic", type=str, help="radiomic, pathomic, genomic, capras, t2w, adc")
parser.add_argument("--bs", default=12, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight-decay", default=1e-5, type=float)
parser.add_argument("--no-early-stop", action="store_false", dest="early_stop", help="Disable early stopping (default is enabled)")
parser.add_argument("--pca", action="store_true")
parser.add_argument("--pca-components", default=50, type=int)
parser.add_argument("--dropout-rate", default=0.1, type=float)
parser.add_argument("--no-dropout", action="store_false", dest="dropout", help="Disable dropout (default is enabled)")
parser.add_argument("--weight-regu", default=0, type=float)
parser.add_argument("--patience", default=7, type=int)
args = parser.parse_args()

exp = os.path.join('exp', datetime.now().strftime('%Y%m%d_%H%M%S'))
if not os.path.exists(exp):
    os.makedirs(exp)
    print(f"'{exp}' has been created.")

num_rep = 10
num_fold = 5 

# load data 
data, surv_status, surv_time, cv_splits = get_data(args.data_path, args.dataset, args.modality)

## Repeated CV 
rep_cv_c_indices_train = []
rep_cv_c_indices_val = [] 
rep_cv_c_indices_test = [] 
for rep in range(num_rep): 
    for fold in range(num_fold): 
        print(f"rep {rep+1} fold {fold+1}")
        train_loader, val_loader, test_loader = create_dataloaders(
            data, surv_time, surv_status, cv_splits, rep=rep, fold=fold, batch_size=args.bs, device=device)
        
        # default 
        if args.pca:
            pca = PCA(n_components=args.pca_components)
        else: 
            pca = None

        # lung radiopathomic - radiomic
        if args.dataset == 'lung_radiopathomic' and args.modality == 'radiomic': 
            encoder = MLP(data.shape[1], 128, 32, dropout=args.dropout, dropoutp=args.dropout_rate).to(device)
            head = MLP(32, 16, 1, dropout=args.dropout, dropoutp=args.dropout_rate).to(device) 
        elif args.dataset == 'lung_radiopathomic' and args.modality == 'pathomic':
            # lung radiopathomic - pathomic
            encoder = Linear(data.shape[1], 128).to(device)
            head = Linear(128, 1).to(device) 
        else: 
            encoder = Linear(data.shape[1], 128).to(device)
            head = Linear(128, 1).to(device)
        # encoder = Identity().to(device)
        # encoder = MLP(data.shape[1], 1024, 512, dropout=args.dropout, dropoutp=args.dropout_rate).to(device)

        # encoder = Identity().to(device)
        # head = MLP(data.shape[1], 4096, 1, dropout=args.dropout, dropoutp=args.dropout_rate).to(device) 

        
        train_c_index, val_c_index, test_c_index = train(
            exp_folder=exp, rep=rep, fold=fold, pca=pca, encoder=encoder, head=head, 
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            total_epochs=args.epochs, early_stop=args.early_stop, max_patience=args.patience, 
            optimtype=torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay, weight_regu=args.weight_regu,
            objective=cox_loss(device=device)
        )
        
        # Track metrics 
        rep_cv_c_indices_train.append(train_c_index)
        rep_cv_c_indices_val.append(val_c_index)
        rep_cv_c_indices_test.append(test_c_index)
        
print(f"Training C-index: {np.mean(rep_cv_c_indices_train):.4f} ({np.std(rep_cv_c_indices_train):.4f})")
# print(f"Validation C-index: {np.mean(rep_cv_c_indices_val):.4f} ({np.std(rep_cv_c_indices_val):.4f})")
print(f"Test C-index: {np.mean(rep_cv_c_indices_test):.4f} ({np.std(rep_cv_c_indices_test):.4f})")
