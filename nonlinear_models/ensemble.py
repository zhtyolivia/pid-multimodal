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

print(f'Using torch version {torch.__version__}')
device = torch.device('cuda')
print(f"Using device: {device}")

def train(encoder, head, train_loader, val_loader, test_loader,
          total_epochs, early_stop, max_patience, optimtype, lr, weight_decay, weight_regu, objective):
    model = nn.Sequential(encoder, head)
    
    best_val_loss = 10000
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
            op.zero_grad()
            out = model(j[0].to(device))
            loss = weight_regu * regularize_weights(model) + objective.loss(j[-2], j[-1], out) # (survtime, censor, hazard_pred) 
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
                out = model(j[0].to(device))
                loss = weight_regu * regularize_weights(model) + objective.loss(j[-2], j[-1], out) # (survtime, censor, hazard_pred) 
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
    return model 

def ensemble(model1, model2, dataloader1, dataloader2): 
    model1.eval() 
    model2.eval() 
    
    with torch.no_grad():
        pred = []
        true_censor = []
        true_time = []
        # for j in dataloader:
        for i, (data1, data2) in enumerate(zip(dataloader1, dataloader2)):
            # import pdb; pdb.set_trace() 
            out1 = model1(data1[0].to(device))
            out2 = model2(data2[0].to(device))
            out_sum = out1+out2
            pred.append(out_sum)
            true_time.append(data1[-2])
            true_censor.append(data2[-1])
    true_time = torch.cat(true_time, 0)
    true_censor = torch.cat(true_censor, 0)
    pred = torch.cat(pred, 0)
    c_index = concordance_index(true_time.cpu().detach().numpy(), -pred[:,0].cpu().detach().numpy(),  true_censor.cpu().detach().numpy())
    return c_index

# set seed for reproducibility 
torch.manual_seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="../datasets/", type=str, help="Path to here all datasets are located")
parser.add_argument("--dataset", default="lung_radiopathomic", type=str, 
                    help="lung_radiopathomic, lung_radiogenomic, brain_radiogenomic, prostate_capras_t2w, prostate_capras_adc, prostate_t2w_adc")
parser.add_argument("--bs", default=30, type=int)

args = parser.parse_args()

if args.dataset == 'lung_radiopathomic':
    mod1 = 'radiomic'
    mod2 = 'pathomic'
elif args.dataset == 'lung_radiogenomic':
    mod1 = 'radiomic'
    mod2 = 'genomic'
elif args.dataset =='prostate_t2w_adc':
    mod1 = 't2w'
    mod2 = 'adc'

# Lung_radiopathomic (radiomic): --bs 12 --epochs 30 --lr 1e-3 --weight-decay 1e-5 --early-stop --patience 6 --dropout --dropout-rate 0.1
# Lung_radiogenomic (radiomic): --bs 12 --epochs 30 --lr 1e-3 --weight-decay 1e-5 --early-stop --patience 6
# Prostate_t2w_adc (t2w): --bs 30 --epochs 30 --lr 1e-2 --weight-decay 1e-5 --weight-regu 1e-3 --early-stop --patience 6 --dropout --dropout-rate 0.1
# lung_radiopathomic -- radiomic-only model 
mod1_epochs = 30
mod1_lr = 1e-4
mod1_weight_decay = 1e-5 
mod1_early_stop = True 
mod1_patience = 7
mod1_weight_regu = 0
mod1_dropout = True 
mod1_dropout_rate = 0.1

# Lung_radiopathomic (pathomic): -- bs 12 --epochs 30 --lr 1e-3 --weight-decay 1e-5 --early-stop --patience 6
# Lung_radiogenomic (genomic): --bs 12 --epochs 30 --lr 1e-3 --weight-decay 1e-5 --early-stop --patience 6 --dropout --dropout-rate 0.1
# Prostate_t2w_adc (adc):  --bs 24 --epochs 40 --lr 5e-3 --weight-decay 1e-5 --weight-regu 1e-5 --early-stop --patience 8 --dropout --dropout-rate 0.2
# lung_radiopathomic -- pathomic-only model 
mod2_epochs = 30
mod2_lr = 1e-4
mod2_early_stop = True 
mod2_patience = 7
mod2_weight_decay = 1e-5
mod2_weight_regu = 0
mod2_dropout = False
mod2_dropout_rate = 0

num_rep = 10
num_fold = 5 

# load data 
x1, surv_status, surv_time, cv_splits = get_data(args.data_path, args.dataset, mod1)
x2, _, _, _ = get_data(args.data_path, args.dataset, mod2)

## Repeated CV 
rep_cv_c_indices_train = []
rep_cv_c_indices_val = [] 
rep_cv_c_indices_test = [] 
for rep in range(num_rep): 
    for fold in range(num_fold): 
        print(f"rep {rep+1} fold {fold+1}")
        x1_train_loader, x1_val_loader, x1_test_loader = create_dataloaders(
            x1, surv_time, surv_status, cv_splits, rep=rep, fold=fold, batch_size=args.bs, device=device)
        x2_train_loader, x2_val_loader, x2_test_loader = create_dataloaders(
            x2, surv_time, surv_status, cv_splits, rep=rep, fold=fold, batch_size=args.bs, device=device)
        encoder1 = MLP(x1.shape[1], 128, 32, dropout=mod1_dropout, dropoutp=mod1_dropout_rate).to(device)
        head1 = MLP(32, 16, 1, dropout=mod1_dropout, dropoutp=mod1_dropout_rate).to(device) 

        model1 = train(
            encoder=encoder1, head=head1, 
            train_loader=x1_train_loader, val_loader=x1_val_loader, test_loader=x1_test_loader,
            total_epochs=mod1_epochs, early_stop=mod1_early_stop, max_patience=mod1_patience, 
            optimtype=torch.optim.AdamW, lr=mod1_lr, weight_decay=mod1_weight_decay, weight_regu=mod1_weight_regu,
            objective=cox_loss(device=device)
        )
        
        # lung_radopathomic 
        encoder2 = Identity().to(device)
        head2 = MLP(x2.shape[1], 128, 1, dropout=mod2_dropout, dropoutp=mod2_dropout_rate).to(device)
        # encoder2 = Linear(x2.shape[1], 128).to(device)
        # head2 = Linear(128, 1).to(device) 
        
        model2 = train(
            encoder=encoder2, head=head2, 
            train_loader=x2_train_loader, val_loader=x2_val_loader, test_loader=x2_test_loader,
            total_epochs=mod2_epochs, early_stop=mod2_early_stop, max_patience=mod2_patience, 
            optimtype=torch.optim.AdamW, lr=mod2_lr, weight_decay=mod2_weight_decay, weight_regu=mod2_weight_regu,
            objective=cox_loss(device=device)
        )

        train_c_index = ensemble(model1, model2, x1_train_loader, x2_train_loader)
        val_c_index = ensemble(model1, model2, x1_val_loader, x2_val_loader)
        test_c_index = ensemble(model1, model2, x1_test_loader, x2_test_loader)

        rep_cv_c_indices_train.append(train_c_index)
        rep_cv_c_indices_val.append(val_c_index)
        rep_cv_c_indices_test.append(test_c_index)
        
print(f"Training C-index: {np.mean(rep_cv_c_indices_train):.4f} ({np.std(rep_cv_c_indices_train):.4f})")
print(f"Validation C-index: {np.mean(rep_cv_c_indices_val):.4f} ({np.std(rep_cv_c_indices_val):.4f})")
print(f"Test C-index: {np.mean(rep_cv_c_indices_test):.4f} ({np.std(rep_cv_c_indices_test):.4f})")
