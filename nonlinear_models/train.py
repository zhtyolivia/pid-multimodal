import torch
import torch.nn as nn
import numpy as np 
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import os 
from util import regularize_weights

class MMDL(nn.Module):
    """
    Implements MMDL classifier.
    
    This implementation is adapted from: 
    https://github.com/pliang279/PID/blob/1f6e9d09598754f0dcf7d4ce7e7ffe1c377b0035/synthetic/supervised_learning.py
    """
    
    def __init__(self, encoders, fusion, head, has_padding=False):
        """Instantiate MMDL Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
        """
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs):
        """Apply MMDL to Layer Input.
        Args:
            inputs (torch.Tensor): Layer Input
        Returns:
            torch.Tensor: Layer Output
        """
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i](
                    [inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))
        self.reps = outs
        if self.has_padding:
            
            if isinstance(outs[0], torch.Tensor):
                out = self.fuse(outs)
            else:
                out = self.fuse([i[0] for i in outs])
        else:
            out = self.fuse(outs)
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            return self.head([out, inputs[1][0]])
        return self.head(out)

    
def test(model, dataloader, device):
    """
    Evaluates the given model on a dataloader by calculating the C-index. 

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The train/validation/test dataloader.
        device (torch.device): Device on which the model and data are located.

    Returns:
        c_index (float): C-index on this data
    """
    model.eval() 

    with torch.no_grad():
        pred = []
        true_censor = []
        true_time = []
        for j in dataloader:
            out = model([i.to(device) for i in j[:-2]])
            pred.append(out)
            true_time.append(j[-2])
            true_censor.append(j[-1])
    true_time = torch.cat(true_time, 0)
    true_censor = torch.cat(true_censor, 0)
    pred = torch.cat(pred, 0)
    c_index = concordance_index(true_time.cpu().detach().numpy(), -pred[:,0].cpu().detach().numpy(),  true_censor.cpu().detach().numpy())
    return c_index


def train(exp_folder, rep, fold, encoders, fusion, head, train_loader, val_loader, test_loader, total_epochs, device, is_packed=False, 
          early_stop=False, max_patience=8, optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0, lambda_reg=0.0,
          objective=nn.CrossEntropyLoss(), save_model=False, save_figs=False):
    """
    Trains a multimodal model using the provided encoders, fusion module, and head. The function performs training, 
    validation, and testing. It also saves the model and plots training/validation loss curves if specified.
    
    Args:
        exp_folder (str): Path to the folder for saving model and plots.
        rep (int): The current repetition for repeated CV. 
        fold (int): Fold index for CV with the current repetition.
        encoders, fusion, head: components of the multimodal model.
        train_loader, val_loader, test_loader (DataLoader): DataLoaders for training, validation, and testing.
        total_epochs (int): Total number of epochs for training.
        device (torch.device): Device to run the training on.
        is_packed: Flag indicating packed input data. 
        early_stop: Whether to enable early stopping based on validation loss. 
        max_patience: Maximum patience for early stopping.
        optimtype: Optimizer type. 
        lr: Learning rate. 
        weight_decay: Weight decay for the optimizer. 
        lambda_reg: Regularization strength for L1 penalty. 
        objective: Loss function. 
        save_model: Whether to save the trained model. 
        save_figs: Whether to save loss curves as figures. 
    
    Returns:
        train_c_index, val_c_index, test_c_index (float): Concordance index for training, validation, and testing datasets.
    """
    model = MMDL(encoders, fusion, head, has_padding=False).to(device)
    model_path = os.path.join(exp_folder, )

    # print(model)
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
            out = model([i.to(device) for i in j[:-2]])
            loss = objective.loss(j[-2], j[-1], out) + lambda_reg * regularize_weights(model) # (survtime, censor, hazard_pred) 
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
                out = model([i.to(device) for i in j[:-2]])
                loss = objective.loss(j[-2], j[-1], out) + lambda_reg * regularize_weights(model) # (survtime, censor, hazard_pred) 
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
            # print("Saving Best")
            if save_model:
                torch.save(model, f'{exp_folder}/loss_rep{rep}_fold{fold}.pth')
        else: 
            patience+=1
        if early_stop and patience > max_patience:
            break
    # ====== End of training loop ======
    if save_figs: 
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

    train_c_index = test(model, train_loader, device)
    val_c_index = test(model, val_loader, device)
    test_c_index = test(model, test_loader, device)

    return train_c_index, val_c_index, test_c_index