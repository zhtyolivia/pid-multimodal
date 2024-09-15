import torch 

def regularize_weights(model):
    """
    Computes the L1 regularization term for the weights of a model. 
    This function calculates the sum of the absolute values of the model's parameters (L1 norm) 
    to serve as a regularization term, which helps to prevent overfitting by penalizing large weights.
    
    Args:
        model (torch.nn.Module): The neural network model whose weights will be regularized.
    
    Returns:
        l1_reg (torch.Tensor): The L1 regularization term, calculated as the sum of absolute values 
                               of the model's parameters.
    
    Note:
        This implementation was adapted from https://github.com/mahmoodlab/PathomicFusion
    """
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg

class cox_loss(): 
    """
    Custom loss function for Cox proportional hazards, used for survival analysis. 
    This loss function calculates the Cox partial log-likelihood to predict hazard ratios.
    
    Args:
        device (torch.device): Device on which the tensors will be processed (e.g., 'cuda' or 'cpu').
    
    Methods:
        loss(survtime, censor, hazard_pred):
            Computes the Cox proportional hazards loss based on the survival times, censor status, and predicted hazards.
    
    Args (for loss method):
        survtime (torch.Tensor): A tensor containing the survival times 
        censor (torch.Tensor): A tensor of binary values, 1 for an event occurred and 0 for censored data.
        hazard_pred (torch.Tensor): A tensor containing predicted hazard scores 
    
    Returns:
        loss_cox (torch.Tensor): The computed Cox proportional hazards loss.
    
    Note:
        This implementation was adapted from https://github.com/mahmoodlab/PathomicFusion
    """
    def __init__(self, device): 
        self.device = device 

    def loss(self, survtime, censor, hazard_pred): 
        # This was modified from https://github.com/mahmoodlab/PathomicFusion/blob/master/utils.py#L346
        current_batch_len = len(survtime)
        R_mat = torch.zeros([current_batch_len, current_batch_len], dtype=torch.float32)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = float(survtime[j] >= survtime[i])

        R_mat = torch.FloatTensor(R_mat).to(self.device)
        theta = hazard_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
        return loss_cox
