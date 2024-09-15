import torch

def regularize_weights(model):
    # This was modified from https://github.com/mahmoodlab/PathomicFusion
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg

# def regularize_weights(model, alpha=0.5, l1_ratio=0.5):
#     """
#     Apply Elastic Net regularization to a PyTorch model.

#     Args:
#     model (torch.nn.Module): The model to regularize.
#     alpha (float): Overall regularization strength.
#     l1_ratio (float): The ratio of L1 to L2 regularization in the Elastic Net formulation.

#     Returns:
#     torch.Tensor: The Elastic Net regularization term.
#     """
#     l1_reg = torch.tensor(0., requires_grad=True)
#     l2_reg = torch.tensor(0., requires_grad=True)

#     for W in model.parameters():
#         l1_reg = l1_reg + W.abs().sum()
#         l2_reg = l2_reg + (W ** 2).sum()

#     # Calculate the Elastic Net regularization term
#     elastic_net_reg = alpha * (l1_ratio * l1_reg + (1 - l1_ratio) * l2_reg)
#     return elastic_net_reg