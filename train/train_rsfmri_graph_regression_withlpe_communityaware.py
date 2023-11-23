"""
    Utility function for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import numpy as np
import scipy.stats as stats


#from train.metrics import MAE, CORR, MSE
import torch

def MSE(y_pred, y_true):
    """
    Compute mean squared error between predicted and true values.
    
    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.
        
    Returns:
        torch.Tensor: Mean squared error.
    """
    return torch.mean((y_pred - y_true) ** 2).item()

def MAE(y_pred, y_true):
    """
    Compute mean absolute error between predicted and true values.
    
    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.
        
    Returns:
        torch.Tensor: Mean absolute error.
    """
    return torch.mean(torch.abs(y_pred - y_true)).item()

def CORR(y_pred, y_true):
    """
    Compute absolute correlation coefficient between predicted and true values.
    
    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.
        
    Returns:
        torch.Tensor: Absolute correlation coefficient.
    """
    mean_pred = torch.mean(y_pred)
    mean_true = torch.mean(y_true)
    covar = torch.sum((y_pred - mean_pred) * (y_true - mean_true))
    std_pred = torch.sqrt(torch.sum((y_pred - mean_pred) ** 2))
    std_true = torch.sqrt(torch.sum((y_true - mean_true) ** 2))
    if std_true==0:
        return std_true.item()
    else:
        corr_coefficient = covar / (std_pred * std_true)
        return abs(corr_coefficient).item()

def topk_loss(s,ratio):
    EPS = 1e-10
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
    return res


def train_epoch(model, optimizer, device, data_loader, epoch, LPE):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    epoch_train_mse = 0
    epoch_corr = 0
    predictions =[]
    targets=[]

    # # Assuming 'model' is your PyTorch model
    # for name, param in model.named_parameters():
    #     print(f"Parameter '{name}' has data type: {param.dtype}")

    
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):

        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)

        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()  

        #Encoding for nodes for BrainGNN
        pseudo = np.zeros((len(batch_targets), 100, 100))
        # set the diagonal elements to 1
        pseudo[:, np.arange(100), np.arange(100)] = 1
        pseudo_arr = np.concatenate(pseudo, axis=0)
        pseudo_torch = torch.from_numpy(pseudo_arr).float()

        if LPE == 'community' or LPE == "communitytopk":
            batch_EigVecs = batch_graphs.ndata['EigVecs'].to(device)
            #random sign flipping
            sign_flip = torch.rand(batch_EigVecs.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            
            batch_EigVals = batch_graphs.ndata['EigVals'].to(device)
            batch_scores, topk_score = model.forward(batch_graphs, batch_x, batch_e, batch_EigVecs, batch_EigVals, pseudo_torch)
        
        elif LPE == 'node':
            batch_EigVecs = batch_graphs.ndata['EigVecs'].to(device)
            #random sign flipping
            sign_flip = torch.rand(batch_EigVecs.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            
            batch_EigVals = batch_graphs.ndata['EigVals'].to(device)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_EigVecs, batch_EigVals)

        elif LPE == 'edge':
            batch_diff = batch_graphs.edata['diff'].to(device)
            batch_prod = batch_graphs.edata['product'].to(device)
            batch_EigVals = batch_graphs.edata['EigVals'].to(device)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_diff, batch_prod, batch_EigVals)
            
        else:

            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            std_output = torch.std(batch_scores)
            #print("Standard Deviation here is: " + str(std_output))
        original_targets = batch_targets
        original_predictions = batch_scores
        

        loss_mse = model.loss(batch_scores, batch_targets)
        loss = loss_mse #+ 0.1 * topk_scores

        
        loss.backward()


        predictions.append(original_targets.detach())
        targets.append(original_predictions.detach())

                    

        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(original_predictions.detach(), original_targets.detach())
        epoch_train_mse += MSE(original_predictions.detach(), original_targets.detach())
        epoch_corr += CORR(original_predictions.detach(), original_targets.detach())

    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    epoch_train_mse /= (iter + 1)
    epoch_corr /= (iter + 1)
    
    return epoch_train_mse, epoch_corr, epoch_loss, epoch_train_mae, optimizer

def evaluate_network(model, device, data_loader, epoch, LPE):
    model.eval()
    max_value = 193.5191
    min_value = 61.0076
    epoch_test_loss = 0
    epoch_test_mae = 0
    epoch_test_mse=0
    epoch_test_corr = 0

    epoch_std = 0

    predictions =[]
    targets=[]
    topk_scores=[]

    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)

            #Encoding for nodes for BrainGNN
            pseudo = np.zeros((len(batch_targets), 100, 100))
            # set the diagonal elements to 1
            pseudo[:, np.arange(100), np.arange(100)] = 1
            pseudo_arr = np.concatenate(pseudo, axis=0)
            pseudo_torch = torch.from_numpy(pseudo_arr).float()

            if LPE == 'community' or LPE =='communitytopk':
                batch_EigVecs = batch_graphs.ndata['EigVecs'].to(device)
                #random sign flipping
                sign_flip = torch.rand(batch_EigVecs.size(1)).to(device)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                
                batch_EigVals = batch_graphs.ndata['EigVals'].to(device)
                batch_scores, topk_score = model.forward(batch_graphs, batch_x, batch_e, batch_EigVecs, batch_EigVals, pseudo_torch)
            
            elif LPE == 'node':
                batch_EigVecs = batch_graphs.ndata['EigVecs'].to(device)
                #random sign flipping
                sign_flip = torch.rand(batch_EigVecs.size(1)).to(device)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                
                batch_EigVals = batch_graphs.ndata['EigVals'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_EigVecs, batch_EigVals)

            elif LPE == 'edge':
                batch_diff = batch_graphs.edata['diff'].to(device)
                batch_prod = batch_graphs.edata['product'].to(device)
                batch_EigVals = batch_graphs.edata['EigVals'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_diff, batch_prod, batch_EigVals)
                
            else:

                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
                std_output = torch.std(batch_scores)

            original_targets = batch_targets
            original_predictions = batch_scores

            loss_mse = model.loss(batch_scores, batch_targets)
            loss = loss_mse #+ 0.1 * topk_scores
            epoch_test_loss += loss

            predictions.append(original_predictions.detach())
            targets.append(original_targets.detach())
                

            epoch_test_mae += MAE(original_predictions.detach(), original_targets.detach())
            epoch_test_mse += MSE(original_predictions.detach(), original_targets.detach())
            epoch_test_corr += CORR(original_predictions.detach(), original_targets.detach())
            epoch_std += torch.std(original_predictions.detach()).item()


        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        epoch_test_mse /= (iter + 1)
        epoch_test_corr /= (iter + 1)
        epoch_std /= (iter + 1)

        print("Test MSE is: " + str(epoch_test_mse))

        print("Predicted TEST Scores:", np.hstack(predictions[0][:10].detach().cpu().numpy()))
        
    return epoch_test_mse, epoch_test_corr, epoch_test_loss, epoch_test_mae, epoch_std

