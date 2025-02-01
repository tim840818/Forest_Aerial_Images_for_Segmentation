import os, sys

import numpy as np
import cv2
from matplotlib import pyplot as plt

import torch
from torch import nn


def pixel_accuracy(predictions, ground_truth, tol=1e-5):
    diff = (predictions - ground_truth).abs()
    tot_pix = diff.numel() # avoid division by zero
    same_pix = (diff < tol).sum().item()
    return same_pix / tot_pix

def iou_accuracy(predictions, ground_truth, tol=1e-5):
    compare = (ground_truth > 0) | (predictions > 0) # we only compare the pixels that are nonzero (union of the two sets)
    tot_pixels = max(compare.sum().item(), 1) # avoid division by zero
    pred_compare = predictions[compare]
    truth_compare = ground_truth[compare]
    diff = (pred_compare - truth_compare).abs()
    correct_pixels = (diff < tol).sum().item() # count the same pixels (within a tolerance) (intersection of the two sets)
    return correct_pixels / tot_pixels

def train(model, device, train_loader, loss_func, optimizer):
    model.train()
    tot_loss = 0
    for _, (X, Y, tag) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = loss_func(Y_pred, Y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    return tot_loss / len(train_loader)
    # print(f'Epoch {epoch}, loss: {loss.item()}')

def evaluate(model, device, val_loader, loss_func, thresh=0.5, verbose=0, plot=False, mode='avg'):
    if mode == 'list':
        val_losses = []
        val_accuracies = []
        val_pix_accs = []
    plot = plot and verbose

    model.eval()
    with torch.no_grad():
        tot_val_loss = 0
        tot_accuracy = 0
        tot_pix_acc = 0
        num_images = len(val_loader)
        if plot:
            fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(6, 2 * num_images)) # figsize=(10, 2 * num_images)
        for i, (X, Y, tag) in enumerate(val_loader):
            X, Y = X.to(device), Y.to(device)
            Y_pred = model(X)
            val_loss = loss_func(Y_pred, Y).item()
            tot_val_loss += val_loss
            print(f'Validation for image-{tag}, loss: {val_loss}') if verbose else None
            # Predicted image
            Y_pred = (Y_pred > thresh).float()
            # print("min/max of Y_pred", Y_pred.min(), Y_pred.max())
            # accuracy = (Y_pred == Y).float().mean().item()
            accuracy = iou_accuracy(Y_pred, Y)
            tot_accuracy += accuracy
            pix_acc = pixel_accuracy(Y_pred, Y)
            tot_pix_acc += pix_acc
            print(f'IOU accuracy: {accuracy:.2f}, pixel accuracy: {pix_acc:.2f}') if verbose else None
            if mode == 'list':
                val_losses.append(val_loss)
                val_accuracies.append(accuracy)
                val_pix_accs.append(pix_acc)
            ## plot
            if plot:
                # Predicted image
                axes[i, 0].imshow(Y_pred[0].cpu().numpy().transpose(1, 2, 0), cmap='gray')
                axes[i, 0].set_title("Prediction")
                axes[i, 0].axis("off")
                # Ground truth image
                axes[i, 1].imshow(Y[0].cpu().numpy().transpose(1, 2, 0), cmap='gray')
                axes[i, 1].set_title("Ground Truth")
                axes[i, 1].axis("off")
                # Original image
                axes[i, 2].imshow(X[0].cpu().numpy().transpose(1, 2, 0))
                axes[i, 2].set_title("Original")
                axes[i, 2].axis("off")
        if plot:
            plt.tight_layout()
            plt.show()
        
        avg_val_loss = tot_val_loss / num_images
        avg_accuracy = tot_accuracy / num_images
        avg_pix_acc = tot_pix_acc / num_images
        print(f'Validation loss: {avg_val_loss:.2f}. Validation accuracy (IOU): {avg_accuracy:.2f}, (pixel): {avg_pix_acc:.2f}') if verbose else None
        
        if mode == 'avg':
            return avg_val_loss, avg_accuracy, avg_pix_acc
        elif mode == 'list':
            return val_losses, val_accuracies, val_pix_accs


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0., start_epoch=50, save_path="model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

        self.start_epoch = start_epoch
        self.epoch = 0
        self.save_opech = 0

        self.save_path = save_path

    def early_stop(self, validation_loss, model=None):
        self.epoch += 1 # update epoch
        if self.epoch < self.start_epoch: # Ensure augmentations will be trained for start_epoch epochs
            return False # do not stop training before start_epoch
        if (validation_loss - self.min_validation_loss) < -self.min_delta: # improvement by min_delta
            self.counter = 0
            if model is not None:
                self.save_epoch = self.epoch
                torch.save(model.state_dict(), self.save_path)
                # print(f"Model saved at epoch {self.epoch} with validation loss: {validation_loss:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience: # stop training
                return self.save_epoch-1 # return the epoch with the best model
        if validation_loss < self.min_validation_loss: # update min_validation_loss as long as validation_loss is smaller
            self.min_validation_loss = validation_loss
        return False


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        """
        Tversky Loss Constructor
        Args:
        - alpha: Weight for False Positives
        - beta: Weight for False Negatives
        - smooth: Small constant to avoid division by zero
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        Compute Tversky Loss
        Args:
        - y_pred: Predicted probabilities (logits or after sigmoid)
        - y_true: Ground truth binary masks
        Returns:
        - loss: Scalar Tversky loss
        """
        
        # Flatten tensors for computation
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # Calculate True Positives, False Positives, and False Negatives
        TP = (y_pred * y_true).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()
        
        # Compute Tversky Index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # Compute Tversky Loss
        loss = 1 - tversky_index
        return loss
    
class CombinedBCETverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, lambda_bce=0.5, smooth=1e-6):
        """
        Combined BCE and Tversky Loss Constructor
        Args:
        - alpha: Weight for False Positives in Tversky Loss
        - beta: Weight for False Negatives in Tversky Loss
        - lambda_bce: Weighting factor for BCE in the combined loss
        - smooth: Small constant to avoid division by zero
        """
        super().__init__()
        self.lambda_bce = lambda_bce
        self.bce = nn.BCELoss()
        self.tversky = TverskyLoss(alpha, beta, smooth)

    def forward(self, y_pred, y_true):
        # Compute BCE Loss
        bce_loss = self.bce(y_pred, y_true)

        # Compute Tversky Loss
        tversky_loss = self.tversky(y_pred, y_true)

        # Combine the two losses
        combined_loss = self.lambda_bce * bce_loss + (1 - self.lambda_bce) * tversky_loss
        return combined_loss
