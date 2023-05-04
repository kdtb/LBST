import pytorch_lightning as pl
import torchmetrics
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorchModel import pytorchModel
import torch

class NN(pl.LightningModule):
    def __init__(self, model, input_size, learning_rate, num_classes):  # In the constructor, you declare all the layers you want to use.
        super().__init__()
        self.lr = learning_rate
        
        # The inherited PyTorch module
        self.model = model
        if hasattr(model, "dropout_proba"):
            self.dropout_proba = model.dropout_proba

        self.save_hyperparameters(ignore=['model'])
        # Metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")


    def forward(self, x):  # Forward function computes output Tensors from input Tensors.
        return self.model(x)
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self(x)
        loss = self.loss_fn(scores, y)
        preds = torch.argmax(scores, dim=1)

        return loss, y, preds
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, y, preds = self._common_step(batch, batch_idx)
        
        train_acc = self.train_acc(preds, y)
         
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": train_acc
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        return {"loss": loss, "y": y, "preds": preds}


    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, y, preds = self._common_step(batch, batch_idx)
        
        val_acc = self.val_acc(preds, y)
        
        self.log_dict(
            {
                "val_loss": loss,
                "val_accuracy": val_acc
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        
        return loss


    def test_step(self, batch, batch_idx):
        loss, y, preds = self._common_step(batch, batch_idx)
        self.log_dict(
            {"test_loss": loss
            },
            on_step=False,
            on_epoch=True
        )
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)  # flattening
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)