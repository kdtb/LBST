import pytorch_lightning as pl
import torchmetrics
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

class NN(pl.LightningModule):
    def __init__(self, model, input_size, learning_rate, num_classes):  # In the constructor, you declare all the layers you want to use.
        super().__init__()
        self.lr = learning_rate
        
        # The inherited PyTorch module
        self.model = model
        if hasattr(model, "dropout_proba"):
            self.dropout_proba = model.dropout_proba

        # Metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes)
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes)
        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes)
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes)
        self.train_f1score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)


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
        train_recall = self.train_recall(preds, y)
        train_precision = self.train_precision(preds, y)
        train_f1score = self.train_f1score(preds, y)
        
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": train_acc,
                "train_recall": train_recall,
                "train_precision": train_precision,
                "train_f1score": train_f1score
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
        val_recall = self.val_recall(preds, y)
        val_precision = self.val_precision(preds, y)
        val_f1score = self.val_f1score(preds, y)
        
        self.log_dict(
            {
                "val_loss": loss,
                "val_accuracy": val_acc,
                "val_recall": val_recall,
                "val_precision": val_precision,
                "val_f1score": val_f1score
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