import pytorch_lightning as pl
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import torchmetrics



class NN(pl.LightningModule):
    def __init__(self, model, input_shape, learning_rate, num_classes):  # In the constructor, you declare all the layers you want to use.
        super().__init__()
        self.lr = learning_rate
        
        # The pretrained module
        self.model = model
        if hasattr(model, "dropout_proba"):
            self.dropout_proba = model.dropout_proba
        
        # layers are frozen by using eval()
        self.model.eval()
        # freeze params
        for param in self.model.parameters():
            param.requires_grad = False
        
        n_sizes = model.classifier[3].out_features # prior linear layers used as input_features to the last linear classifier layer
        model.classifier[6] = nn.Linear(n_sizes, num_classes)
        #n_sizes = model.classifier[6].out_features # returns the size of the output tensor going into the Linear layer from the conv block.
        #self.classifier = nn.Linear(n_sizes, num_classes)
        
        self.save_hyperparameters(ignore=['model'])
        # Metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.test_acc = torchmetrics.Accuracy(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.train_f1score = torchmetrics.F1Score(task='binary')
        self.val_f1score = torchmetrics.F1Score(task='binary')
        #self.train_confmat = torchmetrics.ConfusionMatrix(task='binary')
        #self.val_confmat = torchmetrics.ConfusionMatrix(task='binary')

    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        #x = self.classifier(x)
       
        return x

    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self(x)
        loss = self.loss_fn(scores, y)
        preds = torch.argmax(scores, dim=1)
        #print('PRINT PREDS', preds)
        #print('PRINT Y', y)

        return loss, y, preds
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, y, preds = self._common_step(batch, batch_idx)
        
        train_acc = self.train_acc(preds, y)
        train_recall = self.train_recall(preds, y)
        train_precision = self.train_precision(preds, y)
        train_f1score = self.train_f1score(preds, y)
        #train_confmat = self.train_confmat(preds.int(), y.int())
        
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": train_acc,
                "train_recall": train_recall,
                "train_precision": train_precision,
                "train_f1score": train_f1score,
                #"train_confmat": train_confmat
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        
        return {"loss": loss, "y": y, "preds": preds}
    


    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, y, preds = self._common_step(batch, batch_idx)
        
        val_acc = self.val_acc(preds, y)
        val_recall = self.val_recall(preds, y)
        val_precision = self.val_precision(preds, y)
        val_f1score = self.val_f1score(preds, y)
        #val_confmat = self.val_confmat(preds, y)
        
        self.log_dict(
            {
                "val_loss": loss,
                "val_accuracy": val_acc,
                "val_recall": val_recall,
                "val_precision": val_precision,
                "val_f1score": val_f1score,
                #"val_confmat": val_confmat
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        
        return loss


    def test_step(self, batch, batch_idx):
        loss, y, preds = self._common_step(batch, batch_idx)
        
        test_acc = self.test_acc(preds, y)
        
        self.log_dict(
            {"test_loss": loss,
             "test_acc": test_acc
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