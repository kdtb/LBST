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
        
        #n_sizes = model.classifier[3].out_features # prior linear layers used as input_features to the last linear classifier layer
        model.fc = nn.Linear(512, num_classes)
        #n_sizes = model.classifier[6].out_features # returns the size of the output tensor going into the Linear layer from the conv block.
        #self.classifier = nn.Linear(n_sizes, num_classes)
        
        self.save_hyperparameters(ignore=['model'])
        # Metrics
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.FloatTensor([0.5351, 3.8073]))
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.test_acc = torchmetrics.Accuracy(task='binary')

    
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
            prog_bar=True
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
        
        test_acc = self.test_acc(preds, y)
        
        self.log_dict(
            {
                "test_loss": loss,
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