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
#        if hasattr(model, "dropout_proba"):
#            self.dropout_proba = model.dropout_proba

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):  # Forward function computes output Tensors from input Tensors.
        return self.model(x)

# Old common step which resized.
#    def _common_step(self, batch, batch_idx):
#        x, y = batch
#        x = x.reshape(x.size(0), -1)
#        scores = self.forward(x)
#        loss = self.loss_fn(scores, y)
#        return loss, scores, y
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self(x)
        loss = self.loss_fn(scores, y)
        preds = torch.argmax(scores, dim=1)

        return loss, scores, y#, preds
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        if batch_idx % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 3, 224, 224))
            self.logger.experiment.add_image("lbst_images", grid, self.global_step)
        
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)  # flattening
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
