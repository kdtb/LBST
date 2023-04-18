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
        
        # Hyperparameters
#        self.save_hyperparameters(ignore=["model"])
        # Metrics
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
#        self.loss_fn = nn.BCELoss()
#        self.f1_score = torchmetrics.classification.BinaryF1Score()
#        self.val_f1_score = torchmetrics.classification.BinaryF1Score()
#        self.accuracy = torchmetrics.classification.BinaryAccuracy()
#        self.accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes=num_classes)
#        self.val_accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes=num_classes)
#        self.precision = torchmetrics.classification.BinaryPrecision()
#        self.recall = torchmetrics.classification.BinaryRecall()
#        self.confmat = torchmetrics.classification.ConfusionMatrix(task="binary", num_classes=num_classes)

        

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
        
        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        
        # Metrics
#        f1_score = self.f1_score(preds, y)
#        accuracy = self.accuracy(preds, y)
#        precision = self.precision(preds, y)
#        recall = self.recall(preds, y)
#        self.log_dict(
#            {
#                "train_loss": loss,
#                "train_f1_score": f1_score,
#                "train_accuracy": accuracy#,
#                "train_precision": precision,
#                "train_recall": recall
#            },
#            on_step=False,
#            on_epoch=True,
#            prog_bar=True,
#        )
        
        return {"loss": loss, "y": y, "preds": preds}

    def validation_step(self, batch, batch_idx):
        loss, y, preds = self._common_step(batch, batch_idx)
        
        
        self.valid_acc(preds, y)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
        
        # Metrics
#        val_f1_score = self.val_f1_score(preds, y)
#        val_accuracy = self.val_accuracy(preds, y)
#        self.log_dict(
#            {"val_loss": loss,
#             "val_f1_score": val_f1_score,
#             "val_accuracy": val_accuracy
#            },
#            on_step=False,
#            on_epoch=True,
#            prog_bar=True
#        )
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


# TO GET A BAG OF IMAGES SHOWN IN TENSORBOARD - EG. TO SEE HOW IMAGES ARE AUGMENTED    
#        if batch_idx % 100 == 0:
#            x = x[:8]
#            grid = torchvision.utils.make_grid(x.view(-1, 3, 224, 224))
#            self.logger.experiment.add_image("lbst_images", grid, self.global_step)    
    
    
# Old common step which reshaped.
#    def _common_step(self, batch, batch_idx):
#        x, y = batch
#        x = x.reshape(x.size(0), -1)
#        scores = self.forward(x)
#        loss = self.loss_fn(scores, y)
#        return loss, scores, y

# Original train loop ending:        return {"loss": loss, "y": y, "preds": preds, "scores": scores}