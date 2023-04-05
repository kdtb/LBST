import pytorch_lightning as pl
import torchmetrics
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import config
import VGGBlock

class NN(pl.LightningModule):  # pl.LightningModule inherits from nn.Module and adds extra functionality
    
    
    
    def __init__(self, input_size, learning_rate, num_classes):
        super().__init__()
        self.lr = learning_rate
        self.vgg_block = VGGBlock(3, 64) # input channels er 3 (RGB farve) og output channels er 64
        self.fc1 = nn.Linear(64*14*14, num_classes) # antallet af input features til fully connected-laget er output fra VGG-blokken
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        
    def forward(self, x):
        x = self.vgg_block(x)
        x = x.view(x.size(0), -1) # flattening
        x = self.fc1(x)
        return x
    
    
    
    
    
    
#    def __init__(self, input_size, learning_rate, num_classes):  # In the constructor, you declare all the layers you want to use.
#        super().__init__()
#        self.lr = learning_rate
#        self.fc1 = nn.Linear(input_size, 50)
#        self.fc2 = nn.Linear(50, num_classes)
#        self.loss_fn = nn.CrossEntropyLoss()
#        self.accuracy = torchmetrics.Accuracy(
#            task="multiclass", num_classes=num_classes
#        )
#        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
#
#    def forward(
#        self, x
#    ):  # Forward function computes output Tensors from input Tensors. In the forward function, you define how your model is going to be run, from input to output. We're accepting only a single input in here, but if you want, feel free to use more
#        x = F.relu(self.fc1(x))
#        x = self.fc2(x)
#        return x

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
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
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

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)  # flattening
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
