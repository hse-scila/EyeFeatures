import pytorch_lightning as pl
import torch
import torchmetrics
from sklearn.model_selection import train_test_split
from torch import nn

from eyetracking.features.complex import get_heatmaps


class SimpleCNNclassifier(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        shape,
        cnn_laeyrs_chanels=(4, 8, 4),
        kernel_size=3,
        stride=1,
        padding=0,
        classifier_layers=2,
        learning_rate=1e-3,
    ):

        super().__init__()

        modules = list()
        in_chanels = 1
        padding = padding

        for i in cnn_laeyrs_chanels:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_chanels,
                        i,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(i),
                    nn.ReLU(),
                )
            )
            in_chanels = i
            padding = 0
            shape = (
                (shape[0] - kernel_size + padding + stride) // stride,
                (shape[1] - kernel_size + padding + stride) // stride,
            )

        self.CNN = nn.ModuleList(modules)

        self.flat = nn.Flatten()

        modules = list()
        in_chanels = 1
        padding = padding

        in_features = shape[0] * shape[1] * i

        for i in range(classifier_layers):
            if i != classifier_layers - 1:
                modules.append(nn.Linear(in_features, in_features // 2))
                modules.append(nn.ReLU())
            else:
                modules.append(nn.Linear(in_features, n_classes))
            in_features = in_features // 2

        self.classifier = nn.ModuleList(modules)

        self.accuracy = torchmetrics.Accuracy(
            task="binary" if n_classes == 2 else "multiclass"
        )
        self.loss = nn.CrossEntropyLoss()
        self.prob = nn.Softmax()

        self.LR = learning_rate

    def forward(self, x):
        for layer in self.CNN:
            x = layer(x)
        x = self.flat(x)
        for layer in self.classifier:
            x = layer(x)
        return x

    def loss_fn(self, out, target):
        return self.loss(out, target)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.LR)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        out = self(x)
        loss = self.loss_fn(out, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        out = self(x)
        loss = self.loss_fn(out, y)
        out = self.prob(out)
        logits = torch.argmax(out, dim=1)
        accu = self.accuracy(logits, y)
        self.log("valid_loss", loss)
        self.log("val_acc_step", accu, prog_bar=True)
        return loss, accu
