import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import wandb

from model import LeNetClassifier


def init_zero(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class LeNetLitModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.model = LeNetClassifier(
            num_classes=cfg.model.num_classes,
            in_channels=cfg.model.in_channels,
            img_size=cfg.model.img_size,
        )

        if cfg.model.init == "kaiming":
            self.model.apply(init_kaiming)
        elif cfg.model.init == "xavier":
            self.model.apply(init_xavier)
        elif cfg.model.init == "zero":
            self.model.apply(init_zero)
        else:
            raise ValueError(f"Unknown init method: {cfg.model.init}")

        self.criterion = nn.CrossEntropyLoss()

        self.test_preds = []
        self.test_targets = []


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())

        loss = self.criterion(logits, y)
        acc = (preds == y).float().mean()

        self.log("test/loss", loss)
        self.log("test/acc", acc)

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)

        # Reset buffers
        self.test_preds.clear()
        self.test_targets.clear()

        # Confusion matrix
        cm = torchmetrics.functional.confusion_matrix(
            preds,
            targets,
            num_classes=self.hparams.model.num_classes
        )

        # Log to W&B
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    preds=preds.numpy(),
                    y_true=targets.numpy(),
                    class_names=[str(i) for i in range(self.hparams.model.num_classes)]
                )
            })


    def configure_optimizers(self):
        if self.hparams.optimizer.name == "adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.optimizer.lr,
                weight_decay=self.hparams.optimizer.weight_decay,
            )
        elif self.hparams.optimizer.name == "sgd":
            return torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.optimizer.lr,
                momentum=self.hparams.optimizer.momentum,
            )
