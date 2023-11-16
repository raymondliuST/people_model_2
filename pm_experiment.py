
import torch
from torch import optim
import pytorch_lightning as pl

import torch.nn as nn

import numpy as np
from people_model import *


class pmExperiment(pl.LightningModule):

    def __init__(self,
                 config: dict, vocab_size) -> None:
        super(pmExperiment, self).__init__()

        model_config = config["model_params"]
        self.model = PeopleModel(vocab_size, 
                                 hidden=model_config["hidden"], 
                                 n_layers = model_config["n_layers"], 
                                 attn_heads = model_config["attn_heads"],
                                 dropout=model_config["dropout"])
        
        self.params = config["exp_params"]
        self.vocab_size = vocab_size

        # For batch loss calculation
        self.batch_train_loss = []
        self.batch_val_loss = []

        # For batch acc calculation
        self.batch_train_acc = []
        self.batch_val_acc = []

        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)

    def training_step(self, batch):
        outputs = self.forward(batch)

        train_loss = self.model.loss_func(batch, outputs)
        self.log("train_loss_step", train_loss)
        self.batch_train_loss.append(train_loss.cpu().detach().numpy())

        train_acc, acc_list = self.model.mlm_accuracy(batch, outputs)
        self.log("train_acc_step", train_acc)
        # self.log_dict({f"{i}_train_acc" : acc for i, acc in enumerate(acc_list)})
        self.batch_train_acc.append(train_acc.cpu().detach().numpy())

        return train_loss
    
    def on_train_epoch_end(self):
        train_loss_epoch = np.mean(self.batch_train_loss)

        self.log("training_loss_epoch", train_loss_epoch, on_step=False, on_epoch=True, prog_bar=True)
        self.batch_train_loss.clear()

        train_acc_epoch = np.mean(self.batch_train_acc)

        self.log("training_acc_epoch", train_acc_epoch, on_step=False, on_epoch=True, prog_bar=True)
        self.batch_train_acc.clear()


    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        outputs = self.forward(batch)

        val_loss = self.model.loss_func(batch, outputs)
        self.log("val_loss_step", val_loss, on_step=True, on_epoch=False)
        self.batch_val_loss.append(val_loss.cpu().detach().numpy())

        val_acc, acc_list = self.model.mlm_accuracy(batch, outputs)
        self.log("val_acc_step", val_acc, on_step=True, on_epoch=False)
        self.batch_val_acc.append(val_acc.cpu().detach().numpy())

        return val_loss

    def on_validation_epoch_end(self):
        val_loss_epoch = np.mean(self.batch_val_loss)

        self.log("val_loss_epoch", val_loss_epoch, on_step=False, on_epoch=True, prog_bar=True)
        self.batch_val_loss.clear()

        val_acc_epoch = np.mean(self.batch_val_acc)

        self.log("val_acc_epoch", val_acc_epoch, on_step=False, on_epoch=True, prog_bar=True)
        self.batch_val_acc.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.params["LR"], weight_decay = self.params["weight_decay"])
    
    

       