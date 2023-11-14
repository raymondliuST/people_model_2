import lightning.pytorch as pl
import torch
import os
from ml_dataset import * 
from torch.utils.data import DataLoader

class mlDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.train_batch_size = config["data_params"]["train_batch_size"]
        self.val_batch_size = config["data_params"]["val_batch_size"]

        self.train_dataset = mlDataset(config, partition="train")
        self.val_dataset = mlDataset(config, partition="validation")

        self.num_workers = min(config["data_params"]["num_workers"], os.cpu_count())

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, shuffle=True)
    
    # def collate_fn(self, data):
    #     return list(data)
