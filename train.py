import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pm_experiment import pmExperiment
from ml_dataset import mlDataModule

import yaml
import torch
from pytorch_lightning.loggers import WandbLogger  # Import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping


def main(dataset_config_path):
    # Define your model, dataset, and dataloader
    with open('config/pm.yaml', 'r') as file:
        config = yaml.safe_load(file)

    torch.multiprocessing.set_sharing_strategy("file_system") 


    dataloader = mlDataModule(config)
    vocab_size = len(dataloader.train_dataset.vocab_dict)
    print(f"Vocab size is {vocab_size}")
    print(f"Train Dataset contains {dataloader.train_dataset.__len__()} records")

    model_module = pmExperiment(config, vocab_size)

    # Initialize WandbLogger with your project name and any other desired settings
    wandb_logger = WandbLogger(project=config["model_params"]["name"], log_model=True)
    for key, dict in config.items():
        wandb_logger.experiment.config.update(dict)

    # Early stopping 
    early_stop_callback = EarlyStopping(
        monitor='val_loss_epoch',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='min'
    )
    
    # Define a Lightning Trainer with WandbLogger for experiment tracking
    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1,
        logger=wandb_logger,  # Use the WandbLogger for experiment tracking
        max_epochs=config["model_params"]["epoch"],  # Number of training epochs
        progress_bar_refresh_rate=1,  # Update the progress bar every 10 batches
        log_every_n_steps=1,  # Log metrics every batch
        default_root_dir = "./checkpoints",
        gradient_clip_val=0.5,
        # callbacks=[early_stop_callback]
    )

    # Train the model
    trainer.fit(model_module, dataloader.train_dataloader(), dataloader.val_dataloader())

if __name__ == "__main__":
    main("config/pm.yaml") 