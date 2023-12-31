import torch
import json
from torch.utils.data import Dataset
from pyspark.sql import SparkSession
from .tokenizer import *
import pandas as pd
import numpy as np
import random

class mlDataset(Dataset):
    def __init__(self, config, partition="train"):

        self.dataset_config = config["data_params"]
        self.categorical_columns = ["browserFamily","deviceType", "os","country"]

        # Loading dataset
        self.df = pd.read_parquet(self.dataset_config["data_path"])

        # Loading tokenizer
        with open(self.dataset_config["tokenizer_path"], 'r') as file:
            self.vocab_dict = json.load(file)

        # adding mask token
        self.mask_token = len(self.vocab_dict)
        self.vocab_dict[len(self.vocab_dict)] = "mask"
        
        if partition == "train":
            self.df = self.df
        else:
            self.df = self.df[:5000]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        datapoint = torch.LongTensor(self.df.iloc[index].tolist())

        rand = torch.rand(datapoint.shape)
        # has 80 percent chance to be masked
        mask_arr = rand < 0.15
        # while all(mask_arr) == True:
        #     mask_arr = rand < 0.8
        mask_arr = torch.zeros(4, dtype=torch.bool)
        mask_arr[random.randint(0, 3)] = True

        selection = torch.flatten((mask_arr).nonzero()).tolist()
        not_selection = torch.flatten((~mask_arr).nonzero()).tolist()

        input = datapoint.clone()
        label = datapoint.clone()
        input[selection] = self.mask_token
        label[not_selection] = self.mask_token

        item = {
            "input":input,
            "label":label,
            "mask":mask_arr
        }

        return item
        