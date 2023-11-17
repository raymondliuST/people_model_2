from ml_dataset import *
from people_model import *
import yaml
from pm_experiment import *


with open('config/pm.yaml', 'r') as file:
    config = yaml.safe_load(file)

dataset = mlDataset(config)
dataset.__getitem__(0)
dataloader = mlDataModule(config)

model = PeopleModel(len(dataset.vocab_dict))
exp = pmExperiment(config, len(dataset.vocab_dict))

for batch in dataloader.train_dataloader():
    exp.validation_step(batch, None)

# import pdb
# pdb.set_trace()