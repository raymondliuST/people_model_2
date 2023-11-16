from ml_dataset import *
from people_model import *
import yaml
from pm_experiment import *


with open('config/pm.yaml', 'r') as file:
    config = yaml.safe_load(file)

dataset = mlDataset(config)
dataloader = mlDataModule(config)

model = PeopleModel(len(dataset.vocab_dict))
exp = pmExperiment(config, len(dataset.vocab_dict))
for batch in dataloader.train_dataloader():
    
    output = model.forward(batch)
    loss = model.loss_func(batch, output)
    acc = model.mlm_accuracy(batch, output)

    # t = model.training_step(batch)
    # v = model.on
    exp.training_step(batch)
    import pdb
    pdb.set_trace()

# import pdb
# pdb.set_trace()