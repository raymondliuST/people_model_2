from ml_dataset import *
from people_model import *
import yaml
from pm_experiment import *
import torch.nn as nn

with open('config/test.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open(config["data_params"]["tokenizer_path"], 'r') as file:
    vocab_dict = json.load(file)
    
# adding mask token
mask_token = len(vocab_dict)
vocab_dict[len(vocab_dict)] = "mask"
# dataset = mlDataset(config)
# dataloader = mlDataModule(config)
model = PeopleModel(len(vocab_dict))
exp = pmExperiment(config, len(vocab_dict))

# Loading tokenizer

def test_loss():
    batch = {
        "input": torch.LongTensor([[2,2,1,0],
                                   [2,1,0,2]]),
        "label": torch.LongTensor([[0,1,2,2],
                                   [1,2,2,0]]),
        "mask": torch.BoolTensor([[True,True,False,False],
                                 [True,False,False,True]])
    }
    #[[1,0,1,0],
    #[0,1,1,1]]
    output_logits = torch.FloatTensor([[[0.1,0.9],[0.9,0.1],[0.1,0.9],[0.9,0.1]],
                                       [[0.9,0.1],[0.1,0.9],[0.1,0.9],[0.1,0.9]]]) #batch, seq, hidden
    
    output_log_sft = nn.LogSoftmax(dim=-1)(output_logits)

    labels = batch["label"]
    cross_entropy_loss = []
    for i in range(labels.shape[0]):  # Iterate over batch
        for j in range(labels.shape[1]):  # Iterate over each class
            label = labels[i, j]
            if label!=2:
                prob = output_logits[i, j, label]
                cross_entropy_loss.append(-np.log(prob))


    loss = model.loss_func(batch, output_log_sft)
    import pdb
    pdb.set_trace()
    return

if __name__ == "__main__":
    test_loss()
    import pdb
    pdb.set_trace()