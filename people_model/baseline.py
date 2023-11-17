from ml_dataset import *
import yaml
import torch.nn as nn
import torch
from torchmetrics.functional.classification import multiclass_accuracy
from .embedding import BERTEmbedding


with open('config/pm.yaml', 'r') as file:
    config = yaml.safe_load(file)


class SimpleModel(nn.Module):

    def __init__(self,input_size, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, vocab_size)
        self.activate_func = nn.Tanh()
        self.BCE = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.CEL = nn.CrossEntropyLoss(ignore_index=self.vocab_size)

        # embedding 
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

    def loss_func(self, batch, output):
        batch_size = batch["label"].shape[0]
        seq_length = batch["label"].shape[1]
        
        prediction_matrix = output.view(batch_size * seq_length, self.vocab_size)
        target_matrix = batch["label"].view(batch_size * seq_length)
        
        loss = self.CEL(prediction_matrix, target_matrix)
        
        return loss

    def forward(self, x):
        x = x["input"]
        x = self.embedding(x)
        x = self.activate_func(self.fc1(x.float()))
        x = self.activate_func(self.fc2(x))
        x = self.activate_func(self.fc3(x))

        return x

    def metric(self, batch, output):
        batch_size = batch["label"].shape[0]
        seq_length = batch["label"].shape[1]
        
        prediction_matrix = output.transpose(1,2)
        target_matrix = batch["label"]

        accuracy = multiclass_accuracy(prediction_matrix, target_matrix, num_classes = self.vocab_size, ignore_index=self.vocab_size)
        

        return accuracy
# if __name__ == '__main__':
#     dataset = mlDataset(config)
#     dataloader = mlDataModule(config)

#     model = SimpleModel(input_size = 4,hidden=128,vocab_size = 51)
#     exp = pmExperiment(config, len(dataset.vocab_dict))
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     for batch in dataloader.train_dataloader():
#         input_data = batch['input']
#         label = batch['label']
#         output = model.forward(input_data)
#         loss = model.loss_func(batch, output)
#         prediction_matrix = torch.sigmoid(output)
#         prediction_matrix = torch.where(prediction_matrix > 0.5, 1, 0)
#         acc = torch.sum(label*prediction_matrix)/torch.sum(prediction_matrix)
#         recall = torch.sum(label*prediction_matrix)/torch.sum(label)
#         print('loss',loss)
#         print('postive acc',acc)
#         print('postive recall',recall)
#         #acc = model.mlm_accuracy(batch, output)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     for batch in dataloader.val_dataloader():
#         input_data = batch['input']
#         label = batch['label']
#         output = model.forward(input_data)
#         loss = model.loss_func(batch, output)
#         prediction_matrix = torch.sigmoid(output)
#         prediction_matrix = torch.where(prediction_matrix > 0.5, 1, 0)
#         acc = torch.sum(label*prediction_matrix)/torch.sum(prediction_matrix)
#         recall = torch.sum(label*prediction_matrix)/torch.sum(label)
#         print('val loss',loss)
#         print('val positive acc',acc)
#         print('val positive recall',recall)
