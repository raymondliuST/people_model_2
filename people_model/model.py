import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import TokenEmbedding
import numpy as np
import torch
from .embedding import BERTEmbedding
from torchmetrics.functional.classification import multiclass_accuracy

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # return self.softmax(self.linear(x))
        return self.linear(x)
    

class PeopleModel(nn.Module):
    """
    PeopleModel : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=384, n_layers=6, attn_heads=6, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.dropout = dropout
        
        self.vocab_size = vocab_size - 1
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding 
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        
        self.head = MaskedLanguageModel(self.hidden, self.vocab_size)

        self.NLL = nn.NLLLoss(ignore_index=self.vocab_size)
        self.CEL = nn.CrossEntropyLoss(ignore_index=self.vocab_size)

    def forward(self, batch):
        input = batch["input"]

        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (input != self.vocab_size).unsqueeze(1).repeat(1, input.size(1), 1).unsqueeze(1)

        # batch_masked_position = batch["mask"]
        # batch_size, sequence_length = batch["mask"].shape
        # attention_masks = []
        # for i in range(batch_size):
        #     sequence_mask = batch_masked_position[i]
        #     attention_mask = torch.zeros(sequence_length, sequence_length, device=batch["mask"].device)

        #     for j in range(sequence_length):
        #         for k in range(sequence_length):
        #             if sequence_mask[j] == False or sequence_mask[k] == False:
        #                 attention_mask[j][k] = 1 # Set a very large negative value for masked positions
        #             else:
        #                 attention_mask[j][k] = 0 

        #     attention_masks.append(attention_mask.unsqueeze(0))

        # attention_masks = torch.stack(attention_masks, dim = 0)

        x = self.embedding(input)
        
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        output = self.head(x)
        return output
    
    def loss_func(self, batch, output):
        batch_size = batch["label"].shape[0]
        seq_length = batch["label"].shape[1]
        
        prediction_matrix = output.view(batch_size * seq_length, self.vocab_size)
        target_matrix = batch["label"].view(batch_size * seq_length)
        
        loss = self.CEL(prediction_matrix, target_matrix)
        
        return loss
    
    def mlm_accuracy(self, batch, output):
        batch_size = batch["label"].shape[0]
        seq_length = batch["label"].shape[1]
        
        prediction_matrix = output.transpose(1,2)
        target_matrix = batch["label"]

        accuracy = multiclass_accuracy(prediction_matrix, target_matrix, num_classes = self.vocab_size, ignore_index=self.vocab_size)
        
        accuracy_list = []
        for i in range(seq_length):
            acc_i = multiclass_accuracy(prediction_matrix[:, :, i], target_matrix[:, i], num_classes = self.vocab_size, ignore_index=self.vocab_size)
            accuracy_list.append(acc_i)

        return accuracy, accuracy_list
        