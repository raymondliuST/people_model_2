import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import TokenEmbedding
import numpy as np
import torch

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
        return self.softmax(self.linear(x))
    

class PeopleModel(nn.Module):
    """
    PeopleModel : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=1, attn_heads=12, dropout=0.1):
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
        self.vocab_size = vocab_size
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding 
        self.embedding = TokenEmbedding(self.vocab_size, self.hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        
        self.head = MaskedLanguageModel(self.hidden, self.vocab_size)

        self.CEL = nn.CrossEntropyLoss(ignore_index = -100)

    def forward(self, batch):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        input = batch["input"]

        mask = (input > 0).unsqueeze(1).repeat(1, input.size(1), 1).unsqueeze(1)

        x = self.embedding(input)
        
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        output = self.head(x)
        return output
    
    def loss_func(self, batch, output):

        batch_size = output.shape[0]
        seq_length = output.shape[1]

        label = batch["label"]
        mask = batch["mask"]
        label[~mask] = -100

        output_flat = output.view(batch_size*seq_length, self.vocab_size)
        target_flat = label.view(batch_size*seq_length)
        
        loss = self.CEL(output_flat, target_flat)

        return loss
    
    def mlm_accuracy(self, batch, output):
        
        label = batch["label"].cpu().detach().numpy()
        
        predictions = torch.argmax(output, 2).cpu().detach().numpy()
        
        relevent_indexes = np.where(label != -100)
        relevent_predictions = predictions[relevent_indexes]


        relevent_targets = label[relevent_indexes]

        corrects = np.equal(relevent_predictions, relevent_targets)

        return corrects.mean()
        