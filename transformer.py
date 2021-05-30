import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Embedding(nn.Module):

    def __init__(self, vocab_size: int, dimensions: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dimensions)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 80):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        return x


class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(10)
        self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer())

    def forward(self, src, src_mask):
        pass
