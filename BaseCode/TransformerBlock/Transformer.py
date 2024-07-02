import torch
import torch.nn as nn
import torch.nn.functional as F

from TransformerBlock.SupportBlock.FFN import *
from TransformerBlock.SupportBlock.PositionEmbedding import *
from TransformerBlock.SupportBlock.MultiHeadAttention import *

from TransformerBlock.TransformerModel.Encoder import *
from TransformerBlock.TransformerModel.Decoder import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 trg_pad_idx,
                 enc_voc_size,
                 dec_voc_size,
                 max_len,
                 d_model,
                 n_heads,
                 ffn_hidden,
                 n_layers,
                 drop_prob,
                 device):
        super(Transformer, self).__init__()

        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_heads, n_layers, drop_prob, device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_heads, n_layers, drop_prob, device)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)

        #(Batch, Time, len_q, len_k)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)

        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)

        mask = q & k
        return mask

    def make_casual_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        #mask = torch.tril(torch.ones((len_q, len_k).type(torch.BoolTensor))).to(self.device)
        mask = (
            torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        )
        return mask

    def forward(self, src, trg):
        src = src.to("cuda" if torch.cuda.is_available() else "cpu")
        trg = trg.to("cuda" if torch.cuda.is_available() else "cpu")
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx).to(self.device)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx).to(self.device) * self.make_casual_mask(trg, trg).to(self.device)
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        enc = self.encoder(src, src_mask)
        output = self.decoder(trg, enc, trg_mask, src_trg_mask)
        return output


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

if __name__ == '__main__':
    enc_voc_size = 5893
    dec_voc_size = 7853
    src_pad_idx = 1
    trg_pad_idx = 1
    trg_sos_idx = 2
    batch_size = 128
    max_len = 1024
    d_model = 512
    n_layers = 3
    n_heads = 2
    ffn_hidden = 1024
    drop_prob = 0.1

    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        d_model=d_model,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        max_len=max_len,
        ffn_hidden=ffn_hidden,
        n_heads=n_heads,
        n_layers=n_layers,
        drop_prob=drop_prob,
        device=device,
    ).to(device)

    model.apply(initialize_weights)
    src = torch.load("D:/PythonProject/BaseCode/Data/tensor_src.pt")
    src = torch.cat((src, torch.zeros(src.shape[0], 2, dtype=torch.int)), dim=-1)
    trg = torch.load("D:/PythonProject/BaseCode/Data/tensor_trg.pt")

    result = model(src, trg)
    print(result.shape)