from TransformerBlock.TransformerModel.TransformerEmbedding import *
from TransformerBlock.TransformerModel.DecoderLayer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, drop_prob, device):
        super(Decoder, self).__init__()

        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len, drop_prob, device)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layer)]
        )

        self.fc = nn.Linear(d_model, dec_voc_size)

    def forward(self, dec, enc, t_mask, s_mask):
        dec = dec.to("cuda" if torch.cuda.is_available() else "cpu")
        enc = enc.to("cuda" if torch.cuda.is_available() else "cpu")

        dec = self.embedding(dec)

        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)

        dec = self.fc(dec)
        return dec

