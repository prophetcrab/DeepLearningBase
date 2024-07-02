from TransformerBlock.TransformerModel.TransformerEmbedding import *
from TransformerBlock.TransformerModel.EncoderLayer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, env_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, drop_prob, device):
        super(Encoder, self).__init__()

        self.embedding = TransformerEmbedding(env_voc_size, d_model, max_len, drop_prob, device)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layer)]
        )

    def forward(self, x, s_mask):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, s_mask)
        return x

