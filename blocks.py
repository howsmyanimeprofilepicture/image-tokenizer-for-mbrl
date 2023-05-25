import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 batch_norm=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if batch_norm:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class Quantizer(nn.Module):
    """Quantize embedding vectors"""

    def __init__(self, num_embeddings,
                 embedding_dim) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.emb_table = nn.Embedding(num_embeddings,
                                      embedding_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b_size, emb_dim, h, w = z.size()
        z = z.reshape(b_size,
                      emb_dim,
                      h*w)
        z = z.permute(0, 2, 1)
        z = z.reshape(b_size*h*w,
                      emb_dim)
        z = z.unsqueeze(1)
        z = z.expand(b_size*h*w,
                     self.num_embeddings,
                     emb_dim)
        W = self.emb_table.weight.detach()
        assert W.size() == (self.num_embeddings,
                            emb_dim)
        W = W.unsqueeze(0)
        W = W.expand(b_size*h*w,
                     self.num_embeddings,
                     emb_dim)
        token_ids = torch.argmin(((z - W)**2).mean(dim=-1),
                                 dim=-1)
        assert token_ids.ndim == 1
        token_ids = token_ids.reshape(b_size, h*w)
        quantized_embs = self.emb_table(token_ids)
        assert quantized_embs.size() == (b_size, h*w, emb_dim)
        quantized_embs = quantized_embs.permute(0, 2, 1)
        return quantized_embs.reshape(b_size, emb_dim, h, w)


class AttentionBlock(nn.Module):
    """
    This block is NOT MULTIHEAD!
    - https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py#L140-L192
    - https://github.com/eloialonso/iris/blob/main/src/models/tokenizer/nets.py#LL311C1-L331C44"""

    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.qkv_proj = nn.Conv2d(emb_dim,
                                  3 * emb_dim,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.out_proj = nn.Conv2d(emb_dim,
                                  emb_dim,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x):
        b_size, emb_dim, h, w = x.size()
        qkv = self.qkv_proj.forward(x)
        assert qkv.size() == (b_size, emb_dim*3, h, w)
        qkv = qkv.reshape(b_size, emb_dim*3, h*w)
        qkv = qkv.permute(0, 2, 1)
        q, k, v = qkv.split(emb_dim, dim=-1)
        assert q.size() == (b_size, h*w, emb_dim)
        attn_matrix = torch.matmul(q,
                                   k.permute(0, 2, 1))

        assert attn_matrix.size() == (b_size, h*w, h*w)
        attn_score = F.softmax(attn_matrix / emb_dim**0.5,
                               dim=-1)

        attn_output = attn_score@v
        assert attn_output.size() == (b_size, h*w, emb_dim)
        attn_output = attn_output.permute(0, 2, 1)
        attn_output = attn_output.reshape(b_size, emb_dim, h, w)
        return x + self.out_proj(attn_output)
