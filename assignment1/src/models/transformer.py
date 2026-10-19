import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channel=1, num_hiddens=32):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.conv = nn.Conv2d(in_channel, num_hiddens, kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        return self.conv(X).flatten(2).transpose(1, 2)

class ViTMLP(nn.Module):
    def __init__(self, mlp_num_inputs, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.Linear(mlp_num_inputs, mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(mlp_num_hiddens, mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(self.dense1(x)))))
    
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = nn.MultiheadAttention(num_hiddens, num_heads, dropout, use_bias, batch_first=True)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(num_hiddens, mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X):
        X = X + self.attention(*([self.ln1(X)] * 3))[0]
        return X + self.mlp(self.ln2(X))

class Transformer(nn.Module):
    def __init__(self, img_size=28, patch_size=4, num_hiddens=32, 
                 mlp_num_hiddens=64, num_heads=4, num_blks=3, emb_dropout=0.1, blk_dropout=0.1, 
                 lr=0.1, use_bias=True, num_classes=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, 1, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))
        
    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
        