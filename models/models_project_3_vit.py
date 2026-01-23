import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim) # D=192 -> hidden_dim=768,  (because mlp_ratio = 4)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim) # hidden_dim=768 -> D=192
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.gelu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # norm + mha + (residual) + norm + mlp
        # https://en.wikipedia.org/wiki/Vision_transformer#/media/File:Vision_Transformer.svg
        self.norm1 = nn.LayerNorm(dim) #normalizes each token vector across its (D) 192 features, supposed to help stabilize training (shape unchanged (B,N,D))


        # multi-head attention
        # it projects embeddings to Q,K,V
        # splits into heads, computes attention weights across tokens, mixes values
        # projects back to D=192
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,dropout=dropout, batch_first=True)
        
        self.drop_path = nn.Dropout(dropout)  

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        # x: (B, N, D)
        x_norm = self.norm1(x) # first norm

        # Q=K=V = x_norm (i.e. self-attention)
        # for each token {i}, it computes similarity to every token {j}, turn similarities into weights (softmax), replaces token {i} bya. weighted sum of all token values
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False) #multi head attention (output : (B,N,D))

        x = x + self.drop_path(attn_out) # dropout + residual connexion

        # for each token independently, input token vector length 192, expand to 768 (4*192), apply GELU, and shrink back to 192
        # it's like applying the same small neural network to each token
        x = x + self.mlp(self.norm2(x)) # second norm + mlp
        return x


class ViTTiny(nn.Module):
    def __init__(
        self,
        num_classes,
        image_size=224,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        # Note : with image_size = 224 and patch_size = 16 we have 224//16 = 14 patches per side (so patch=14)
        # We divie the 224*224 image into 196 patches of size 16*16
        assert image_size % patch_size == 0
        patch = image_size // patch_size
        num_patches = patch * patch
        

        # Patch embedding: here images are converted to tokens
        # (B,3,224,224) -> (B,embed_dim,14,14) -> (B,196,embed_dim)
        # self.patch_embed splits the image into non-overlapping 16x16 patches (because stride=patch_size)
        # and also, each patch is projected to a vector of size embed_dim (192)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # The [CLS] token is a learnable vector that acts as a global feature extractor
        # after the transformer, we use the final CLS embeding and classify from it
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # As transformers don't know order by default, we need to used position 
        # embedding to tell the model where each patch came from.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) # (1,197,192)
        # dropout to help generalization
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout)
            for _ in range(depth)
        ])


        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # input dim : (B,channel=3,image_size_h=224,image_size_w=224)
        x = self.patch_embed(x)             # (B, embed_dim=192, patch=14, patch=14)
        x = x.flatten(2).transpose(1, 2)    # (B, num_patches=196, 192)
        # at this stage, we have 196 tokens, each token being a vector of dim 192

        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)      
        # at this stage, sequence length is 1 CLS token + 196 patch tokens, so 197 tokens

        # we add the positional embedding + dropout for regularizaion (we used dropout instead of drop_path)
        x = self.pos_drop(x + self.pos_embed) # (1,197,192) to (B,197,192)

        for bloc in self.blocks:
            # with depth=6, we are using 6 TransformersBlock sequentially
            # each block let's tokens communicate (via attention) and refine their representations (via MLP)
            # after block n°1, token {i} sees all tokens once; after many blocks, token representaitons become more abstract
            # the CLS token attends to all patch tokens in every block, over depth it becomes a good summary --> used for classification
            x = bloc(x) # (B,197,192)

        x = self.norm(x)
        cls_out = x[:, 0]

        return self.head(cls_out)
