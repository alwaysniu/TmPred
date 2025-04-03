import torch.nn as nn
from MHA import MultiHeadAttention

class GraphAttention(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=256, num_heads=8, dropout=0.1):
        super(GraphAttention, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.ln_attn = nn.LayerNorm(embed_dim)
        self.ln_ffn = nn.LayerNorm(embed_dim)
        
        self.calc_attn = MultiHeadAttention(embed_size=self.embed_dim, heads=self.num_heads, dropout=self.dropout)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout_module = nn.Dropout(dropout)

    def forward(self, query, centrality, bias, key_padding_mask):
        ## input size is (N, L, GCN_out_size)
        ## centrality size is (N, L)
        ## bias size is (N, L, L)
        ## key_padding_mask size is (N, L)

        x = query
        assert x.size(-1) == self.embed_dim, f"Expected embed_dim {self.embed_dim}, but got {x.size(-1)}"
        # now x size is (N, L, embed_size=64)
        
        bias = bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # now bias size is (N, num_heads, L, L)
        
        # MHA + resi
        res = x
        x = self.ln_attn(x)  
        x, attn_weights = self.calc_attn(query=x, attn_bias=bias, mask=key_padding_mask)
        # now x size is (N, L, embed_size), attn_weights size is (N, L, L)
        x = self.dropout_module(x)
        x += res
        
        # FFN + 残差
        res = x
        x = self.ln_ffn(x)  
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout_module(x)
        x += res 
        
        ## output size is (N, L, 64)
        ## output attn_weights size is (N, L, L)

        return x, attn_weights

        
        
        
        
        
        
