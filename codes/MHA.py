import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.scaling = self.head_dim ** -0.5

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
        self.dropout_module = nn.Dropout(dropout)
        
    def split_heads(self, x):
        # Split the embedding into multiple heads
        batch_size, seq_length, embed_size = x.size()
        x = x.view(batch_size, seq_length, self.heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch_size, heads, seq_length, head_dim)

    def forward(self, query, attn_bias=None, mask=None):
        ## input size is (N, query_len, embed_size)
        N = query.shape[0]  # batch size
        query_len = query.shape[1]

        values = self.values(query)
        keys = self.keys(query)
        queries = self.queries(query)*self.scaling

        # Split into multiple heads
        values = self.split_heads(values)  # (N, heads, value_len, head_dim)
        keys = self.split_heads(keys)      # (N, heads, key_len, head_dim)
        queries = self.split_heads(queries)  # (N, heads, query_len, head_dim)

        # Calculate attn_weights
        attn_weights = torch.matmul(queries, keys.transpose(2, 3))  # (N, heads, query_len, key_len)
        
        # Add bias
        if attn_bias is not None:
            attn_weights += attn_bias
        
        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)  # (N, heads, query_len, key_len)
        attn_probs = self.dropout_module(attn_weights)  # (N, heads, query_len, key_len)
        attn_weights_avg = attn_weights.mean(dim=1)
        
        # Apply attention
        attn = torch.matmul(attn_probs, values)  # (N, heads, query_len, head_dim)

        # Concatenate heads
        attn = attn.permute(0, 2, 1, 3).contiguous()  # (N, query_len, heads, head_dim)
        attn = attn.view(N, query_len, self.embed_size)  # (N, query_len, embed_size)

        # Final linear layer
        attn = self.fc_out(attn)
        
        ## output attn size is (N, query_len, embed_size)
        ## output attn_weights_avg size is (N, query_len, key_len)

        return attn, attn_weights_avg

