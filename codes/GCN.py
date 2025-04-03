import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

# def Graph Convolution Layer
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        mask = input[:, :, 0].ne(0) #[bsz,LENG_SIZE]
        support = input @ self.weight    # X * W
        output = adj @ support           # A * X * W
        if self.bias is not None:        # A * X * W + b
            output = (output + self.bias)*mask.unsqueeze(-1)
            return output #[bsz,LENG_SIZE,hid_dim]
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# define GCN module
class GCN(nn.Module):
    def __init__(self, GCN_input_dim, GCN_hidden_dim, GCN_output_dim):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolution(GCN_input_dim, GCN_hidden_dim)
        self.ln1 = nn.LayerNorm(GCN_hidden_dim)
        self.re1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = GraphConvolution(GCN_hidden_dim, GCN_output_dim)
        self.ln2 = nn.LayerNorm(GCN_output_dim)
        self.re2 = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = self.ln1(x)
        x = self.re1(x)
        x = self.conv2(x, adj)
        x = self.ln2(x)
        output = self.re2(x)
        return output