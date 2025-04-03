import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from graphs import get_contact_map, compute_SPD_centrality
from embeddings import get_PB_embeddings
from GA import GraphAttention
from GCN import GCN

#------------------------------------------------------------------------------

torch.set_default_dtype(torch.float32)
warnings.filterwarnings('ignore')

# Seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
try:
    device = torch.device('cuda:1' if torch.cuda.device_count() >= 2 else 'cuda:0')
    torch.cuda.set_device(device)
except RuntimeError:
    device = torch.device('cuda:0')

# parameters
LENG_SIZE = 1000
GCN_input_dim = 1024
GCN_hidden_dim = 256
GCN_output_dim = 63
Attention_output_dim = 64
Num_classes = 1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001

Model_Path = './Model/best_model.pkl'

#------------------------------------------------------------------------------

def get_graphs(pdb_path):
    
    contact_map, _ = get_contact_map(pdb_path)
    SPD, centrality = compute_SPD_centrality(contact_map)
    contact_map = contact_map.astype(np.float32)
    SPD = SPD.astype(np.float32)
    centrality = centrality.astype(np.float32)
    
    bias = SPD
    non_zero_mask = SPD != 0.0
    bias[non_zero_mask] = 1 / bias[non_zero_mask]
    
    centrality = centrality * 100
    
    return contact_map, bias, centrality
    # contact_map: array, size = [max_L, max_L]
    # bias : array, size = [max_L, max_L]
    # centrality: array, size = [max_L, 1]
    
def get_embedding(sequence):
    
    embedding = get_PB_embeddings(sequence)
    embedding = (embedding - embedding.mean(axis=1, keepdims=True))/embedding.std(axis =1, keepdims=True)
    embedding = np.pad(embedding,((0,LENG_SIZE-len(sequence)),(0,0)),'constant')
    
    return embedding
    # embedding: array, size = [max_L, 1024]

#------------------------------------------------------------------------------

# define training model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.gcn = GCN(GCN_input_dim, GCN_hidden_dim, GCN_output_dim)
        self.graph_attention = GraphAttention()
        self.LN = nn.LayerNorm(Attention_output_dim)
        self.fc = nn.Linear(Attention_output_dim, Num_classes)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay= WEIGHT_DECAY)

    def forward(self, x, adj, attn_bias, centrality):
        ## input x size is [bsz, max_L, 1024]
        
        # GCN module
        x = self.gcn(x, adj) #[bsz, max_L, 63]
        
        # Graphormer module
        padding_mask = x[:, :, 0].eq(0) #[bsz, max_L]
        x = torch.cat((x, centrality.unsqueeze(-1)), dim=-1)  #[bsz, max_L, 64]
        x, _ = self.graph_attention(x, centrality, attn_bias, padding_mask) #[bsz, max_L, 64]
        
        # FC Layer
        x_avg = torch.mean(x, dim=1) #[bsz, 64]
        x_avg = self.LN(x_avg)
        output = torch.sigmoid(self.fc(x_avg))
        
        return output.squeeze(0)

#------------------------------------------------------------------------------

# load variable from array to tensor
def load(array):
    array = torch.squeeze(array).float()
    array = Variable(array.cuda())
    
    return array

def predict(model, embedding, contact_map, bias, centrality):
    model.eval()
    
    with torch.no_grad():
        embedding = load(embedding)
        contact_map = load(contact_map)
        bias = load(bias)
        centrality = load(centrality)
        
        y_pred = model(embedding, contact_map, bias, centrality)
        y_pred = torch.squeeze(y_pred)
        y_pred = y_pred.float()
        y_pred = y_pred.cpu().numpy()
        
    return y_pred

#------------------------------------------------------------------------------

model = Model()
if torch.cuda.is_available():
    model.cuda()
model.load_state_dict(torch.load(Model_Path))

pdb_path = input('Input PDB File Path:')
contact_map, bias, centrality = get_graphs(pdb_path)

sequence = input('Input Sequence:')
embedding = get_embedding(sequence)

y_pred = predict(model, embedding, contact_map, bias, centrality)
print('Predicted Tm (℃):', '%.2f'% y_pred)
    
    
    
    