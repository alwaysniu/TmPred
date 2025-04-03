import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import multiprocessing
from tqdm import tqdm

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

Result_Path = './Result/'
Model_Path = './Model/best_model.pkl'
Structure_Path = './structures/'

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

class ProDataset(Dataset):
    def __init__(self, dataframe):
        self.names = dataframe['Entry'].values
        self.sequences = dataframe['Sequence'].values
        
    def __getitem__(self, index):
        uniprot_id = self.names[index]
        sequence = self.sequences[index]
        
        contact_map, bias, centrality = get_graphs(Structure_Path + uniprot_id + '.pdb')
        embedding = get_embedding(sequence)

        return uniprot_id, sequence, embedding, contact_map, bias, centrality
    
    def __len__(self):
        return len(self.labels)

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

def predict(model, data_loader):
    model.eval()
    
    pred_lst = []
    name_lst = []
    seqs_lst = []
        
    for data in tqdm(data_loader):
        with torch.no_grad():
            name, seq, embedding, c_map, bias, centrality = data
                
            embedding = load(embedding)
            c_map = load(c_map)
            bias = load(bias)
            centrality = load(centrality)
                
            y_pred = model(embedding, c_map, bias, centrality)
            y_pred = torch.squeeze(y_pred)   
            y_pred = y_pred.cpu().detach().numpy().tolist()
            
            flag = isinstance(y_pred,float)
            if(flag):
                a = []
                a.append(y_pred)
                y_pred = a
                
            pred_lst.extend(y_pred)
            name_lst.extend(name)
            seqs_lst.extend(seq)

    return pred_lst, name_lst, seqs_lst

def predict_df(dataframe):
    loader = DataLoader(dataset=ProDataset(dataframe), batch_size=16, shuffle=True, num_workers=8)

    model = Model()
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(Model_Path))
    
    print("\n========== Predicting ==========")
    
    pred_lst, name_lst, seqs_lst = predict(model, loader)
    pred_lst_T=list(np.array(pred_lst) * 100)
    
    detail_dataframe = pd.DataFrame({'Entry': name_lst, 'Sequence': seqs_lst, 'Predicted Tm': pred_lst_T})
    detail_dataframe.sort_values(by=['id'], inplace=True)
    detail_dataframe.to_csv(Result_Path + "prediction.csv", header=True, sep=',')

#------------------------------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    data_path = input('input data path:')
    dataframe = pd.read_csv(data_path)
    predict_df(dataframe)