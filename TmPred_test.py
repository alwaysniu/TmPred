import os
import math
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import multiprocessing

from GA import GraphAttention
from GCN import GCN
from train_and_evaluate import evaluate

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
Model_Path = './Model/'

Graph_Path = './graphs/'
SPD_Path = './SPD/'
centrality_Path = './centrality/'
embed_path = './embeddings/'

#------------------------------------------------------------------------------

def load_graphs_embedding(uniprot_id):
    
    contact_map = np.load(Graph_Path + uniprot_id + '.npy').astype(np.float32)
    SPD = np.load(SPD_Path + uniprot_id + '.npy').astype(np.float32)
    centrality = np.load(centrality_Path + uniprot_id + '.npy').astype(np.float32)
    embedding = np.load(embed_path + uniprot_id + '.npy').astype(np.float32)
    
    bias = SPD
    non_zero_mask = SPD != 0.0
    bias[non_zero_mask] = 1 / bias[non_zero_mask]
    
    centrality = centrality * 100
    
    return contact_map, bias, centrality, embedding
    # contact_map: array, size = [max_L, max_L]
    # bias : array, size = [max_L, max_L]
    # centrality: array, size = [max_L, 1]
    # embedding: array, size = [length, 1024]

class ProDataset(Dataset):
    def __init__(self, dataframe):
        self.names = dataframe['Entry'].values
        self.sequences = dataframe['Sequence'].values
        self.labels = dataframe['Tm'].values
        
    def __getitem__(self, index):
        uniprot_id = self.names[index]
        sequence = self.sequences[index]
        label = self.labels[index]
        
        contact_map, bias, centrality, embedding = load_graphs_embedding(uniprot_id)
        embedding = (embedding - embedding.mean(axis=1, keepdims=True))/embedding.std(axis =1, keepdims=True)
        embedding = np.pad(embedding,((0,LENG_SIZE-len(sequence)),(0,0)),'constant') #[max_L, 1024]

        return uniprot_id, sequence, label, embedding, contact_map, bias, centrality
    
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

def analysis(y_true, y_pred):
    pearson = pearsonr(y_true, y_pred).statistic
    r2 = r2_score(y_true, y_pred)
    result = {'pearson': pearson, 'r2': r2}
    
    return result

# train model
def test(test_dataframe):
    test_loader = DataLoader(dataset=ProDataset(test_dataframe), batch_size=16, shuffle=True, num_workers=8)
    test_result = {}

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = Model()
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Model_Path + model_name, map_location='cuda:0'))

        test_loss, test_true, test_pred, test_name = evaluate(model, test_loader)
        result_test = analysis(test_true, test_pred)
        
        print("\n========== Evaluate Test set ==========")
        print("Test loss:", '%.3f'% test_loss,
              ", Test RMSE:", '%.2f'% (100.0 * math.sqrt(test_loss)),
              ", Test pearson:", '%.3f'% result_test['pearson'],
              ", Test r2:", '%.3f'% result_test['r2'])

        test_result[model_name] = [
            test_loss,
            100.0 * math.sqrt(test_loss),
            result_test['pearson'],
            result_test['r2'],
        ]
        
        test_true_T=list(np.array(test_true) * 100)
        test_pred_T=list(np.array(test_pred) * 100)
        test_detail_dataframe = pd.DataFrame({'id': test_name, 'y_true': test_true, 'y_pred': test_pred, 'Tm':test_true_T, 'prediction':test_pred_T})
        test_detail_dataframe.sort_values(by=['id'], inplace=True)
        test_detail_dataframe.to_csv(Result_Path + model_name + "_test_detail.csv", header=True, sep=',')

    test_result_dataframe = pd.DataFrame.from_dict(test_result, orient='index',
                                                   columns=['loss', 'rmse','pearson', 'r2'])
    test_result_dataframe.to_csv(Result_Path + "test_result.csv", index=True, header=True, sep=',')

#------------------------------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    data_path = input('input data path:')
    test_dataframe = pd.read_csv(data_path)
    test(test_dataframe)