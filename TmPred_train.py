import os
import math
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import multiprocessing

from GA import GraphAttention
from GCN import GCN
from train_and_evaluate import train_one_epoch, evaluate

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
Model_Path = './Model'

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
def train(model, train_dataframe, valid_dataframe, fold=0):
    train_loader = DataLoader(dataset=ProDataset(train_dataframe), batch_size=16, shuffle=True, num_workers=8, drop_last=True)
    valid_loader = DataLoader(dataset=ProDataset(valid_dataframe), batch_size=16, shuffle=True, num_workers=8, drop_last=True)

    train_losses = []
    train_rmse = []
    train_pearson = []
    train_r2 = []
    
    valid_losses = []
    valid_rmse = []
    valid_pearson = []
    valid_r2 = []

    best_val_loss = 1000
    best_epoch = 0

    for epoch in range(40):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        
        model.train()
        train_loss = train_one_epoch(model, train_loader, epoch + 1)
        
        print("========== Evaluate Train set ==========")
        
        _, train_true, train_pred, train_name = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)
        print("Train loss:", '%.3f'% train_loss,
              ", Train RMSE:", '%.2f'% (100.0 * math.sqrt(train_loss)),
              ", Train pearson:", '%.3f'% result_train['pearson'],
              ", Train r2:", '%.3f'% result_train['r2'])
        train_losses.append(train_loss)
        train_rmse.append(100.0 * math.sqrt(train_loss))
        train_pearson.append(result_train['pearson'])
        train_r2.append(result_train['r2'])
        
        print("========== Evaluate Valid set ==========")
        
        valid_loss, valid_true, valid_pred, valid_name = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred)
        print("Valid loss:", '%.3f'% valid_loss,
              ", Valid RMSE:", '%.2f'% (100.0 * math.sqrt(valid_loss)),
              ", Valid pearson:", '%.3f'% result_valid['pearson'],
              ", Valid r2:", '%.3f'% result_valid['r2'])
        valid_losses.append(valid_loss)
        valid_rmse.append(100.0 * math.sqrt(valid_loss))
        valid_pearson.append(result_valid['pearson'])
        valid_r2.append(result_valid['r2'])

        if best_val_loss > valid_loss:
            nodrop = 0
            best_val_loss = valid_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pkl'))
            
            valid_true_T=list(np.array(valid_true)*100.0)
            valid_pred_T=list(np.array(valid_pred)*100.0)
            valid_detail_dataframe = pd.DataFrame({'uniprot_id': valid_name, 'Tm':valid_true_T, 'prediction':valid_pred_T})
            valid_detail_dataframe.sort_values(by=['Tm'], inplace=True)
            valid_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_valid_detail.csv", header=True, sep=',')
            
            train_true_T=list(np.array(train_true)*100.0)
            train_pred_T=list(np.array(train_pred)*100.0)
            train_detail_dataframe = pd.DataFrame({'uniprot_id': train_name, 'Tm':train_true_T, 'prediction':train_pred_T})
            train_detail_dataframe.sort_values(by=['Tm'], inplace=True)
            train_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_train_detail.csv", header=True, sep=',')
        
        else:
            nodrop += 1
        
        print(f'Now nodrop is {nodrop}.')
        
        if nodrop == 10:
            break

    # save calculation information
    result_all = {
        'Train_loss': train_losses,
        'Train_rmse': train_rmse,
        'Train_pearson': train_pearson,
        'Train_r2': train_r2,
        
        'Valid_loss': valid_losses,
        'Valid_rmse': valid_rmse,
        'Valid_pearson': valid_pearson,
        'Valid_r2': valid_r2,
        
        'Best_epoch': [best_epoch for _ in range(len(train_losses))]
    }
    result = pd.DataFrame(result_all)
    print("Fold", str(fold), "Best epoch at", str(best_epoch))
    result.to_csv(Result_Path + "Fold" + str(fold) + "_result.csv", sep=',')

def cross_validation(all_dataframe,fold_number=10):
    print("split_seed: ", SEED)
    sequence_names = all_dataframe['Entry'].values
    sequence_labels = all_dataframe['Tm'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Training on", str(train_dataframe.shape[0]), "examples, Validation on", str(valid_dataframe.shape[0]),
              "examples")
        model = Model().to(device)

        train(model, train_dataframe, valid_dataframe, fold + 1)
        fold += 1

#------------------------------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    data_path = input('input data path:')
    train_dataframe = pd.read_csv(data_path)
    cross_validation(train_dataframe,fold_number=5)