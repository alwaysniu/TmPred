import torch
from torch.autograd import Variable
from tqdm import tqdm

# load variable from array to tensor
def load(array):
    array = torch.squeeze(array).float()
    array = Variable(array.cuda())
    
    return array

# train one epoch
def train_one_epoch(model, data_loader, epoch):
    epoch_loss = 0.0
    n_batches = 0
    for data in tqdm(data_loader):
        model.optimizer.zero_grad()
        _, _, label, embedding, c_map, bias, centrality = data
            
        embedding = load(embedding)
        c_map = load(c_map)
        bias = load(bias)
        centrality = load(centrality)

        y_true = load(label)
        y_true = y_true.float()/100.0
            
        y_pred = model(embedding, c_map, bias, centrality)
        y_pred = torch.squeeze(y_pred)
        
        loss = model.criterion(y_pred, y_true)
        loss = loss.float()
        loss.backward()
        
        model.optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
       
    epoch_loss_avg = epoch_loss / n_batches
    return epoch_loss_avg

# evaluate model
def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n_batches = 0
    pred_lst = []
    true_lst = []
    name_lst = []
        
    for data in tqdm(data_loader):
        with torch.no_grad():
            name, _, label, embedding, c_map, bias, centrality = data
                
            embedding = load(embedding)
            c_map = load(c_map)
            bias = load(bias)
            centrality = load(centrality)

            y_true = load(label)
            y_true = y_true.float()/100.0
                
            y_pred = model(embedding, c_map, bias, centrality)
            y_pred = torch.squeeze(y_pred)
                
            loss = model.criterion(y_pred, y_true)
            loss = loss.float()
            
            y_pred = y_pred.cpu().detach().numpy().tolist()
            y_true = y_true.cpu().detach().numpy().tolist()
            
            flag = isinstance(y_pred,float)
            if(flag):
                a = []
                a.append(y_pred)
                y_pred = a
                
            pred_lst.extend(y_pred)
            true_lst.extend(y_true)
            name_lst.extend(name)

        epoch_loss += loss.item()
        n_batches += 1
              
    epoch_loss_avg = epoch_loss / n_batches

    return epoch_loss_avg, true_lst, pred_lst, name_lst
