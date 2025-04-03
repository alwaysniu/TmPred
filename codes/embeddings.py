from transformers import BertTokenizer, BertModel
import torch
import warnings

SEED = 42
LENG_SIZE = 1000

torch.set_default_dtype(torch.float32)
warnings.filterwarnings('ignore')
device_ids = [0]
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ProtBert model initialization
tokenizer = BertTokenizer.from_pretrained("./prot_bert")
bert_model = BertModel.from_pretrained("./prot_bert").to(device)

def get_PB_embeddings(sequence):
    seq = ' '.join(list(sequence))
    inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length= LENG_SIZE).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 1:-1, :].squeeze().cpu().numpy()
