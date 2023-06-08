import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32 # independent sequences to process in parallel
block_size = 8  # maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

torch.manual_seed(1337)

def readData(filename):
  with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()
    return text
  
def createVocab(text):
  chars = sorted(list(set(text)))
  vocab_size = len(chars)
  return chars, vocab_size

def createCharMapping(chars):
  stoi = {ch:i for i,ch in enumerate(chars)}
  itos = {i:ch for i,ch in enumerate(chars)}
  encode = lambda s: [stoi[c] for c in s]
  decode = lambda l: ''.join([itos[i] for i in l])
  return encode, decode

def encodeData(text, encode):
  data = torch.tensor(encode(text), dtype=torch.long)
  return data

def trainValSplit(data, train_ratio = 0.9):
  n = int(train_ratio * len(data))
  train_data = data[:n]
  val_data = data[n:]
  return train_data, val_data

def getBatch(data):
  ix = torch.randint(len(data) - block_size, (batch_size, ))
  x = torch.stack([data[i:i+block_size] for i in ix]) # shape: ()
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x,y

class BigramLM(nn.Module):
  
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)
    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss
  
  def generate(self, idx, max_new_toekns):
    # idx is (B,T) array of indices in the current context
    for _ in range(max_new_toekns):
      logits, loss = self(idx) # get predictions
      logits = logits[:,-1,:] # focus only on last time step, becomes (B,C)
      probs = F.softmax(logits, dim=1) # (B,C)
      idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
      idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
    return idx

  
if __name__ == '__main__':
  text = readData('input.txt')
  chars, vocab_size = createVocab(text)
  encode, decode = createCharMapping(chars)
  data = encodeData(text, encode)
  train_data, val_data = trainValSplit(data)
  
  xb, yb = getBatch(train_data)
  m = BigramLM(vocab_size)
  logits, loss = m(xb, yb)
  idx = torch.zeros((1,1), dtype=torch.long)
  print(decode(m.generate(idx, max_new_toekns=400)[0].tolist()))
