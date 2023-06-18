import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32 # independent sequences to process in parallel
block_size = 8  # maximum context length for predictions
max_iters = 10000
eval_interval = 1000
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
  x, y = x.to(device), y.to(device)
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
  
  @torch.no_grad()
  def estimate_loss(self, train_data, val_data):
    out = {}
    dataDict = {
      'train': train_data,
      'val': val_data
    }
    self.eval()
    for split in dataDict.keys():
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
        x, y = getBatch(dataDict[split])
        logits, loss = self.forward(x, y)
        losses[k] = loss.item()
      out[split] = losses.mean()
    self.train()
    return out
  
  def generate(self, idx, max_new_toekns):
    # idx is (B,T) array of indices in the current context
    for _ in range(max_new_toekns):
      logits, loss = self(idx) # get predictions
      logits = logits[:,-1,:] # focus only on last time step, becomes (B,C)
      probs = F.softmax(logits, dim=1) # (B,C)
      idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
      idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
    return idx
  
  def generate_text(self, max_new_toekns=400):
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    result_idx = self.generate(idx, max_new_toekns)
    return decode(result_idx[0].tolist())
  
  def trainModel(self, train_data, val_data):
    self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    for iter in range(max_iters):
      xb, yb = getBatch(train_data)
      logits, loss = self.forward(xb, yb)
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      self.optimizer.step()
      if iter % eval_interval == 0:
        losses = self.estimate_loss(train_data, val_data)
        print(f"Step:{iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  
if __name__ == '__main__':
  text = readData('data/input.txt')
  chars, vocab_size = createVocab(text)
  encode, decode = createCharMapping(chars)
  data = encodeData(text, encode)
  train_data, val_data = trainValSplit(data)
  
  xb, yb = getBatch(train_data)
  
  model = BigramLM(vocab_size)
  m = model.to(device)
  
  logits, loss = model(xb, yb)
  print("---BEFORE TRAIN ---")
  print(model.generate_text(max_new_toekns=400))
  model.trainModel(train_data, val_data)
  print("---AFTER TRAIN ---")
  print(model.generate_text(max_new_toekns=400))