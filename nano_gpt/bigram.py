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
  
if __name__ == '__main__':
  text = readData('input.txt')
  chars, vocab_size = createVocab(text)
  encode, decode = createCharMapping(chars)
  data = encodeData(text, encode)
  print(data[:100])