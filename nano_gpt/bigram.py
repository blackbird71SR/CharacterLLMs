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

def read_data(filename):
  with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()
    return text
  
if __name__ == '__main__':
  text = read_data('input.txt')
  print(text[:100])