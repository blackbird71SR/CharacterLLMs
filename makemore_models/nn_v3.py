import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

class Linear:

  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in ** 0.5
    self.bias = torch.zeros(fan_out) if bias else None

  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out

  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)

  def __call__(self, x):
    if self.training:
      xmean = x.mean(0, keepdims=True)
      xvar = x.var(0, keepdims=True, unbiased=True)
    else:
      xmean = self.running_mean
      xvar = self.running_var
    
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta

    if self.training:
      with torch.no_grad():
        self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1-self.momentum) * self.running_var + self.momentum * xvar
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

class Tanh:

  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out

  def parameters(self):
    return []

def createWordsMapping(filename = 'data/names.txt'):
  words = open(filename, 'r').read().splitlines()
  chars = sorted(list(set(''.join(words))))
  stoi = {s:i+1 for i,s in enumerate(chars)}
  stoi['.'] = 0
  itos = {i:s for s,i in stoi.items()}
  n_vocab = len(stoi)
  return words, stoi, itos, n_vocab

def buildDataset(words, block_size):
  X, Y = [], []
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix]
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X,Y

def buildDatasets(words, block_size):
  random.seed(42)
  random.shuffle(words)
  
  n1 = int(0.8 * len(words))
  n2 = int(0.9 * len(words))
  
  Xtr, Ytr = buildDataset(words[:n1], block_size)
  Xdev, Ydev = buildDataset(words[n1:n2], block_size)
  Xte, Yte = buildDataset(words[n2:], block_size)

  return Xtr, Ytr, Xdev, Ydev, Xte, Yte

def initializeModelWeights(n_vocab, block_size, n_embed, n_hidden):
  C = torch.randn((n_vocab, n_embed), generator=g)

  layers = [
    Linear(n_embed * block_size, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_vocab), BatchNorm1d(n_vocab)
  ]

  with torch.no_grad():
    layers[-1].gamma *= 0.1 # Make last layer less confident
    for layer in layers[:-1]:
      if isinstance(layer, Linear):
        layer.weight *= 5/3

  parameters = [C] + [p for layer in layers for p in layer.parameters()]
  for p in parameters:
    p.requires_grad = True
  
  print(f'Total Parameters: {sum(p.nelement() for p in parameters)}')
  return layers, parameters

def trainModel(X, Y, layers, parameters, n_epochs, batch_size):
  lossi = []

  for epoch in range(n_epochs):

    # Minibatch construct
    ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)
    X_batch, Y_batch = X[ix], Y[ix]

    # Forward Pass
    emb = parameters[0][X_batch] # embed characters into vectors
    x = emb.view(emb.shape[0], -1) # concatentae the vectors
    for layer in layers:
      x = layer(x)
    loss = F.cross_entropy(x, Y_batch)

    # Backward Pass
    for layer in layers:
      layer.out.retain_grad()
    for p in parameters:
      p.grad = None
    loss.backward()

    # Update parameters
    lr = 0.1 if epoch < 100000 else 0.01 # stop learning rate decay
    for p in parameters:
      p.data += -lr * p.grad
    
    # Track Stats
    lossi.append(loss.log10().item())
    if epoch % 10000 == 0:
      print(f'{epoch:7d}/{n_epochs:7d}: {loss.item():.4f}')
    
  return lossi, parameters

@torch.no_grad()
def loss(X, Y, layers, parameters):
  emb = parameters[0][X]
  x = emb.view(emb.shape[0], -1)
  for layer in layers:
      if isinstance(layer, BatchNorm1d):
        layer.training = False
      x = layer(x)
  loss = F.cross_entropy(x, Y)
  return loss

def generateExample(layers, parameters, block_size, itos):
  out = []
  context = [0] * block_size
  while True:
    emb = parameters[0][torch.tensor([context])] # (1,block_size, d)
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
      if isinstance(layer, BatchNorm1d):
        layer.training = False
      x = layer(x)
    probs = F.softmax(x, dim=1)
    ix = torch.multinomial(probs, num_samples=1, generator=g).item()
    context = context[1:] + [ix]
    out.append(ix)
    if ix == 0:
      break
  return ''.join(itos[i] for i in out)

def generateExamples(layers, parameters, block_size, itos, numExamples = 20):
  examples = []
  for _ in range(numExamples):
    example = generateExample(layers, parameters, block_size, itos)
    examples.append(example)
  return examples

if __name__ == '__main__':
  BLOCK_SIZE = 3
  N_EMBED = 10
  N_HIDDEN = 100
  N_EPOCHS = 200000
  BATCH_SIZE = 32
  g = torch.Generator().manual_seed(2147483647)

  words, stoi, itos, n_vocab = createWordsMapping()
  Xtr, Ytr, Xdev, Ydev, Xte, Yte = buildDatasets(words, BLOCK_SIZE)
  layers, parameters = initializeModelWeights(n_vocab, BLOCK_SIZE, N_EMBED, N_HIDDEN)
  lossi, parameters = trainModel(Xtr, Ytr, layers, parameters, N_EPOCHS, BATCH_SIZE)

  plt.plot(lossi)

  print(f'Train Loss: {loss(Xtr, Ytr, layers, parameters)}')
  print(f'Val Loss: {loss(Xdev, Ydev, layers, parameters)}')

  examples = generateExamples(layers, parameters, BLOCK_SIZE, itos)
  print(f'Generated Examples: {examples}')