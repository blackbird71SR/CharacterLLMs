import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

def createWordsMapping(filename = 'names.txt'):
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

def initializeWeights(n_vocab, block_size, n_embed, n_hidden):
  g = torch.Generator().manual_seed(2147483647)
  
  C = torch.randn((n_vocab, n_embed), generator=g)
  W1 = torch.randn((n_embed * block_size, n_hidden), generator=g)
  b1 = torch.randn(n_hidden, generator=g)
  W2 = torch.randn((n_hidden, n_vocab), generator=g)
  b2 = torch.randn(n_vocab, generator=g)

  parameters = [C, W1, b1, W2, b2]
  for p in parameters:
    p.requires_grad = True

  print(f'Total Parameters: {sum(p.nelement() for p in parameters)}')
  return parameters

def trainModel(X, Y, parameters, block_size, n_embed, n_epochs, batch_size):
  C, W1, b1, W2, b2 = parameters
  g = torch.Generator().manual_seed(2147483647)
  lossi = []

  for epoch in range(n_epochs):

    # Minibatch Construct
    ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)
    X_batch, Y_batch = X[ix], Y[ix] # batch X, Y 

    # Forward Pass
    emb = C[X_batch] # embed characters into vectors
    embcat = emb.view(emb.shape[0], -1) # concatentae the vectors
    hpreact = embcat @ W1 + b1 # hidden layer preactivation
    h = torch.tanh(hpreact) # hidden layer
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y_batch) # loss function

    # Backward Pass
    for p in parameters:
      p.grad = None
    loss.backward()

    # Update Parameters
    lr = 0.1 if epoch < 100000 else 0.01 # stop learning rate decay
    for p in parameters:
      p.data += -lr * p.grad
    
    # Track Stats
    lossi.append(loss.log10().item())
    if epoch % 10000 == 0:
      print(f'{epoch:7d}/{n_epochs:7d}: {loss.item():.4f}')
  
  trainedParameters = [C, W1, b1, W2, b2]
  return lossi, trainedParameters

@torch.no_grad()
def loss(X, Y, parameters):
  C, W1, b1, W2, b2 = parameters
  emb = C[X]
  embcat = emb.view(emb.shape[0], -1)
  hpreact = embcat @ W1 + b1
  h = torch.tanh(hpreact)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Y)
  return loss

def generateExample(parameters, block_size, g, itos):
  C, W1, b1, W2, b2 = parameters
  out = []
  context = [0] * block_size
  while True:
    emb = C[torch.tensor([context])] # (1,block_size, d)
    embcat = emb.view(1, -1)
    hpreact = embcat @ W1 + b1
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1, generator=g).item()
    context = context[1:] + [ix]
    out.append(ix)
    if ix == 0:
      break
  return ''.join(itos[i] for i in out)

def generateExamples(parameters, block_size, itos, numExamples = 20):
  g = torch.Generator().manual_seed(2147483647 + 10)
  examples = []
  for _ in range(numExamples):
    example = generateExample(parameters, block_size, g, itos)
    examples.append(example)
  return examples

if __name__ == '__main__':
  BLOCK_SIZE = 3
  N_EMBED = 10
  N_HIDDEN = 200
  N_EPOCHS = 200000
  BATCH_SIZE = 32

  words, stoi, itos, n_vocab = createWordsMapping()
  Xtr, Ytr, Xdev, Ydev, Xte, Yte = buildDatasets(words, BLOCK_SIZE)
  parameters = initializeWeights(n_vocab, BLOCK_SIZE, N_EMBED, N_HIDDEN)
  lossi, trainedParameters = trainModel(Xtr, Ytr, parameters, BLOCK_SIZE, N_EMBED, N_EPOCHS, BATCH_SIZE)

  plt.plot(lossi)

  print(f'Train Loss: {loss(Xtr, Ytr, trainedParameters)}')
  print(f'Val Loss: {loss(Xdev, Ydev, trainedParameters)}')

  examples = generateExamples(trainedParameters, BLOCK_SIZE, itos)
  print(f'Generated Examples: {examples}')