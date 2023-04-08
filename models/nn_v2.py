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
  numChars = len(stoi)
  return words, stoi, itos, numChars

def buildDataset(words, blockSize):
  X, Y = [], []
  for w in words:
    context = [0] * blockSize
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix]
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X,Y

def buildDatasets(words, blockSize):
  random.seed(42)
  random.shuffle(words)
  
  n1 = int(0.8 * len(words))
  n2 = int(0.9 * len(words))
  
  Xtr, Ytr = buildDataset(words[:n1], blockSize)
  Xdev, Ydev = buildDataset(words[n1:n2], blockSize)
  Xte, Yte = buildDataset(words[n2:], blockSize)

  return Xtr, Ytr, Xdev, Ydev, Xte, Yte

def initializeWeights(numChars, blockSize, numDimensions, layer1Neurons):
  g = torch.Generator().manual_seed(2147483647)
  
  C = torch.randn((numChars, numDimensions), generator=g)
  W1 = torch.randn((numDimensions * blockSize, layer1Neurons), generator=g)
  b1 = torch.randn(layer1Neurons, generator=g)
  W2 = torch.randn((layer1Neurons, numChars), generator=g)
  b2 = torch.randn(numChars, generator=g)

  parameters = [C, W1, b1, W2, b2]
  for p in parameters:
    p.requires_grad = True

  print(f'Total Parameters: {sum(p.nelement() for p in parameters)}')
  return parameters

def trainModel(X, Y, parameters, blockSize, numDimensions, epochs):
  C, W1, b1, W2, b2 = parameters
  stepi, lossi = [], []

  for epoch in range(epochs):

    # Minibatch Construct
    ix = torch.randint(0, X.shape[0], (32,))

    # Forward Pass
    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1, blockSize * numDimensions) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])

    # Backward Pass
    for p in parameters:
      p.grad = None
    loss.backward()

    # Update Parameters
    lr = 0.1 if epoch < 50000 else 0.01
    for p in parameters:
      p.data += -lr * p.grad
    
    # Track Stats
    stepi.append(epoch)
    lossi.append(loss.log10().item())

    if epoch % 10000 == 0:
      print(f'Epoch:{epoch}, Minibatch Loss:{loss.item()}')
  
  trainedParameters = [C, W1, b1, W2, b2]
  return stepi, lossi, trainedParameters

def loss(X, Y, parameters, blockSize, numDimensions):
  C, W1, b1, W2, b2 = parameters
  emb = C[X]
  h = torch.tanh(emb.view(-1, blockSize * numDimensions) @ W1 + b1)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Y)
  return loss

def generateExample(parameters, blockSize, g, itos):
  C, W1, b1, W2, b2 = parameters
  out = []
  context = [0] * blockSize
  while True:
    emb = C[torch.tensor([context])] # (1,blockSize, d)
    h = torch.tanh(emb.view(1,-1) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1, generator=g).item()
    context = context[1:] + [ix]
    out.append(ix)
    if ix == 0:
      break
  return ''.join(itos[i] for i in out)

def generateExamples(parameters, blockSize, itos, numExamples = 20):
  g = torch.Generator().manual_seed(2147483647 + 10)
  examples = []
  for _ in range(numExamples):
    example = generateExample(parameters, blockSize, g, itos)
    examples.append(example)
  return examples

if __name__ == '__main__':
  BLOCK_SIZE = 3
  EMBEDDING_DIMENSIONS = 10
  LAYER_1_SIZE = 200
  NUM_EPOCHS = 100000

  words, stoi, itos, numChars = createWordsMapping()
  Xtr, Ytr, Xdev, Ydev, Xte, Yte = buildDatasets(words, BLOCK_SIZE)
  parameters = initializeWeights(numChars, BLOCK_SIZE,EMBEDDING_DIMENSIONS, LAYER_1_SIZE)
  stepi, lossi, trainedParameters = trainModel(Xtr, Ytr, parameters, BLOCK_SIZE, EMBEDDING_DIMENSIONS, NUM_EPOCHS)

  print(f'Training Loss: {loss(Xtr, Ytr, trainedParameters, BLOCK_SIZE, EMBEDDING_DIMENSIONS)}')
  print(f'Dev Loss: {loss(Xdev, Ydev, trainedParameters, BLOCK_SIZE, EMBEDDING_DIMENSIONS)}')
  print(f'Test Loss: {loss(Xte, Yte, trainedParameters, BLOCK_SIZE, EMBEDDING_DIMENSIONS)}')

  examples = generateExamples(trainedParameters, BLOCK_SIZE, itos)
  print(f'Generated Examples: {examples}')