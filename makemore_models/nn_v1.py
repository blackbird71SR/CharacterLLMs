import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def createWordsMapping(filename = 'data/names.txt'):
  words = open(filename, 'r').read().splitlines()
  chars = sorted(list(set(''.join(words))))
  stoi = {s:i+1 for i,s in enumerate(chars)}
  stoi['.'] = 0
  itos = {i:s for s,i in stoi.items()}
  numChars = len(stoi)
  return words, stoi, itos, numChars

def createTrainingData(words):
  xs, ys = [], []
  for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
      ix1 = stoi[ch1]
      ix2 = stoi[ch2]
      xs.append(ix1)
      ys.append(ix2)
  xs = torch.tensor(xs)
  ys = torch.tensor(ys)
  return xs, ys

def initalizeWeights(numChars):
  g = torch.Generator().manual_seed(2147483647)
  W = torch.randn((numChars, numChars), generator=g, requires_grad=True)
  return W  

def trainModel(xs, ys, W, numChars, epochs):
  for k in range(epochs):
    # Forward Pass
    xenc = F.one_hot(xs, num_classes=numChars).float() # Input to the network
    logits = xenc @ W # Predict log-counts using current weights
    counts = logits.exp() # Counts (non-neg) equivalent to N in bigram approach
    probs = counts/ counts.sum(1, keepdim=True) # Probabs for next char by normalizing
    loss = - probs[torch.arange(len(ys)), ys].log().mean() # Nomralized negative log-likelihood

    # Backward Pass
    W.grad = None
    loss.backward()

    # Update weights
    W.data += -50 * W.grad

    print(f'Epoch {k}: Loss {loss.data}')
  return W

def loss(examples, W, numChars):
  xs, ys = createTrainingData(examples)
  xenc = F.one_hot(xs, num_classes=numChars).float()
  logits = xenc @ W
  counts = logits.exp()
  probs = counts/counts.sum(1, keepdim=True)
  loss = - probs[torch.arange(len(ys)), ys].log().mean()
  return loss

def generateExample(W, g, itos, numChars):
  out = []
  ix = 0
  while True:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=numChars).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts/counts.sum(1, keepdim=True)
    ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  return ''.join(out)

def generateExamples(W, itos, numChars, numExamples=10):
  g = torch.Generator().manual_seed(2147483647)
  examples = []
  for i in range(numExamples):
    example = generateExample(W, g, itos, numChars)
    examples.append(example)
  return examples

if __name__ == '__main__':
  words, stoi, itos, numChars = createWordsMapping()
  xs, ys = createTrainingData(words)
  
  W = initalizeWeights(numChars)
  trainedW = trainModel(xs, ys, W, numChars, epochs=20)
  
  examples = generateExamples(trainedW, itos, numChars)

  train_loss = loss(words, trainedW, numChars)
  test_loss = loss(examples, trainedW, numChars)

  print(f'Generated Examples: {examples}')
  print(f'Training Loss: {train_loss}')
  print(f'Testing Loss: {test_loss}')