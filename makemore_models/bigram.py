
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

def createFreqMapping(words, stoi, numChars):
  N = torch.zeros((numChars, numChars), dtype=torch.int32)
  for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
      ix1 = stoi[ch1]
      ix2 = stoi[ch2]
      N[ix1][ix2] = N[ix1][ix2] + 1
  return N

def plotFreqMapping(N, itos, numChars):
  plt.figure(figsize=(16, 16))
  plt.imshow(N, cmap='Blues')

  for i in range(numChars):
    for j in range(numChars):
      chstr = itos[i] + itos[j]
      plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
      plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
  plt.axis('off')

def createProbMatrix(N, isSmooth=True):
  if isSmooth:
    P = (N+1).float() # to avoid division by zero
  else:
    P = N.float()
  P /= P.sum(dim=1, keepdim=True)
  return P

def generateExample(P, g, itos):
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  return ''.join(out)

def generateExamples(P, itos, numExamples=10):
  g = torch.Generator().manual_seed(2147483647)
  examples = []
  for i in range(numExamples):
    example = generateExample(P, g, itos)
    examples.append(example)
  return examples

def loss(examples, showLogs = False):
  log_likelihood = 0.0
  n = 0

  for word in examples:
    if showLogs:
      print(f'Word: {word}')
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
      ix1 = stoi[ch1]
      ix2 = stoi[ch2]
      prob = P[ix1][ix2]
      logprob = torch.log(prob)
      log_likelihood += logprob
      n += 1
      if showLogs:
        print(f'{ch1}{ch2}: {prob=:.4f} {logprob=:.4f}')
    
  nll = - log_likelihood # negative log likelihood
  normalized_nll = nll/n

  if showLogs:
    print(f'{log_likelihood=}')
    print(f'{nll=}')
    print(f'{normalized_nll=}')
  
  return normalized_nll
    

if __name__ == '__main__':
  words, stoi, itos, numChars = createWordsMapping()
  N = createFreqMapping(words, stoi, numChars)
  plotFreqMapping(N, itos, numChars)
  P = createProbMatrix(N)
  examples = generateExamples(P, itos)
  
  train_loss = loss(words)
  test_loss = loss(examples)

  print(f'Generated Examples: {examples}')
  print(f'Training Loss: {train_loss}')
  print(f'Testing Loss: {test_loss}')
  
  loss(['josh'], showLogs=True)
  loss(['abcdjq'], showLogs=True)