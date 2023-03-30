import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def createWordsMapping(filename = 'names.txt'):
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


if __name__ == '__main__':
  words, stoi, itos, numChars = createWordsMapping()
  xs, ys = createTrainingData(words)