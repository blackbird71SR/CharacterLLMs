
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
  return words, numChars, stoi, itos

def createFreqMapping(words, numChars, stoi):
  N = torch.zeros((numChars, numChars), dtype=torch.int32)
  for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
      ix1 = stoi[ch1]
      ix2 = stoi[ch2]
      N[ix1][ix2] = N[ix1][ix2] + 1
  return N  