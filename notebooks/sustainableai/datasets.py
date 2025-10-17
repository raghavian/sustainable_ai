### Import torch and utils
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from pathlib import Path

class AerialNIST(Dataset):
    def __init__(self,path='../data/',data=['AerialNIST_00.pt','AerialNIST_01.pt','AerialNIST_02.pt','AerialNIST_03.pt'], \
                 normalize=True):
        super().__init__()

        # Load data from the pytorch file
        xs, ys = zip(*(torch.load(path+p, map_location="cpu") for p in data))
        self.data = torch.cat(xs, dim=0)
        self.target = torch.cat(ys, dim=0)

        ### Normalize intensities to be between 0-1
        if normalize:
            self.data = self.data/ self.data.max() ##########

    def __len__(self):
        ### Method to return number of data points
        return len(self.target)

    def __getitem__(self,index):
        ### Method to fetch indexed element
        return self.data[index], self.target[index].type(torch.FloatTensor)

def collate_float_long(batch):
    x, y = default_collate(batch)
    # images to float32, scale if they are uint8
    x = x.float()
    if x.dtype == torch.float32 and x.max() > 1.0:  # in case data is 0..255 as float
        x = x / 255.0
    if x.dtype == torch.uint8:                      # rare, but handle anyway
        x = x.float() / 255.0
    # labels to long for CrossEntropyLoss
    if isinstance(y, torch.Tensor) and y.dtype != torch.long:
        y = y.long()
    return x, y

def split_40k_10k_10k(ds, seed=0):
    n = len(ds)
    if n < 60000:
        raise ValueError(f"need at least 60000 samples, got {n}")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()[:60000]
    train_idx = idx[:40000]
    val_idx   = idx[40000:50000]
    test_idx  = idx[50000:60000]
    return Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx)
    
class FAIRYTALES(Dataset):
    def __init__(self, path='../data/fairytales.txt'): 
        self.lines = Path(path).read_text(encoding="utf-8").splitlines()

    def __len__(self): 
        return len(self.lines)
        
    def __getitem__(self, i): 
        return self.lines[i]

class TextDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.ids = ids
        self.seq_len = seq_len
    def __len__(self):
        return len(self.ids) - self.seq_len
    def __getitem__(self, idx):
        x = self.ids[idx:idx + self.seq_len]
        y = self.ids[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

### Textdataset handlers
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD, UNK, BOS, EOS = range(4)

def simple_subword_tokenize(text, subword_vocab):
    # Greedy longest-match per word; first subtoken of each word gets a '▁' prefix for detokenization.
    tokens = []
    for raw_word in text.split():                  # simple whitespace tokenization
        word = raw_word
        start = 0
        first = True
        while start < len(word):
            matched = False
            for end in range(len(word), start, -1):
                sub = word[start:end]
                if sub in subword_vocab:
                    tok = ("▁" + sub) if first else sub
                    tokens.append(tok)
                    start = end
                    first = False
                    matched = True
                    break
            if not matched:
                # fallback to single character
                tok = ("▁" + word[start]) if first else word[start]
                tokens.append(tok)
                start += 1
                first = False
    return tokens

def build_subword_vocab(text, seed=("un","ing","ed","er","est","ly","tion","al","ive","ize","ous","ism","able","ment"),
                        max_tokens=None, lowercase=False):
    corpus = text.lower() if lowercase else text
    base = set(seed)
    toks = simple_subword_tokenize(corpus, base)
    vocab = list(dict.fromkeys(toks))              # preserve first-seen order
    if max_tokens is not None:
        vocab = vocab[:max_tokens]
    itos = SPECIALS + vocab
    stoi = {s:i for i,s in enumerate(itos)}
    return stoi, itos
    
def encode(text, stoi, lowercase=False):
    corpus = text.lower() if lowercase else text
    base = set(k for k in stoi.keys() if k not in SPECIALS)
    toks = simple_subword_tokenize(corpus, base)
    ids = [BOS] + [stoi.get(t, UNK) for t in toks]
    return ids

def decode(ids, itos):
    # merge subtokens; '▁' marks word start -> becomes a space (except at the very beginning)
    outs = []
    for i in ids:
        if i < 0 or i >= len(itos): 
            continue
        tok = itos[i]
        if tok in SPECIALS:
            continue
        if tok.startswith("▁"):
            word = tok[1:]
            if outs:
                outs.append(" ")
            outs.append(word)
        else:
            outs.append(tok)
    return "".join(outs)