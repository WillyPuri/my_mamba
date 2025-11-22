import torch
from torch.utils.data import Dataset
import random
from transformers import AutoTokenizer

class TokenInductionHeadDataset(Dataset):
    def __init__(self, tokenizer, seq_len=30, dataset_size=1000, special_token=None):                 # Volendo si può usare un token (formato stringa) presente nel dizionario di tokenizer
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_size = dataset_size

        if special_token is None:
            special_token = tokenizer.pad_token

        self.special_token = special_token

        self.data = []
        self.targets = []

        self._generate_data()

    def _generate_data(self):
      all_tokens = list(self.tokenizer.get_vocab().keys())
      for _ in range(self.dataset_size):
        seq = []

        for i in range(self.seq_len):
          tok = random.choice(all_tokens)
          while tok == self.special_token:
            tok = random.choice(all_tokens)
          seq.append(tok)
        seq.append(self.special_token)                                      #l'ultimo token lo eguaglio a quello speciale
        
        pos = random.randint(0, self.seq_len - 1)
        seq[pos] = self.special_token
        target = seq[pos + 1]

        self.data.append(seq.copy())
        self.targets.append(target)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
