import torch
from torch.utils.data import Dataset
import random

class InductionHeadDataset(Dataset):
    def __init__(self, vocab_size=50, seq_len=30, dataset_size=1000, special_token=0):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dataset_size = dataset_size
        self.special_token = special_token
        self.data = []
        self.targets = []

        self._generate_data()

    def _generate_data(self):
      for _ in range(self.dataset_size):
        seq = torch.zeros(self.seq_len+1, dtype=torch.long)
        for i in range(self.seq_len):
          random_num = random.randint(0, self.vocab_size-1)                                      # random.randint(a,b) ritorna un numero random tra a e b inclusi
          while random_num == self.special_token:                                                # evito di generare il token speciale per non creare ambiguità
            random_num = random.randint(0, self.vocab_size-1)
          seq[i] = random_num
        pos = random.randint(0, self.seq_len-1)
        seq[pos] = self.special_token
        seq[-1] = self.special_token
        target = seq[pos+1]

        self.data.append(seq.clone())
        self.targets.append(target.clone())

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
