import numpy as np
import torch
from torch.utils.data import Dataset
import random
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import string

class TokenInductionHeadDataset(Dataset):
    def __init__(self, tokenizer, seq_len=30, dataset_size=1000,vocab_size=100, special_token="A", num_special_tokens = 1):                 # Volendo si può usare un token (formato stringa) presente nel dizionario di tokenizer
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_size = dataset_size
        self.vocab_size = vocab_size
        self.special_token = special_token
        self.num_special_tokens = num_special_tokens
        self.data = []
        self.targets = []
        self._generate_data()

    def _generate_data(self):
      all_tokens = list(self.tokenizer.id_to_token(i) for i in range(self.vocab_size))

      for _ in range(self.dataset_size):
        seq = []
        if self.special_token == "random":
          special_tok = random.choice(all_tokens[:self.num_special_tokens])  # Non tutto l'alfabeto è utilizzabile come special tokens
        else:
          special_tok = self.special_token

        for i in range(self.seq_len-1):                                   # seq_len-1 perchè l'ultimo token sarà quello speciale
          tok = random.choice(all_tokens)
          while tok == special_tok:                                # Evito che lo special token sia ripetuto più volte
            tok = random.choice(all_tokens)
          seq.append(tok)
        seq.append(special_tok)                                    # l'ultimo token è quello speciale

        pos = random.randint(0, self.seq_len - 2)
        seq[pos] = special_tok
        target = seq[pos + 1]

        self.data.append(seq.copy())
        self.targets.append(target)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def Make_Tokenizer(vocab_size):
  caratteri = (
        string.ascii_uppercase +
        string.ascii_lowercase +
        string.digits +
        string.punctuation
    )
  vocab = list(caratteri)

  if vocab_size > len(vocab):
    extra = vocab_size - len(vocab)
    vocab.extend(str(i+10) for i in range(extra))                                  # Nel caso vocab_size sia troppo grande => aggiungo numeri al vocabolario

  vocab = vocab[:vocab_size]

  tokenizer = Tokenizer(models.WordLevel())                                        # Creo il tokenizer più semplice possibile che converte una parola in un token
  tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()                            # Gli sto dicendo di riconoscere gli spazi così posso dirgli: "A B C D" = ["A","B","C","D"]

  trainer = trainers.WordLevelTrainer(vocab_size=len(vocab))
  tokenizer.train_from_iterator(vocab, trainer)                                    # crea il tokenizer vero e proprio

  return tokenizer

def From_Seq_To_Numb(tokenizer,dataset):                                           # Questa funzione cambia il dataset da stringhe ad interi tramite il tokenizer
  vocab = tokenizer.get_vocab()

  data = []
  targets = []

  for seq, target in dataset:
      seq_ids = [vocab[token] for token in seq]
      target_id = vocab[target]

      data.append(seq_ids)
      targets.append([target_id])

  data = np.array(data)
  targets = np.array(targets)

  data = torch.Tensor(data).long()
  targets = torch.Tensor(targets).long()

  return data,targets

def print_sequence(seq, special_token):
    colored_seq = []
    for tok in seq:
        if tok == special_token:
            colored_seq.append(f"{RED}{tok}{RESET}")
        else:
            colored_seq.append(tok)
    print(" ".join(colored_seq))
