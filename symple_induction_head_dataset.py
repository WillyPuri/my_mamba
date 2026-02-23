import numpy as np
import torch
from torch.utils.data import Dataset
import random
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import string

class TokenInductionHeadDataset(Dataset):
    def __init__(self, seq_len=30, dataset_size=1000,vocab_size=100, special_token=None, num_special_tokens = None, spacing = None, fix_indx = None, sep_vocab = None):                 # Volendo si può usare un token (formato stringa) presente nel dizionario di tokenizer
        self.seq_len = seq_len
        self.dataset_size = dataset_size
        self.vocab_size = vocab_size
        self.special_token = special_token
        self.num_special_tokens = 1 if num_special_tokens is None else num_special_tokens
        self.spacing = 1 if spacing is None else spacing
        self.fix_indx = fix_indx
        self.sep_vocab = sep_vocab
        self.tokenizer = Make_Tokenizer(self.vocab_size + self.num_special_tokens)
        self.data = []
        self.targets = []
        self._generate_data()

    def _generate_data(self):
      if self.num_special_tokens > self.vocab_size or self.num_special_tokens<= 0:
        raise ValueError("Valore di num_special_tokens non valido")

      if self.spacing > self.seq_len-1 or self.spacing <= 0:
        raise ValueError("Valore di spacing non valido")

      if self.fix_indx is not None and (self.fix_indx > self.seq_len - self.spacing or self.fix_indx<0):
          raise ValueError("Valore di fix_indx non valido")

      all_tokens = list(self.tokenizer.id_to_token(i) for i in range(self.vocab_size+self.num_special_tokens))
      print(all_tokens)

      for _ in range(self.dataset_size):
        seq = []
        if self.special_token is not None and self.num_special_tokens == 1:
          special_tok = self.special_token
        elif self.sep_vocab is not None and self.sep_vocab == True:
          special_tok = random.choice(all_tokens[self.vocab_size:])  # se la variabile sep_vocab è true allora uso un vocabolario separato per gli special tokens
        else:
          special_tok = random.choice(all_tokens[:self.num_special_tokens])  # Non tutto l'alfabeto è utilizzabile come special tokens


        for i in range(self.seq_len-1):                                   # seq_len-1 perchè l'ultimo token sarà quello speciale
          tok = random.choice(all_tokens[:self.vocab_size])
          while tok == special_tok:                                # Evito che lo special token sia ripetuto più volte
            tok = random.choice(all_tokens[:self.vocab_size])
          seq.append(tok)
        seq.append(special_tok)                                    # l'ultimo token è quello speciale

        if self.fix_indx is not None:
          pos = self.fix_indx
        else:
          pos = random.randint(0, self.seq_len - (1+self.spacing))

        seq[pos] = special_tok
        target = seq[pos + self.spacing]

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

def print_sequence(seq, special_token, spacing, prediction=None):
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    colored_seq = list(seq)
    target_index = None

    for i, tok in enumerate(seq):
        if tok == special_token:
            colored_seq[i] = f"{RED}{tok}{RESET}"
            target_idx = i + spacing

            if 0 <= target_idx < len(seq):
                colored_seq[target_idx] = f"{GREEN}{colored_seq[target_idx]}{RESET}"
                target_index = target_idx

    print(" ".join(colored_seq))

    if isinstance(prediction, str) and target_index is not None:
      if prediction == seq[target_index]:
        print(f"Target: {GREEN}{colored_seq[target_index]}{RESET}  Prediction: {GREEN}{prediction}{RESET}  Result: {GREEN}True{RESET}")
      else:
        print(f"Target: {GREEN}{colored_seq[target_index]}{RESET}  Prediction: {RED}{prediction}{RESET}  Result: {RED}False{RESET}")
    else:
      print(f'Target: {GREEN}{colored_seq[target_index]}{RESET}')
