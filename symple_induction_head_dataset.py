import torch
from torch.utils.data import Dataset
import random
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

class TokenInductionHeadDataset(Dataset):
    def __init__(self, tokenizer, seq_len=30, dataset_size=1000,vocab_size=100, special_token=None):                 # Volendo si può usare un token (formato stringa) presente nel dizionario di tokenizer
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_size = dataset_size
        self.vocab_size = vocab_size

        if special_token is None:
            special_token = '+'

        self.special_token = special_token

        self.data = []
        self.targets = []

        self._generate_data()

    def _generate_data(self):
      all_tokens = list(self.tokenizer.get_vocab().keys())[:self.vocab_size]
      for _ in range(self.dataset_size):
        seq = []

        for i in range(self.seq_len):
          tok = random.choice(all_tokens)
          while tok == self.special_token:
            tok = random.choice(all_tokens)
          seq.append(tok)
        seq.append(self.special_token)                                      #l'ultimo token è quello speciale

        pos = random.randint(0, self.seq_len - 1)
        seq[pos] = self.special_token
        target = seq[pos + 1]

        self.data.append(seq.copy())
        self.targets.append(target)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
        
def Make_Tokenizer():
  vocab = [
      "0","1","2","3","4","5","6","7","8","9",
      "A","B","C","D","E","F","G","H","I","J","K","L","M","+"
  ]

  tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))                       # Creo il tokenizer più semplice possibile che converte una parola in un token
  tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()                            # Gli sto dicendo di riconoscere gli spazi così posso dirgli: "A B C D" = ["A","B","C","D"]
                                                                                   # invece "ABCD" viene inteso come "[UNK]"
  trainer = trainers.WordLevelTrainer(                                             # Questo definisce il vocabolario aggiungendoci il token "[UNK]"
      vocab_size=len(vocab) + 1,                                                   # +1 per [UNK]
      special_tokens=["[UNK]"]
  )
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
