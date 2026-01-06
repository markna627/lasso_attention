import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

class ARDataset(Dataset):
    def __init__(self, num_pairs, num_samples=5000):
        self.BOS = 0
        self.SEP = 1
        self.QRY = 2

        KEY_START = 3
        VALUE_START = 1000

        self.num_pairs = num_pairs
        self.num_samples = num_samples

        self.all_keys = torch.arange(KEY_START, KEY_START + num_samples)
        self.all_values = torch.arange(VALUE_START, VALUE_START + num_samples)

        perm = torch.randperm(num_samples)
        self.all_values = self.all_values[perm]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        perm = torch.randperm(self.num_samples)[:self.num_pairs]

        keys = self.all_keys[perm]
        values = self.all_values[perm]

        q_idx = torch.randint(0, self.num_pairs, (1,)).item()
        query_key = keys[q_idx]
        target_value = values[q_idx]

        seq = [self.BOS]
        for k, v in zip(keys.tolist(), values.tolist()):
            seq.extend([k, v, self.SEP])
        seq.extend([self.QRY, query_key])

        return torch.tensor(seq, dtype=torch.long), target_value

#Regular Dataset
def get_data():
    softmax_dataset = ARDataset(5, 1000)

    softmax_train_dataset, softmax_valid_dataset = random_split(softmax_dataset, [0.8, 0.2])

    softmax_train_loader = DataLoader(softmax_train_dataset, batch_size = 32, shuffle = True)
    softmax_valid_loader = DataLoader(softmax_valid_dataset, batch_size = 32, shuffle = True)



    #LASSO Dataset
    lasso_dataset = ARDataset(5, 1000)
    lasso_train_dataset, lasso_valid_dataset = random_split(lasso_dataset, [0.8, 0.2])

    lasso_train_loader = DataLoader(lasso_train_dataset, batch_size = 32, shuffle = True)
    lasso_valid_loader = DataLoader(lasso_valid_dataset, batch_size = 32, shuffle = True)
    return softmax_train_loader, softmax_valid_loader, lasso_train_loader, lasso_valid_loader

