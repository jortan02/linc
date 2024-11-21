# %%
import torch
assert torch.cuda.is_available()
from datasets import load_dataset
from tqdm.auto import tqdm

from NaiveModel import NaiveModel
from LincModel import LincModel

import torch
from torch.utils.data import Dataset, DataLoader

class LogicDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        premises = self.df.iloc[idx]["premises"]
        conclusion = self.df.iloc[idx]["conclusion"]
        return premises, conclusion
# %%
import pickle
    
output_name = f"FOLIO-Llama-3.2-1B-Instruct-linc_2024-11-17_12-56-05.pkl"
with open(output_name, 'rb') as file:
    results = pickle.load(file)

# %%
results

# %%
results[0]["responses"][0][0]
# %%
