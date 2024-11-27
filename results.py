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
    
output_name = f"results/FOLIO-Llama-2-7b-chat-hf-cot_2024-11-26_00-50-43.json"
with open(output_name, 'rb') as file:
    results = pickle.load(file)

# %%
results

# %%
results[0]["responses"][0]
