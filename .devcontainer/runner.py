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
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model and dataset information.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--model_choice', type=str, required=True, help='Choice of the model')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--batch_size', type=int, required=True, help='Size of the batches')

    args = parser.parse_args()
    model_name = args.model_name # "bigcode/starcoder2-3b" "meta-llama/Llama-3.2-3B-Instruct"
    model_choice = args.model_choice # "naive"
    dataset_name = args.dataset_name # "yale-nlp/FOLIO"
    batch_size = args.batch_size # 2
    print(f"Model Name: {args.model_name}")
    print(f"Model Choice: {args.model_choice}")
    print(f"Dataset Name: {args.dataset_name}")
    print(f"Batch Size: {args.batch_size}")

    # %%
    match(model_choice):
        case "naive":        
            model = NaiveModel(model_name)
        case "linc":
            model = LincModel(model_name)
        case _:
            raise ValueError(f"Model choice {model_name} does not exist.")

    # %%
    # Login using e.g. `huggingface-cli login` to access this dataset
    if dataset_name == "yale-nlp/FOLIO":
        ds = load_dataset("yale-nlp/FOLIO")

        validation_df = ds["validation"].to_pandas()
        error_indices = [3, 109, 110, 111, 6, 28, 30, 48, 113, 115, 139, 140, 10, 11, 12, 88, 106, 107, 108, 174, 175, 176]
        validation_df = validation_df.drop(error_indices)

    # %%
    
    validation_dataset = LogicDataset(validation_df)
    dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    results = []
    for premises, conclusion in tqdm(dataloader):
        batch_results = model.answer(premises, conclusion)
        results.extend(batch_results)
        
    # %%
    import pickle

    output_name = f"{dataset_name.split("/")[-1]}-{model_name.split("/")[-1]}-{model_choice}.pkl"
    with open(output_name, 'wb') as file:
        pickle.dump(results, file)

    # %%

    # with open(output_name, 'rb') as file:
    #     results = pickle.load(file)

# %%
