import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch.nn import functional as F
from typing import Optional

category_mapping = {
    "business": 0,
    "entertainment": 1,
    "health": 2,
    "science_and_technology": 3
}

class CustomDataset(Dataset):
    """
    Custom dataset
    """
    def __init__(self, anot: pd.DataFrame, model_id: str = 'google-bert/bert-base-cased', max_seq_len: int = 30, device='cpu', two_stage=False):
        super().__init__()
        self.anot = anot
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.two_stage = two_stage

    def __len__(self):
        return len(self.anot)

    def __getitem__(self, idx):
        r = self.anot.iloc[idx]
        input_ids = self.tokenizer(r.TITLE, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_seq_len).to(self.device)
        if not self.two_stage:
            return {
                "input_ids": input_ids["input_ids"][0],
                "attention_mask": input_ids["attention_mask"][0],
                "labels": torch.tensor(category_mapping[r.CATEGORY]).to(self.device),
            }
        return {
                "input_ids": input_ids["input_ids"][0],
                "attention_mask": input_ids["attention_mask"][0],
                "labels": torch.tensor(category_mapping[r.CATEGORY]).to(self.device),
                'unique_id': torch.tensor(idx)
            }
    
    def get_list_item(self, idxs):
        texts = [self.anot.iloc[idx].TITLE for idx in idxs]
        input_ids = self.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_seq_len).to(self.device)
        return {
                "input_ids": input_ids["input_ids"],
                "attention_mask": input_ids["attention_mask"]
            }


def constrastive_loss(
        reps: torch.Tensor,
        labels: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    reps: [B, hidden_size]
    labels: [B]
    """
    batch_size, hidden_size = reps.shape[0], reps.shape[1]

    if temperature is None:
        temperature = hidden_size
    
    sim = (reps@reps.T)/hidden_size # pairwise dot product

    denominator = sim[:,1:].sum(dim=-1) # ignore self dot product

    # sim[i][j] = 0 if j is negative of i
    mask = labels.unsqueeze(1) == labels  
    sim[~mask] = 0
    sim[torch.eye(batch_size, dtype=torch.bool)] = 0

    # number of positive samples in cluster
    number_of_pos = labels.clone()
    for label in labels.unique():
        pos_mask = labels == label
        number_of_pos[pos_mask] = pos_mask.sum()

    sim = sim.sum(dim=1)/denominator
    sim = torch.where(sim > 0, sim, 1) # case inf
    loss = -(torch.log(sim)/number_of_pos).sum()
    return loss

# def construct_triplet(
#         prev_model,
#         dataloader,
# )