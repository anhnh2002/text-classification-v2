from torch.nn import functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils import *
from bert_classifier import *
import torch
import wandb
from accelerate import Accelerator
import logging
from sklearn.utils import resample

logging.basicConfig(filename='/home/vu.ho_ttsds/FAILED_EXPERIMENTS/dsai_text_classification/logs/cls_bert_base_uncase_contrastive.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
path_to_checkpoint_model = "/home/vu.ho_ttsds/FAILED_EXPERIMENTS/dsai_text_classification/checkpoint/cls_bert_base_uncase_contrastive.pt"


def get_df(path: str) -> pd.DataFrame:
    with open(path, 'r') as file:
        content = file.readlines()
    data = [line[:-1].split('\t') for line in content]
    df = pd.DataFrame(data=data, columns=['TITLE', 'CATEGORY'])
    return df


def train(
        model_id='google-bert/bert-base-uncased',
        n_epochs=15,
        batch_size=8,
        device = "cuda:0"
):
    
    model = CLSBertClassifier.from_pretrained(model_id,
                                                token='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',
                                                num_labels=4,
                                                device_map=device
                                                )

    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.00005}], weight_decay=1e-4)
    
    

    train_anot = get_df("/home/vu.ho_ttsds/FAILED_EXPERIMENTS/dsai_text_classification/data/train.txt")
    val_anot = get_df("/home/vu.ho_ttsds/FAILED_EXPERIMENTS/dsai_text_classification/data/valid.txt")
    test_anot = get_df("/home/vu.ho_ttsds/FAILED_EXPERIMENTS/dsai_text_classification/data/test.txt")

    # # upsample health
    # n_surp_sample = 500
    # health = train_anot[train_anot["CATEGORY"] == "health"]
    # health_upsample = resample(health, random_state = 35, n_samples=n_surp_sample, replace = True)
    # # upsample science_and_technology
    # n_love_sample = 250
    # tech = train_anot[train_anot["CATEGORY"] == "science_and_technology"]
    # tech_upsample = resample(tech, random_state = 35, n_samples=n_love_sample, replace = True)

    # train_anot = pd.concat([train_anot, health_upsample, tech_upsample])

    train_dataset = CustomDataset(anot=train_anot, model_id=model_id, max_seq_len=30, device=device)
    val_dataset = CustomDataset(anot=val_anot, model_id=model_id, max_seq_len=30, device=device)
    test_dataset = CustomDataset(anot=test_anot, model_id=model_id, max_seq_len=30, device=device)

    trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # wandb.init(
    #   project="is_macro_v2",
    #   name=f"PhoBert-base-separate-encoder-new",
    #   config={
    #       "learning_rate": 0.0005,
    #       "architecture": "bert",
    #       "epochs": 20,
    #   }
    # )
    accelerator = Accelerator()

    model, optimizer, trainloader, valloader, testloader = accelerator.prepare(model, optimizer, trainloader, valloader, testloader)

    best_loss = 1e6
    best_acc = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0

        torch.cuda.empty_cache()
        for batch in tqdm(trainloader):

            optimizer.zero_grad()

            outputs = model(**batch)
            
            loss = outputs['loss'] + 0.1*constrastive_loss(reps=outputs['reps'], labels=batch['labels'])

            accelerator.backward(loss)
            optimizer.step()

            train_loss += loss.item()

        else:
            train_loss = train_loss/len(trainloader)

            with torch.no_grad():
                model.eval()

                # valid
                val_loss = 0
                category_pred = []
                category_true = []
                torch.cuda.empty_cache()
                for batch in tqdm(valloader):

                    outputs = model(**batch)

                    val_loss += (outputs["loss"].item() + 0.1*constrastive_loss(reps=outputs['reps'], labels=batch['labels']).item())

                    category_pred += outputs['logits'].argmax(dim=1).cpu().tolist()
                    category_true += batch['labels'].cpu().tolist()
                
                val_loss = val_loss/len(valloader)

                val_acc = sum(np.array(category_true) == np.array(category_pred))/len(category_true)

                # checkpoint
                if best_loss > val_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), path_to_checkpoint_model)

                # test
                category_pred = []
                category_true = []
                torch.cuda.empty_cache()
                for batch in tqdm(testloader):

                    outputs = model(**batch)

                    category_pred += outputs['logits'].argmax(dim=1).cpu().tolist()
                    category_true += batch['labels'].cpu().tolist()
                
                test_acc = sum(np.array(category_true) == np.array(category_pred))/len(category_true)

                print(f"epoch: {epoch}\ttrain loss: {train_loss: .4f}\tval loss: {val_loss: .4f}\tval accuracy: {val_acc: .4f}\ttest accuracy: {test_acc: .4f}")
                logging.info(f"epoch: {epoch}\ttrain loss: {train_loss: .4f}\tval loss: {val_loss: .4f}\tval accuracy: {val_acc: .4f}\ttest accuracy: {test_acc: .4f}")
            
    # wandb.finish()
    

if __name__ == "__main__":
    train(n_epochs=5,
        batch_size=64)