# -*- coding: utf8 -*-
# @author: chenguangyi
# @date: 2021-06-08
# @desc: 
# 基于huggingface transformers和datasets实现的SimCSE
# SimCSE: Simple Contrastive Learning of Sentence Embeddings


import logging
import os
from pathlib import Path

from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig, AlbertModel, AlbertConfig


class CSECollator(object):
    def __init__(self,
                 tokenizer,
                 features=["input_ids", "attention_mask", "token_type_ids"],
                 max_len=32):
        self.tokenizer = tokenizer
        self.features = features
        self.max_len = max_len

    def collate(self, batch):
        
        new_batch = []
        for example in batch:
            for i in range(2):
                # 每个句子重复两次
                new_batch.append({fea: example[fea] for fea in self.features})
        new_batch = self.tokenizer.pad(
            new_batch,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return new_batch


class CSENetwork(nn.Module):
    def __init__(self, pretrained="hfl/chinese-bert-wwm-ext", pool_type="cls"):
        super().__init__()
        Config = BertConfig.from_pretrained(pretrained)
        # Config = AlbertConfig.from_pretrained(pretrained)
        Config.attention_probs_dropout_prob = 0.3
        Config.hidden_dropout_prob = 0.3
        self.encoder = BertModel.from_pretrained(pretrained, config=Config)
        # self.encoder = AlbertModel.from_pretrained(pretrained, config=Config)
        assert pool_type in ["cls", "pooler"], "invalid pool_type: %s" % pool_type 
        self.pool_type = pool_type
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        if self.pool_type == "cls":
            output = output.last_hidden_state[:,0]
        elif self.pool_type == "pooler":
            output = output.pooler_output
        return output


def load_data(data_files, tokenizer, max_len=100, batch_size=32, num_proc=1):
    ds = load_dataset("text", data_files=data_files)
    ds_tokenized = ds.map(lambda example: tokenizer(example["text"]), num_proc=num_proc)
    collator = CSECollator(tokenizer, max_len=max_len)
    dl = DataLoader(ds_tokenized["train"],
                    batch_size=batch_size,
                    collate_fn=collator.collate)
    return dl


def compute_loss(y_pred, tao=0.05, device="cuda"):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / tao
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def main():
    pretrained = "/root/data/transformers/model_download/chinese-bert-wwm-ext"
    tokenizer = BertTokenizer.from_pretrained(pretrained, mirror="tuna")
    train_file = ""
    model_out = Path("./model/")

    num_proc = 10
    max_len = 100
    batch_size = 64
    learning_rate = 1e-5
    tao = 0.05
    device = "cuda"
    display_interval = 50
    save_interval = 500

    dl = load_data({"train": train_file}, tokenizer, max_len, batch_size, num_proc)
    model = CSENetwork(pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if not model_out.exists():
        os.mkdir(model_out)
    total_cnt = len(dl.dataset)
    for batch_idx, data in tqdm(enumerate(dl, 1)):
        pred = model(input_ids=data["input_ids"].to(device),
                    attention_mask=data["attention_mask"].to(device),
                    token_type_ids=data["token_type_ids"].to(device))
        loss = compute_loss(pred, tao, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % display_interval == 0:
            loss, current = loss.item(), batch_idx * batch_size
            logging.info(f"loss: {loss:>10f}  [{current:>10d}/{total_cnt:>10d}]")
        if batch_idx % save_interval == 0:
            torch.save(model.state_dict(), model_out / "batch-{0}".format(batch_idx))


if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
