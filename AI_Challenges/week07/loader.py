# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Split line into label and text
                parts = line.split(",", 1)
                if len(parts) != 2:
                    print(f"Invalid line format: {line}")
                    continue

                label_str, title = parts
                try:
                    label = int(label_str)
                except ValueError:
                    print(f"Invalid label in line: {line}")
                    continue

                if self.config["model_type"] == "bert":
                    # Tokenize with attention mask
                    encoded = self.tokenizer(
                        title,
                        max_length=self.config["max_length"],
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    input_id = encoded["input_ids"].squeeze(0)
                    attention_mask = encoded["attention_mask"].squeeze(0)
                else:
                    input_id = self.encode_sentence(title)
                    attention_mask = torch.ones(len(input_id))  # No attention mask for custom tokenization

                input_id = torch.LongTensor(input_id)
                attention_mask = torch.LongTensor(np.array(attention_mask, dtype=np.int64))
                label_index = torch.LongTensor([label])
                self.data.append([input_id, attention_mask, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_data.txt", Config)
    print(dg[1])