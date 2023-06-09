import importlib
import numpy as np
from torch.utils.data import DataLoader
import copy
import numpy as np
import pandas as pd
import math
import torch
import torch.utils.data.distributed as dist

from TPsource.data.list_dataset import ListDataset
from TPsource.data.batch import Batch, BatchPAD


def get_dataset(config):
    """
    according the config['dataset_class'] to create the dataset

    Args:
        config(ConfigParser): config

    Returns:
        AbstractDataset: the loaded dataset
    """
    try:
        return getattr(importlib.import_module('TPsource.data.dataset'),
                       config['dataset_class'])(config)
    except AttributeError:
        try:
            return getattr(importlib.import_module('TPsource.data.dataset.dataset_subclass'),
                           config['dataset_class'])(config)
        except AttributeError:
            raise AttributeError('dataset_class is not found')


def generate_dataloader(train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers, shuffle=True,
                        pad_with_last_sample=False):
    """
    create dataloader(train/test/eval)

    Args:
        train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        shuffle(bool): shuffle
        pad_with_last_sample(bool): 对于若最后一个 batch 不满足 batch_size的情况，是否进行补齐（使用最后一个元素反复填充补齐）。

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)
    """
    if pad_with_last_sample:
        num_padding = (batch_size - (len(train_data) % batch_size)) % batch_size
        data_padding = np.repeat(train_data[-1:], num_padding, axis=0)  # 最后一各元素
        train_data = np.concatenate([train_data, data_padding], axis=0)
        num_padding = (batch_size - (len(eval_data) % batch_size)) % batch_size
        data_padding = np.repeat(eval_data[-1:], num_padding, axis=0)
        eval_data = np.concatenate([eval_data, data_padding], axis=0)
        num_padding = (batch_size - (len(test_data) % batch_size)) % batch_size
        data_padding = np.repeat(test_data[-1:], num_padding, axis=0)
        test_data = np.concatenate([test_data, data_padding], axis=0)

    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    def collator(indices):
        # 如果 Batch 中的元素是一个字典类型，那么可以使用 collate_fn 参数来指定如何将这些元素拼接成一个张量,以适应模型的输入格式
        batch = Batch(feature_name)
        for item in indices:
            batch.append(copy.deepcopy(item))
        return batch

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=shuffle)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=False)
    return train_dataloader, eval_dataloader, test_dataloader


def generate_dataloader_pad(train_data, eval_data, test_data, feature_name,
                            batch_size, num_workers, pad_item=None,
                            pad_max_len=None, shuffle=True):
    """
    create dataloader(train/test/eval)

    Args:
        train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        pad_item(dict): 用于将不定长的特征补齐到一样的长度，每个特征名作为 key，若某特征名不在该 dict 内则不进行补齐。
        pad_max_len(dict): 用于截取不定长的特征，对于过长的特征进行剪切
        shuffle(bool): shuffle

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)
    """
    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    def collator(indices):
        batch = BatchPAD(feature_name, pad_item, pad_max_len)
        for item in indices:
            batch.append(copy.deepcopy(item))
        batch.padding()
        return batch

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=shuffle)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    return train_dataloader, eval_dataloader, test_dataloader

def cal_corr(a, b, intervals):
    def z_normalize(data, intervals):
        data_min, data_max = np.min(data), np.max(data)
        norm_data = (data - data_min) / (data_max - data_min)
        for i in range(intervals):
            data_day = norm_data[i::intervals]
            day_mean, day_std = np.mean(data_day), np.std(data_day)
            norm_data[i::intervals] = np.where(day_std == 0, 0, (data_day - day_mean) / (day_std + 1e-8))
        return norm_data

    a1, b1 = z_normalize(a, intervals), z_normalize(b, intervals)
    a1_mean, b1_mean = np.mean(a1), np.mean(b1)
    corr = np.dot(a1 - a1_mean, b1 - b1_mean) / (np.linalg.norm(a1 - a1_mean) * np.linalg.norm(b1 - b1_mean)+ 1e-8)
    return corr