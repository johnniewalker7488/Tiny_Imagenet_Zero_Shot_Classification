import tarfile
import os
import numpy as np
import torch
import torchvision
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
import pandas as pd
import gensim
from gensim.models import KeyedVectors

import warnings
warnings.filterwarnings("ignore")


data_dir = '../zsl_dataset'
# print('Directories in data_dir:', os.listdir(data_dir))
# print('Train classes:', len(os.listdir(data_dir + '/train')))
# print('Zero-shot classes:', len(os.listdir(data_dir + '/zero_shot_from_train')))


def load_vectors(name='google'):
    if name == 'google':
        vectors = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
        print('Google word embeddings loaded')
        return vectors
    if name == 'wiki':
        vectors = KeyedVectors.load_word2vec_format('../wiki-news-300d-1M.vec', binary=False)
        print('Wikipedia word embeddings loaded')
        return vectors


def average_label(label):
    phrases = label.split(sep=',')
    label_words = []
    for phrase in phrases:
        words = phrase.split(sep=' ')
        for word in words:
            label_words.append(word)
    clean_words = [word for word in label_words if word != '']
    vectors = [GOOGLE_VECS.get_vector(word) if word in GOOGLE_VECS.vocab else np.zeros(300) for word in clean_words]
    average_vector = sum(vectors) / len(vectors)
    return GOOGLE_VECS.similar_by_vector(average_vector)[0][0]


def create_datasets():
    # data transforms
    train_tfms = tt.Compose([
#         tt.Resize(64),
        tt.RandomHorizontalFlip(),
        tt.RandomAffine(degrees=30, shear=30, scale=(0.9, 1.1)),
        tt.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        tt.ToTensor(),
        tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)])

    valid_tfms = tt.Compose([
#         tt.Resize(64),
        tt.ToTensor(),
        tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    train_dataset = ImageFolder(data_dir + '/train', train_tfms)
    valid_dataset = ImageFolder(data_dir + '/validation', valid_tfms)
    print('Train set images:', len(train_dataset))
    print('Validation set images:', len(valid_dataset))
    return train_dataset, valid_dataset


# split classes into seen (train) and unseen(zsl)
def split_classes():
    categories = pd.read_csv('./tiny-imagenet-200/wnids.txt', header=None).to_numpy().flatten()
    zsl_cat = categories[:50]
    train_cat = categories[50:]
    print('Categories split into seen and unseen')


def preprocess_labels():
    # get labels for classes from annotations
    labels = pd.read_csv('./tiny-imagenet-200/words.txt', sep='\t', header=None)
    train_labels_df = labels[labels[0].isin(train_cat)]
    zsl_labels_df = labels[labels[0].isin(zsl_cat)]

    # get one-word labels and corresponding vectors for train and zsl labels
    train_labels_df['average_label'] = train_labels_df[1].transform(average_label)
    train_labels_df['average_vector'] = train_labels_df['average_label'].transform(GOOGLE_VECS.get_vector)
    zsl_labels_df['average_label'] = zsl_labels_df[1].transform(average_label)
    zsl_labels_df['average_vector'] = zsl_labels_df['average_label'].transform(GOOGLE_VECS.get_vector)
    print('Labels transformed into average labels')

    label_id_list = train_labels_df[0].tolist()  # list of 'n01234567'-type train classnames
    label_vecs = {
        classname: torch.from_numpy(GOOGLE_VECS[train_labels_df[train_labels_df[0] == classname]['average_label']])
        for classname in label_id_list}  # 'n01234567'-type train classname : 1x300 google vector

    # target index in DataLoader : target 'n01234567'-type id -- for comparing vectors
    target_labels = {v: k for k, v in train_ds.class_to_idx.items()}

    label_vecs_list = list(label_vecs.values())  # list of 150 1x300 train labels vectors
    train_target_vectors = torch.cat(label_vecs_list)  # 150x300 tensor of train labels vectors

    # normalization
    train_target_vectors_norm = torch.stack([vec / torch.norm(vec) for vec in train_target_vectors])
    train_target_vectors_norm.requires_grad = False
    print('Target vectors normalized')


# train_ds, valid_ds = create_datasets()