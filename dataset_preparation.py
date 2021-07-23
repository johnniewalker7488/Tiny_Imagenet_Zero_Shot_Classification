import os
# import numpy as np
# import tarfile
import shutil
# from urllib.request import urlretrieve
import zipfile
import pandas as pd

dataset_name = 'tiny-imagenet-200'
path = '.'

def prepare_dataset():
    with zipfile.ZipFile(os.path.join(path, dataset_name + ".zip"), 'r') as archive:
        archive.extractall()

    # move validation images to subfolders by class
    val_root = os.path.join(path, dataset_name, "val")
    with open(os.path.join(val_root, "val_annotations.txt"), 'r') as f:
        for image_filename, class_name, _, _, _, _ in map(str.split, f):
            class_path = os.path.join(val_root, class_name)
            os.makedirs(class_path, exist_ok=True)
            os.rename(
                os.path.join(val_root, "images", image_filename),
                os.path.join(class_path, image_filename))

    os.rmdir(os.path.join(val_root, "images"))

    os.remove(os.path.join(val_root, "val_annotations.txt"))

    os.mkdir('zsl_dataset')

    original_data_dir = './tiny-imagenet-200/train'
    # dataset split directory
    base_dir = './zsl_dataset'

    # directiories for split data
    train_dir = os.path.join(base_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    zsl_train_dir = os.path.join(base_dir, 'zero_shot_from_train')
    if not os.path.exists(zsl_train_dir):
        os.mkdir(zsl_train_dir)
    zsl_val_dir = os.path.join(base_dir, 'zero_shot_from_val')
    if not os.path.exists(zsl_val_dir):
        os.mkdir(zsl_val_dir)

    categories = pd.read_csv('./tiny-imagenet-200/wnids.txt', header=None).to_numpy().flatten()
    zsl_cat = categories[:50]
    train_cat = categories[50:]

    # copy zsl train directories from train
    zsl_train_dirs = ['{}'.format(i) for i in zsl_cat]
    for zsl_dir in zsl_train_dirs:
        src = os.path.join(original_data_dir, zsl_dir)
        dst = os.path.join(zsl_train_dir, zsl_dir)
        shutil.move(src, dst)

    # copy train directories
    train_dirs = ['{}'.format(i) for i in train_cat]
    for train_dir_name in train_dirs:
        src = os.path.join(original_data_dir, train_dir_name)
        dst = os.path.join(train_dir, train_dir_name)
        shutil.move(src, dst)

    # copy zsl val directories from train
    zsl_val_dirs = ['{}'.format(i) for i in zsl_cat]
    for zsl_dir in zsl_val_dirs:
        src = os.path.join('./tiny-imagenet-200/val', zsl_dir)
        dst = os.path.join(zsl_val_dir, zsl_dir)
        shutil.move(src, dst)

    # copy val directories
    val_dirs = ['{}'.format(i) for i in train_cat]
    for val_dir_name in val_dirs:
        src = os.path.join('./tiny-imagenet-200/val', val_dir_name)
        dst = os.path.join(validation_dir, val_dir_name)
        shutil.move(src, dst)

    data_dir = './zsl_dataset'

    print(os.listdir(data_dir))
    classes = os.listdir(data_dir + '/train')
    print(len(classes), 'classes in train')

    val_classes = os.listdir(data_dir + '/validation')
    print(len(val_classes), 'classes in validation')

    zsl_classes = os.listdir(data_dir + '/zero_shot_from_val')
    print(len(zsl_classes), 'classes in zero shot validation')
