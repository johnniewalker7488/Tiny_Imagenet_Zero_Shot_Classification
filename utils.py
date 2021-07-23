import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time

def get_lr(optimizer):
    """ Returns current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def save_checkpoint(state, file_name='model_weights.pth.tar'):
    #     print('--> Saving checkpoint')
    torch.save(state, file_name)


def load_checkpoint(checkpoint, optimizer):
    print('--> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def test_model(model, num_examples=256, num_classes=150, input_size=64, input_channels=3):
    
    x = torch.randn((num_examples, input_channels, input_size, input_size))
    model = model(input_channels, num_classes)
    start = time()
    print(f'Random batch of size {num_examples} output shape:', model(x)[0].shape)
    end = time() - start
    print(f'Forward pass ran in {round(end, 2)} seconds')


def plot_losses(history, learning_rates, train_loader, file_name='loss', figsize=(10, 7)):
    train_losses = [x['train_loss'] for x in history]
    val_losses = [x['val_loss'] for x in history]
    lrs = [x for i, x in enumerate(learning_rates) if i % (len(train_loader)) == 0]

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    ax1.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')

    min_val_loss = val_losses.index(min(val_losses)) + 1
    ax1.axvline(min_val_loss, linestyle='--', color='r', label='Early Stopping Checkpoint')

    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.set_ylabel('learning rate')
    ax2.plot(range(1, len(lrs) + 1), lrs, label='Learning Rate', color='green', linestyle='--')

    ax1.grid(True)
    fig.legend()
    fig.tight_layout()
#     print('Saving losses plot')
#     plt.savefig(f'./learning_curves/{file_name}')
    plt.show()


def plot_accuracies(history, learning_rates, train_loader, file_name='accs', figsize=(10, 7)):
    train_accs = [x['train_acc'] for x in history]
    val_accs = [x['val_acc'] for x in history]
    lrs = [x for i, x in enumerate(learning_rates) if i % (len(train_loader)) == 0]

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accs')
    ax1.plot(range(1, len(train_accs) + 1), train_accs, label='Training Accuracy')
    ax1.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy')

    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.set_ylabel('learning rate')
    ax2.plot(range(1, len(lrs) + 1), lrs, label='Learning Rate', color='green', linestyle='--')

    ax1.grid(True)
    fig.legend()
    fig.tight_layout()
#     print('Saving accuracies plot')
#     plt.savefig(f'./learning_curves/{file_name}')
    plt.show()


def predict_image(img, model):
    x = img.unsqueeze(0).to(device)
    y = model(x)[0]
    _, pred = torch.max(y, dim=1)
    return train_ds.classes[pred[0].item()]


def get_embedding(img, model):
    x = img.unsqueeze(0).to(device)
    emb = model(x)[1]
    print()
    return emb
