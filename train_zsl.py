import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
from time import time

from tqdm import tqdm

import utils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_zsl(model, 
              num_epochs, 
              train_loader, 
              val_loader,
              label_vecs,
              target_labels,
              max_lr=0.001,
              crit='ces',
              coef=1,
              scheduler='one', 
              cycles=1):

    optimizer = optim.Adam(model.parameters(), lr=max_lr, weight_decay=1e-4)
    
    if scheduler == 'one':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=num_epochs,
                                                        steps_per_epoch=len(train_loader))
    if scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_epochs*len(train_loader))//cycles)

    history = []
    learning_rates = []
    total_val_loss = []

    for epoch in range(num_epochs):
        train_losses = []
        train_img_losses = []
        train_vec_losses = []
        train_accs = []
        lrs = []
        start = time()
        
#         loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for batch_idx, (train_data, train_target) in enumerate(train_loader):
            # allocate batch tensors to device
            train_data = train_data.to(device)
            train_target = train_target.to(device)
            
             # target class indices from DataLoader
            target_batch = [class_num.item() for class_num in train_target]
            
            # target class ids from target batch
            class_names_batch = [target_labels[class_num] for class_num in target_batch] 
            
            # batch of target vectors
            vector_batch = torch.cat([label_vecs[class_name] for class_name in class_names_batch]).to(device)

            model.zero_grad()  # zero gradients for next calculation

            train_pred, train_emb = model(train_data)  # forward pass

            train_img_loss = F.cross_entropy(train_pred, train_target)
            
            if crit == 'ces':
                cosine_loss = nn.CosineEmbeddingLoss()
                cosine_target = torch.ones(300, len(train_target)).to(device)
                train_vec_loss = cosine_loss(train_emb, vector_batch, cosine_target)
            
            if crit == 'mse':
                train_vec_loss = F.mse_loss(train_emb, vector_batch)
            
            train_loss = train_img_loss + (coef * train_vec_loss)
            train_img_losses.append(train_img_loss.item())
            train_vec_losses.append((coef * train_vec_loss).item())
            train_losses.append(train_loss.item())

            train_acc = utils.accuracy(train_pred, train_target)  # calculate accuracy per batch
            train_accs.append(train_acc.item())

            train_loss.backward()  # calculate gradients with backprop
            optimizer.step()  # update weights
            scheduler.step()
            lrs.append(utils.get_lr(optimizer))
#             loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')

        # validation step
        val_accs = []  # accuracies for every batch in validation set
        val_losses = []  # losses for every batch in validation set
        val_img_losses = []
        val_vec_losses = []
        
        with torch.no_grad():
            for batch_idx, (val_data, val_target) in enumerate(val_loader):
    #             model.eval()  # enable no_grad for evaluation

                val_data = val_data.to(device)  # allocate batch tensor to device
                val_target = val_target.to(device)  # allocate batch tensor to device

                val_pred, val_emb = model(val_data)  # forward pass for batch in validation set

                target_batch = [class_num.item() for class_num in val_target]
                class_names_batch = [target_labels[class_num] for class_num in target_batch]
                vector_batch = torch.cat([label_vecs[class_name] for class_name in class_names_batch]).to(device)

                val_img_loss = F.cross_entropy(val_pred, val_target)

                if crit == 'ces':
                    cosine_loss = nn.CosineEmbeddingLoss()
                    cosine_target = torch.ones(300, len(val_target)).to(device)
                    val_vec_loss = cosine_loss(val_emb, vector_batch, cosine_target)

                if crit == 'mse':
                    val_vec_loss = F.mse_loss(val_emb, vector_batch)

                val_loss = val_img_loss + (coef * val_vec_loss)
                val_img_losses.append(val_img_loss.item())
                val_vec_losses.append((coef * val_vec_loss).item())
                val_losses.append(val_loss.item())

                val_acc = utils.accuracy(val_pred, val_target)  # calculate accuracy per batch
                val_accs.append(val_acc.item())

        avg_train_loss = torch.tensor(train_losses).mean().item()  # train loss per epoch
        avg_train_acc = torch.tensor(train_accs).mean().item()  # train accuracy per epoch
        avg_val_loss = torch.tensor(val_losses).mean().item()  # validation loss per epoch
        avg_val_acc = torch.tensor(val_accs).mean().item()  # validation accuracy per epoch
        avg_val_img_loss = torch.tensor(val_img_losses).mean().item()
        avg_val_vec_loss = torch.tensor(val_vec_losses).mean().item()
        
        total_val_loss.append(avg_val_loss)

        end_time = time() - start
        line = "Epoch[{:d}] Train_loss: {:.4f}  Val_loss: {:.4f} Val_CE: {:.4f} Val_vec_loss: {:.4f}\t Train_acc: {:.4f}  Val_acc: {:.4f} LR: {:.5f} time: {:.2f}".format(
            epoch,
            train_loss.item(),
            avg_val_loss,
            avg_val_img_loss,
            avg_val_vec_loss,
            avg_train_acc,
            avg_val_acc,
            utils.get_lr(optimizer),
            end_time)
        

        print(line)

        if avg_val_loss <= torch.tensor(total_val_loss).min().item():
#             checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
#             utils.save_checkpoint(checkpoint)
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, './model_weights/model_weights.pth')

        # dict with metrics per epoch for further visualization
        epoch_end = {'train_loss': avg_train_loss, 'train_acc': avg_train_acc,
                     'val_loss': avg_val_loss, 'val_acc': avg_val_acc}
        history.append(epoch_end)
        learning_rates.append(lrs)

    print(f'Max validation accuracy: {round(torch.tensor([x["val_acc"] for x in history]).max().item(), 4)}')
    learning_rates = torch.tensor(learning_rates).clone().detach().flatten()

    return history, learning_rates