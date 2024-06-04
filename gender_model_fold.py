import random
import pandas as pd
import numpy as np
import os
import cv2
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
# import wandb
import datetime
import copy
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

from PIL import Image

import math
from torch.optim.lr_scheduler import _LRScheduler
import wandb
wandb.login()
wandb.init(project='1221_fold_conv_gender_model')
import time
start=time.time()
device = torch.device('cuda')


# In[2]:


CFG={
    'IMG_HEIGHT':512,
    'IMG_WIDTH':384,
    'NUM_CLASS':2,
    'EPOCHS':20,
    'LR': 3e-4,
    'BATCH_SIZE':32,
    'SEED':41,
    'NUM_FOLDS': 5
}


# In[3]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정


# In[4]:


train_transform=A.Compose([A.Resize(CFG['IMG_HEIGHT'], CFG['IMG_WIDTH']),
                           A.CenterCrop(300, 220, p=1),
                           A.HorizontalFlip(p=0.3),
                           A.OneOf([
                               A.MotionBlur(p=1),
                               A.OpticalDistortion(p=1),
                               A.GaussNoise(p=1)
                           ], p=0.3),
                           A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.3),
                           A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                           ToTensorV2()])

test_transform=A.Compose([A.Resize(CFG['IMG_HEIGHT'], CFG['IMG_WIDTH']),
                          A.CenterCrop(300, 220, p=1),
                          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                           ToTensorV2()])


# In[5]:


gender=pd.read_csv('/root/WORKSPACE/gender.csv')


# In[6]:


class CustomDataset(Dataset):
    def __init__(self, img_path, labels, transform=None):
        self.img_path=img_path
        self.labels=labels
        self.transform=transform

    
    def __getitem__(self, idx):
        img_path=self.img_path[idx]
        # img=np.fromfile(img_path, np.uint8)
        # img=cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image=self.transform(image=img)['image']

        if self.labels is not None:
            label=self.labels[idx]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path)


# In[7]:


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.backbone=models.convnext_large(pretrained=True)
        self.classifier=nn.Linear(1000, num_classes)

    def forward(self, x):
        x=self.backbone(x)
        x=self.classifier(x)

        return x


# In[8]:


def competition_metric(true, pred):
    return f1_score(true, pred, average="macro") 


# In[9]:


def validation(model, criterion, test_loader, device):
    model.eval() 
    
    model_preds = []
    true_labels = []
    
    val_loss = []
    
    with torch.no_grad():  
        for img, label in tqdm(iter(test_loader)): 
            img, label = img.float().to(device), label.to(device)
            
            model_pred = model(img)
            
            loss = criterion(model_pred, label)
            
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist() 
            true_labels += label.detach().cpu().numpy().tolist()

    val_f1 = competition_metric(true_labels, model_preds)  
    return np.mean(val_loss), val_f1


# In[10]:


def train(model, optimizer, train_loader, test_loader, scheduler, device, k_idx):
    model.to(device)
    
    criterion=nn.CrossEntropyLoss().to(device)
    best_score=0.0
    best_model=None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss=[]

        for i, (img, label) in enumerate(iter(train_loader)):
            img, label=img.float().to(device), label.to(device)

            optimizer.zero_grad()

            model_pred=model(img)

            loss=criterion(model_pred, label)
            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        tr_loss=np.mean(train_loss)

        val_loss, val_score=validation(model, criterion, test_loader, device)
        print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val F1 Score : [{val_score:.5f}]')
        wandb.log({'val_loss':val_loss,
                   'val_f1': val_score})

        if scheduler is not None:
            scheduler.step()

        if best_score<val_score:
            best_model=model
            best_score=val_score
            torch.save(best_model.state_dict(), f'/root/WORKSPACE/classification/soft_voting/gender/fold_best_model{k_idx+1}.pt')

    return best_model


# In[11]:


skf=StratifiedKFold(n_splits=CFG['NUM_FOLDS'], shuffle=True, random_state=CFG['SEED'])

for fold, (train_index, val_index) in enumerate(skf.split(gender['id'], gender['label'])):
    age_train_dataset=CustomDataset(gender['id'].iloc[train_index].values, gender['label'].iloc[train_index].values, transform=train_transform)
    age_train_loader=DataLoader(age_train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    age_val_dataset=CustomDataset(gender['id'].iloc[val_index].values, gender['label'].iloc[val_index].values, transform=test_transform)
    age_val_loader=DataLoader(age_val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    age_model=BaseModel(num_classes=CFG['NUM_CLASS'])
    age_optimizer=torch.optim.AdamW(params=age_model.parameters(), lr=CFG['LR'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(age_optimizer, gamma=0.95)

    train(age_model, age_optimizer, age_train_loader, age_val_loader, scheduler, device, k_idx=fold)


# In[12]:


def predict(model: nn.Module, test_loader, weight_save_path, device) -> np.array:
    model = model.to(device) 
    weight_path_list = weight_save_path
    test_probs = np.zeros(shape=(len(test_loader.dataset), CFG['NUM_CLASS']))
    for weight in weight_path_list :
        model.load_state_dict(torch.load(weight))
        model.eval()
        probs = None
        
        with torch.no_grad(): 
            for img in tqdm(iter(test_loader)):
                img = img.float().to(device)
                model_pred = model(img).cpu().numpy()
                if probs is None:
                    probs = model_pred
                else:
                    probs = np.concatenate([probs, model_pred])                

        test_probs += (probs / CFG['NUM_FOLDS']) 
    _, test_preds = torch.max(torch.tensor(test_probs), dim=1) ## 최대값과 인덱스

    return test_preds ## 라벨값 


# In[13]:


test_list=[]
path='/root/WORKSPACE/test_images'
test_data=pd.read_csv('/root/WORKSPACE/info.csv')
for i in test_data['ImageID']:
    test_list.append(path+'/'+i)


# In[14]:


test_dataset=CustomDataset(test_list, None, transform=test_transform)
test_loader=DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


# In[15]:


weight_save_path=[f'/root/WORKSPACE/classification/soft_voting/gender/fold_best_model{i}.pt' for i in range(1,6)]


# In[16]:


infer_model=BaseModel(num_classes=CFG['NUM_CLASS'])

pred=predict(infer_model, test_loader, weight_save_path, device)


# In[17]:


df_gender=pd.DataFrame({'ImageID':test_data['ImageID'],'gender':pred})


# In[18]:


df_gender.to_csv('/root/WORKSPACE/df_gender.csv', index=False)
