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

from sklearn.metrics import f1_score

from PIL import Image

import wandb
wandb.login()
wandb.init(project='1219_conv_age_model')

device = torch.device('cuda')


# In[2]:


CFG={
    'IMG_SIZE':224,
    'EPOCHS':20,
    'LR': 3e-4,
    'BATCH_SIZE':32,
    'SEED':41
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


train_transform=A.Compose([A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
                           ToTensorV2()])

test_transform=A.Compose([A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
                           ToTensorV2()])


# In[5]:


age=pd.read_csv('/root/WORKSPACE/age.csv')


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
    def __init__(self, num_classes=3):
        super(BaseModel, self).__init__()
        self.backbone=models.convnext_large(pretrained=True)
        # self.backbone=models.convnext_large(pretrained=True)
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


def train(model, optimizer, train_loader, test_loader, scheduler, device):
    model.to(device)
    best_model=copy.deepcopy(model.state_dict())  
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

        if best_score<val_score:
            # best_model=model
            best_score=val_score
            best_model=copy.deepcopy(model.state_dict())

    return best_model


# In[11]:


age_train, age_val, _, _=train_test_split(age, age['label'].values, test_size=0.2, random_state=CFG['SEED'])
age_age=age_train.sort_values(by=['id'])
age_val=age_val.sort_values(by=['id'])


# In[12]:


age_img=age_train['id'].values
age_labels=age_train['label'].values
age_val_img=age_val['id'].values
age_val_labels=age_val['label'].values


# In[13]:


age_train_dataset=CustomDataset(age_img, age_labels, transform=train_transform)
age_train_loader=DataLoader(age_train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=1)
age_val_dataset=CustomDataset(age_val_img, age_val_labels, transform=test_transform)
age_val_loader=DataLoader(age_val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=1)


# In[14]:


age_model=BaseModel(num_classes=3)
age_optimizer = torch.optim.AdamW(params = age_model.parameters(), lr = CFG["LR"])
scheduler = None


# In[15]:


age_train=train(age_model, age_optimizer, age_train_loader, age_val_loader, scheduler, device)


# In[20]:


torch.save(age_train, '/root/WORKSPACE/classification/saved_model/age_model.pt')


# In[23]:


test_list=[]
path='/root/WORKSPACE/test_images'
test_data=pd.read_csv('/root/WORKSPACE/info.csv')
for i in test_data['ImageID']:
    test_list.append(path+'/'+i)


# In[24]:


test_dataset=CustomDataset(test_list, None, transform=test_transform)
test_loader=DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
