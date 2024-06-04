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
wandb.init(project='1219_conv_gender_model')

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


gender_train, gender_val, _, _=train_test_split(gender, gender['label'].values, test_size=0.2, random_state=CFG['SEED'])
gender_train=gender_train.sort_values(by=['id'])
gender_val=gender_val.sort_values(by=['id'])


# In[12]:


gender_img=gender_train['id'].values
gender_labels=gender_train['label'].values
gender_val_img=gender_val['id'].values
gender_val_labels=gender_val['label'].values


# In[13]:


gender_train_dataset=CustomDataset(gender_img, gender_labels, transform=train_transform)
gender_train_loader=DataLoader(gender_train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=1)
gender_val_dataset=CustomDataset(gender_val_img, gender_val_labels, transform=test_transform)
gender_val_loader=DataLoader(gender_val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=1)


# In[14]:


gender_model=BaseModel(num_classes=2)


# In[15]:


gender_optimizer = torch.optim.AdamW(params = gender_model.parameters(), lr = CFG["LR"])
scheduler = None


# In[16]:


gender_train=train(gender_model, gender_optimizer, gender_train_loader, gender_val_loader, scheduler, device)


# In[21]:


torch.save(gender_train, '/root/WORKSPACE/classification/saved_model/gender_model.pt')


# In[29]:


test_list=[]
path='/root/WORKSPACE/test_images'
test_data=pd.read_csv('/root/WORKSPACE/info.csv')
for i in test_data['ImageID']:
    test_list.append(path+'/'+i)


# In[30]:


test_dataset=CustomDataset(test_list, None, transform=test_transform)
test_loader=DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


# In[31]:


def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    
    model_preds = []
    
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.float().to(device)
            
            model_pred = model(img)
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
    
    print('Done.')
    return model_preds


# In[32]:


test_model=BaseModel(num_classes=2)


# In[33]:


test_model.load_state_dict(torch.load('/root/WORKSPACE/classification/saved_model/gender_model.pt'))


# In[34]:


gender_pred=inference(test_model, test_loader, device)


# In[40]:


df_gender=pd.DataFrame({'ImageID':test_data['ImageID'],'gender':gender_pred})


# In[41]:


df_gender.head(10)


# In[43]:


df_gender.to_csv('/root/WORKSPACE/df_gender.csv', index=False)
