import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from tqdm.notebook import tqdm

#device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SOFDataset(Dataset):
  
  def __init__(self, csv_file):
    # NaN값 제거
    self.dataset = pd.read_csv(csv_file).dropna(axis=0) 
    
    # 중복제거
    self.dataset.drop_duplicates(subset=['Body'], inplace=True)
    
    # Y 값 숫자로 레이블링
    
    self.dataset['Y'] = self.dataset['Y'].map({'LQ_CLOSE':0, 
                                      'LQ_EDIT':1,
                                      'HQ':2})
    
    self.tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator') # google/electra-small-discriminator, distilbert-base-uncased

    print(self.dataset.describe())
  
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, idx):
    row = self.dataset.iloc[idx,[4,7]].values # 데이터셋 따라 idx 변경
    text = row[0]
    y = row[1]

    inputs = self.tokenizer(
        text, 
        return_tensors='pt',
        truncation=True,
        max_length=256,
        pad_to_max_length=True,
        add_special_tokens=True
        )
    
    input_ids = inputs['input_ids'][0]
    attention_mask = inputs['attention_mask'][0]

    return input_ids, attention_mask, y

train_dataset = SOFDataset("/home/harock96/hw/project/data/train_concat.csv")
valid_dataset = SOFDataset("/home/harock96/hw/project/data/valid_concat.csv")
test_dataset = SOFDataset("/home/harock96/hw/project/data/test_concat.csv")

#model

model = ElectraForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 3)
model = nn.DataParallel(model)
model.to(device)

# parameter

epochs = 1

optimizer = AdamW(model.parameters(), lr=1e-4)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Learn

losses = []
accuracies = []
val_losses = []
val_accuracies = []

for i in range(epochs):
  total_loss = 0.0
  correct = 0
  total = 0
  batches = 0

  model.train()

  for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_loader):
    optimizer.zero_grad()
    y_batch = y_batch.to(device)
    y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
    loss = F.cross_entropy(y_pred, y_batch)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    _, predicted = torch.max(y_pred, 1)
    correct += (predicted == y_batch).sum()
    total += len(y_batch)

    batches += 1
    #if batches % 100 == 0:
      #print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)
  
  losses.append(total_loss)
  accuracies.append(correct.float() / total)
  print(i+1,"-","Train Loss:", total_loss, "Accuracy:", correct.float() / total)
    
  #torch.save(model.state_dict(), "model_practice.pt")
  #model.load_state_dict(torch.load("model_practice.pt"))
    
  test_correct = 0
  test_total = 0
  test_total_loss = 0

  for input_ids_batch, attention_masks_batch, y_batch in tqdm(valid_loader):
    #optimizer.zero_grad()
    y_batch = y_batch.to(device)
    y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]

    loss = F.cross_entropy(y_pred, y_batch)
    #loss.backward()
    #optimizer.step()

    test_total_loss += loss.item()

    _, predicted = torch.max(y_pred, 1)
    test_correct += (predicted == y_batch).sum()
    test_total += len(y_batch)

  val_losses.append(test_total_loss)
  val_accuracies.append(test_correct.float() / test_total)
    
  print(i+1,"-","Valid Loss:", test_total_loss, "Accuracy:", test_correct.float() / test_total)


# accuracy 캐스팅

for i in range(epochs):
    accuracies[i] = accuracies[i].item()
for i in range(epochs):
    val_accuracies[i] = val_accuracies[i].item() 


# plot

plt.plot(losses)
plt.plot(val_losses)
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(['Train','Valid'], loc='upper left')
plt.show()

# plot

plt.plot(accuracies)
plt.plot(val_accuracies)
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend(['Train','Valid'], loc='upper left')
plt.show()


# 모델 저장하기

torch.save(model.state_dict(), "model_electra.pt") # max_length 256, batch_size 128, optimizer AdamW, learning rate 1e-4

# 모델 불러오기

model.load_state_dict(torch.load("model_electra.pt"))

# model.eval()

test_correct = 0
test_total = 0

for input_ids_batch, attention_masks_batch, y_batch in tqdm(test_loader):
  y_batch = y_batch.to(device)
  y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
  _, predicted = torch.max(y_pred, 1)
  test_correct += (predicted == y_batch).sum()
  test_total += len(y_batch)

print("Test Accuracy:", test_correct.float() / test_total)