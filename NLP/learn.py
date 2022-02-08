import torch
import numpy as np
import load
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = load.SOFDataset("/home/harock96/hw/project/data/train_concat.csv")

#model

model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels = 3)
model = nn.DataParallel(model)
model.to(device)


# parameter

# parameter

epochs = 5

optimizer = AdamW(model.parameters(), lr=1e-4)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=43, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=43, shuffle=True)

# Learn

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

# 모델 저장하기

# 모델 저장하기

torch.save(model.state_dict(), "model_electra.pt") # max_length 256, batch_size 128, optimizer AdamW, learning rate 1e-4

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
