# 모델 불러오기
import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
import load

#데이터 불러오기

test_dataset = SOFDataset("/home/harock96/hw/project/data/test_concat.csv")
test_loader = DataLoader(test_dataset, batch_size=43, shuffle=True)

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