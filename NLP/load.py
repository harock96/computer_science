# Dataset 불러오는 class

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW

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