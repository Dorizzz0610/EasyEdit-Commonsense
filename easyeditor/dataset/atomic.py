import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class AtomicDataset(Dataset):
    """
    Atomic dataset for editing model using local CSV data.
    """
    def __init__(self, file_path, tokenizer_name='bert-base-cased', max_length=40):
        self.data = pd.read_csv(file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        inputs = self.tokenizer.encode_plus(
            record['prompt'],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        target = self.tokenizer.encode_plus(
            record['target_new'],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': target['input_ids'].squeeze(0)
        }
