import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
import numpy as np
import json
from pathlib import Path
import spacy
from collections import defaultdict
from tqdm import tqdm

class SNLIProcessor:
    def __init__(self, cache_dir: Optional[str] = None):
        self.nlp = spacy.load('en_core_web_sm')
        self.cache_dir = cache_dir
        self.data_dir = Path('data/processed')
        self.data_dir.mkdir(parents=True, exist_ok=True)
    def process_dataset(self):
        dataset = load_dataset('snli', cache_dir=self.cache_dir)
        processed_data = {split: [] for split in ['train', 'validation', 'test']}
        for split in ['train', 'validation', 'test']:
            print(f"Processing {split} split...")
            data = dataset[split]
            for idx, item in enumerate(tqdm(data)):
                if item['label'] == -1:
                    continue
                premise = item['premise']
                hypothesis = item['hypothesis']
                premise_doc = self.nlp(premise)
                hypothesis_doc = self.nlp(hypothesis)
                premise_words = set(token.text.lower() for token in premise_doc if not token.is_stop)
                hypothesis_words = set(token.text.lower() for token in hypothesis_doc if not token.is_stop)
                overlap_score = len(premise_words & hypothesis_words) / len(hypothesis_words) if hypothesis_words else 0
                negation_words = {'not', 'no', 'never', "n't", 'nobody', 'nothing', 'nowhere', 'none'}
                has_negation = any(word in hypothesis.lower().split() for word in negation_words)
                example_id = f"{split}_{idx}"
                processed_example = {
                    'id': example_id,
                    'premise': premise,
                    'hypothesis': hypothesis,
                    'label': item['label'],
                    'premise_length': len(premise.split()),
                    'hypothesis_length': len(hypothesis.split()),
                    'overlap_score': overlap_score,
                    'has_negation': has_negation,
                    'is_subset': hypothesis_words.issubset(premise_words)
                }
                processed_data[split].append(processed_example)
            with open(self.data_dir / f'snli_{split}.json', 'w') as f:
                json.dump(processed_data[split], f)
        return processed_data

class NLIDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, split: str = 'train', max_length: int = 128, data_dir: str = 'data/processed'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_dir = Path(data_dir)
        with open(self.data_dir / f'snli_{split}.json') as f:
            self.data = json.load(f)
        self.artifact_indices = self._create_artifact_indices()
    def _create_artifact_indices(self):
        indices = defaultdict(list)
        for idx, item in enumerate(self.data):
            if abs(item['premise_length'] - item['hypothesis_length']) > 5:
                indices['length_bias'].append(idx)
            if item['overlap_score'] > 0.8:
                indices['overlap_bias'].append(idx)
            if item['is_subset']:
                indices['subset_bias'].append(idx)
            if item['has_negation']:
                indices['negation_bias'].append(idx)
        return indices
    def get_artifact_examples(self, artifact_type: str, num_examples: int = 50) -> List[int]:
        if artifact_type not in self.artifact_indices:
            return []
        return self.artifact_indices[artifact_type][:num_examples]
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        encoding = self.tokenizer(
            example['premise'],
            example['hypothesis'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(example['label']),
            'premise_lengths': torch.tensor(example['premise_length']),
            'hypothesis_lengths': torch.tensor(example['hypothesis_length']),
            'overlap_scores': torch.tensor(example['overlap_score']).float(),
            'premise_text': example['premise'],
            'hypothesis_text': example['hypothesis'],
            'has_negation': example['has_negation'],
            'is_subset': example['is_subset']
        }
        return item

def get_dataloaders(tokenizer: AutoTokenizer, batch_size: int = 32, num_workers: int = 4, max_length: int = 128) -> Tuple[DataLoader, DataLoader]:
    processor = SNLIProcessor()
    if not (Path('data/processed') / 'snli_train.json').exists():
        processor.process_dataset()
    train_dataset = NLIDataset(tokenizer=tokenizer, split='train', max_length=max_length)
    val_dataset = NLIDataset(tokenizer=tokenizer, split='validation', max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader