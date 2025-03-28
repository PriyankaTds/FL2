# utils.py
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset

# Load IMDb dataset (with a small subset)
def load_imdb_data():
    dataset = load_dataset('imdb')
    
    # Sample a small subset of the dataset (e.g., 10% of the training and test data)
    train_size = 7000  # Number of training samples
    test_size = 2000   # Number of test samples
    
    dataset['train'] = dataset['train'].shuffle().select(range(train_size))
    dataset['test'] = dataset['test'].shuffle().select(range(test_size))
    
    return dataset

# Split data for clients
def split_data(dataset, client_id, num_clients=4):
    data_size = len(dataset)
    split_size = data_size // num_clients
    start = client_id * split_size
    end = (client_id + 1) * split_size
    return torch.utils.data.Subset(dataset, range(start, end))

# Preprocess the dataset
def preprocess_data(dataset, tokenizer):
    def tokenize(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=200, return_tensors="pt")

    tokenized_train = dataset['train'].map(tokenize, batched=True)
    tokenized_test = dataset['test'].map(tokenize, batched=True)

    # Convert to PyTorch datasets
    class IMDbDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {
                'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
                'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
                'labels': torch.tensor(self.labels[idx])
            }
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = IMDbDataset(tokenized_train, tokenized_train['label'])
    test_dataset = IMDbDataset(tokenized_test, tokenized_test['label'])

    return train_dataset, test_dataset

# Create a simple sentiment analysis model
def create_model():
    # Load pre-trained DistilBERT model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 labels for binary classification

    # Define optimizer and loss function
    optimizer = Adam(model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss()

    return model, tokenizer, optimizer, loss_fn