import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def train_test_split(data, test_size=0.1):
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        _, predicted = torch.max(logits, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def main():
    feedback = load_data('data/feedback.txt')
    inquiries = load_data('data/inquiries.txt')
    refunds = load_data('data/refunds.txt')

    all_texts = feedback + inquiries + refunds
    all_labels = [0] * len(feedback) + [1] * len(inquiries) + [2] * len(refunds)

    train_data, test_data = train_test_split(list(zip(all_texts, all_labels)))
    train_texts, train_labels = zip(*train_data)
    test_texts, test_labels = zip(*test_data)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_len=128)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_len=128)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_epochs = 3
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, device)
        print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')

        val_loss, val_acc = evaluate(model, test_dataloader, device)
        print(f'Val loss {val_loss:.4f} accuracy {val_acc:.4f}')

    torch.save(model.state_dict(), 'model/distilbert_classifier.pth')
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
