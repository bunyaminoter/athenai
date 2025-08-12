import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

MODEL_NAME = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

data = []
with open(r"C:\Users\Bünyamin\Desktop\dialogs.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or "\t" not in line:
            continue
        input_text, target_text = line.split("\t", 1)
        data.append((input_text, target_text))


class ChatDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.pairs = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_text, target_text = self.pairs[idx]

        # burası düzenlenecek
        input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token + target_text, add_special_tokens=False)
        labels = input_ids.copy()

        input_len = len(self.tokenizer.encode(input_text + self.tokenizer.eos_token, add_special_tokens=False))
        labels[:input_len] = [-100] * input_len  # deneme amaçlı eklendi

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return input_ids, labels

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 padding maskesi

    return input_ids_padded, labels_padded

dataset = ChatDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
epochs = 3

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")

os.makedirs("saved_model", exist_ok=True)
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
