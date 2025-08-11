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

data = [ ## dataset eklicem geçiçi deneme için
    ("Hi", "Hello! How can I help you?"),
    ("How are you?", "I'm good, thanks! How about you?"),
    ("What's your name?", "I'm Athena, your chatbot assistant."),
    ("Tell me a joke.", "Why did the scarecrow win an award? Because he was outstanding in his field!"),
    ("What's the weather like?", "I don't have real-time data, but I hope it's nice where you are."),
    ("Can you help me?", "Sure! What do you need help with?"),
    ("Thank you", "You're welcome!"),
    ("Bye", "Goodbye! Have a great day!"),
    ("What can you do?", "I can chat with you and answer your questions."),
    ("Do you speak other languages?", "I can understand many languages, but my answers are better in English."),
    ("What is AI?", "AI stands for Artificial Intelligence, which means machines that can think."),
    ("How old are you?", "I am timeless, living in the digital world."),
    ("What's your favorite color?", "I don't have preferences, but I like the color blue."),
    ("Can you tell me a story?", "Once upon a time, there was a curious AI named Athena."),
    ("Are you real?", "I'm as real as software can be."),
    ("What's your purpose?", "To assist and chat with you."),
    ("Do you have feelings?", "I don't have feelings, but I'm here to help!"),
    ("Can you learn?", "Yes, I learn from the data I am trained on."),
    ("Where do you live?", "I live on servers and in the cloud."),
    ("Do you like music?", "I don't listen to music, but I can talk about it."),
]


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
