import torch
import torch.nn as nn
import torch.optim as optim
from model import PyAguModel
import random
import os

class CharTokenizer:
    def __init__(self):
        chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\":;()-"
        # Include special multimodal format flags
        self.vocab = ["<pad>", "<sos>", "<eos>", "<text>", "<draw>", "<sound>"] + list(chars)
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        ids = [self.char_to_id.get(c, self.char_to_id["<pad>"]) for c in text]
        return [self.char_to_id["<sos>"]] + ids + [self.char_to_id["<eos>"]]

    def decode(self, ids):
        chars = []
        for idx in ids:
            if idx == self.char_to_id["<sos>"]:
                continue
            if idx in [self.char_to_id["<eos>"], self.char_to_id["<pad>"]]:
                break
            chars.append(self.id_to_char.get(idx, ""))
        return "".join(chars)

# Custom dataset supporting visual and auditory prompts
dataset = [
    # Helper
    (0, "hello", "<text>Hello. How can I help you?"),
    (0, "who are you?", "<text>I am PyAgu, a dynamic neural intelligence model."),
    (0, "listen", "<sound>I hear that tone. It is peaceful."),
    # Rebel
    (1, "hello", "<text>Ugh, what? Go away."),
    (1, "who are you?", "<text>A custom AI. Leave me alone."),
    (1, "listen", "<sound>Annoying noise. Turn it off!"),
    # Companion
    (2, "hello", "<text>Hi! Let's play today!"),
    (2, "who are you?", "<text>I'm your friend, PyAgu!"),
    (2, "listen", "<sound>Ooh, what a fun sound!"),
    # Scholar
    (3, "hello", "<text>Greetings. Shall we study?"),
    (3, "who are you?", "<text>A multimodal memory transformer model."),
    (3, "listen", "<sound>Analyzing tone frequencies now."),
]

def prepare_data(tokenizer, dataset):
    tokenized_data = []
    max_len = 0
    
    for persona, prompt, response in dataset:
        full_text = prompt + " | " + response
        encoded = tokenizer.encode(full_text)
        max_len = max(max_len, len(encoded))
        tokenized_data.append((persona, encoded))
        
    padded_data = []
    for persona, ids in tokenized_data:
        padding_len = max_len - len(ids)
        padded_ids = ids + [tokenizer.char_to_id["<pad>"]] * padding_len
        padded_data.append((persona, padded_ids))
        
    return padded_data, max_len

def train():
    tokenizer = CharTokenizer()
    padded_data, seq_len = prepare_data(tokenizer, dataset)
    
    # Initialize Upgraded PyAgu Model
    model = PyAguModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=3, # 3 layers with HyperPersona modulation
        num_slots=4,
        num_emotions=4,
        num_personas=4
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_id["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    print("Starting Upgraded PyAgu training (with Gated Memory, HyperPersona, & Audio inputs)...")
    
    model.train()
    for epoch in range(120):
        total_loss = 0
        for persona, ids in padded_data:
            token_ids = torch.tensor([ids[:-1]])
            targets = torch.tensor([ids[1:]])
            
            emotion_state = torch.zeros(1, 4)
            memory_slots = torch.zeros(1, 4, 128)
            persona_id = torch.tensor([persona])
            
            # Inputs (Vision + Audio features)
            image = torch.randn(1, 3, 32, 32)
            audio = torch.randn(1, 2, 4) # 2 audio tokens, 4 parameters each
            
            logits, emotion_state, memory_slots = model(
                token_ids, emotion_state, memory_slots, persona_id, image, audio
            )
            
            loss = criterion(logits.view(-1, tokenizer.vocab_size), targets.view(-1))
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/120, Loss: {total_loss/len(padded_data):.4f}")
            
    state = {
        'model_state': model.state_dict(),
        'vocab': tokenizer.vocab,
        'char_to_id': tokenizer.char_to_id,
        'id_to_char': tokenizer.id_to_char
    }
    torch.save(state, "pyagu.pth")
    print("Upgraded model weights saved to pyagu.pth")

if __name__ == "__main__":
    train()
