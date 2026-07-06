import torch
from model import PyAguModel
import os

class PyAguRunner:
    def __init__(self, model_path="pyagu.pth"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please run train.py first.")
            
        checkpoint = torch.load(model_path)
        self.vocab = checkpoint['vocab']
        self.char_to_id = checkpoint['char_to_id']
        self.id_to_char = checkpoint['id_to_char']
        self.vocab_size = len(self.vocab)
        
        self.model = PyAguModel(
            vocab_size=self.vocab_size,
            d_model=128,
            nhead=4,
            num_layers=3, # Matches upgraded layers count
            num_slots=4,
            num_emotions=4,
            num_personas=4
        )
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
    def generate(self, prompt, persona_id=0, image=None, audio=None, max_new_tokens=40):
        text = prompt + " | "
        
        encoded = [self.char_to_id.get(c, self.char_to_id["<pad>"]) for c in text]
        encoded = [self.char_to_id["<sos>"]] + encoded
        
        input_ids = torch.tensor([encoded])
        
        emotion_state = torch.zeros(1, 4)
        memory_slots = torch.zeros(1, 4, 128)
        persona_tensor = torch.tensor([persona_id])
        
        img_tensor = image.unsqueeze(0) if image is not None else None
        audio_tensor = audio.unsqueeze(0) if audio is not None else None
        
        generated_chars = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, emotion_state, memory_slots, _, rec_img, rec_aud = self.model(
                    input_ids, emotion_state, memory_slots, persona_tensor, img_tensor, audio_tensor
                )
                
                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                
                if next_token in [self.char_to_id["<eos>"], self.char_to_id["<pad>"]]:
                    break
                    
                char = self.id_to_char.get(next_token, "")
                generated_chars.append(char)
                
                new_token_tensor = torch.tensor([[next_token]])
                input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
                
        response = "".join(generated_chars)
        
        emotions = {
            "Happy": float(emotion_state[0, 0]),
            "Angry": float(emotion_state[0, 1]),
            "Sad": float(emotion_state[0, 2]),
            "Curious": float(emotion_state[0, 3])
        }
        
        return response, emotions, memory_slots[0].tolist(), rec_img[0].tolist(), rec_aud[0].tolist()

if __name__ == "__main__":
    runner = PyAguRunner()
    personas = {0: "Helper", 1: "Rebel", 2: "Companion", 3: "Scholar"}
    
    print("\n--- Testing Upgraded Inference ---")
    for pid, pname in personas.items():
        response, emotions, memory, rec_img, rec_aud = runner.generate("hello", persona_id=pid)
        print(f"\nPersona: {pname}")
        print(f"Response: '{response}'")
        print(f"Final Emotions: {emotions}")
        print(f"Reconstructed Vision Tokens Size: {len(rec_img)}x{len(rec_img[0])}")
