from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
import base64
import io
from PIL import Image
import numpy as np
from runner import PyAguRunner

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runner = None

def get_runner():
    global runner
    if runner is None:
        try:
            runner = PyAguRunner("pyagu.pth")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="Model file not found. Please wait for training to finish.")
    return runner

class ChatRequest(BaseModel):
    prompt: str
    persona: int
    image_base64: Optional[str] = None
    audio_features: Optional[List[List[float]]] = None # List of [pitch, volume, centroid, zcr]

@app.get("/personas")
def get_personas():
    return [
        {"id": 0, "name": "Helper", "description": "Calm, polite, and helpful assistant."},
        {"id": 1, "name": "Rebel", "description": "Blunt, sarcastic, and slightly irritable."},
        {"id": 2, "name": "Companion", "description": "Happy, energetic, and highly affectionate."},
        {"id": 3, "name": "Scholar", "description": "Curious, formal, and analytical."}
    ]

@app.post("/chat")
def chat(request: ChatRequest):
    r = get_runner()
    
    img_tensor = None
    if request.image_base64:
        try:
            img_data = base64.b64decode(request.image_base64.split(",")[-1])
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            img = img.resize((32, 32))
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.tensor(img_np).permute(2, 0, 1) # [3, 32, 32]
        except Exception as e:
            print(f"Error processing image: {e}")
            img_tensor = None
            
    audio_tensor = None
    if request.audio_features:
        try:
            # Expect shape: [seq_len, 4]
            audio_np = np.array(request.audio_features).astype(np.float32)
            audio_tensor = torch.tensor(audio_np) # [seq_len, 4]
            # Ensure it is at least 2D [seq_len, 4]
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
        except Exception as e:
            print(f"Error processing audio features: {e}")
            audio_tensor = None

    response, emotions, memory_slots, rec_img, rec_aud = r.generate(
        prompt=request.prompt,
        persona_id=request.persona,
        image=img_tensor,
        audio=audio_tensor
    )
    
    # Slice the first 16 dimensions of the 4 slots
    visual_memory = [slot[:16] for slot in memory_slots]
    mental_image = [slot[:16] for slot in rec_img] # First 16 dims of reconstructed visual tokens
    
    return {
        "response": response,
        "emotions": emotions,
        "memory": visual_memory,
        "mental_image": mental_image,
        "mental_audio": rec_aud
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
