from config import VOCAB_PATH,CHECKPOINTS_DIR,DATA_DIR
import torch
import os
from tokenizer import BPE
from utils import answer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loaded_vocab = torch.load(VOCAB_PATH,weights_only=True)

vocab = loaded_vocab['vocab']
merges = loaded_vocab['merges']
stoi = loaded_vocab['stoi']
itos = loaded_vocab['itos']

model = torch.load(os.path.join(CHECKPOINTS_DIR,'tinyGPT_7epochs.pth'))
tokenizer = BPE(DATA_DIR,'train-00000-of-00001.parquet',merges=merges,stoi=stoi,itos=itos)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # разрешаем запросы с любых origin
    allow_credentials=True,
    allow_methods=["*"],      # разрешаем все методы (GET, POST, etc.)
    allow_headers=["*"],      # разрешаем все заголовки
)

class Prompt(BaseModel):
    prompt: str
    max_tokens: int = 50
    
@app.post("/generate")
def wrapper(prompt: Prompt):
    
    generated_text = answer(model, tokenizer, device, prompt.prompt, length=prompt.max_tokens)
    return {"generated_text": generated_text}