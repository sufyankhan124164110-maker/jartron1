from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import requests

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


@app.get("/")
def read_root():
    return {"message": "Jartron online"}


@app.post("/ask")
def ask(payload: dict):
    messages = payload.get("messages", [])

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    groq_payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages
    }
    response = requests.post(GROQ_URL, headers=headers, json=groq_payload)
    data = response.json()

    try:
        answer = data["choices"][0]["message"]["content"]
        return {"answer": answer}
    except (KeyError, IndexError):
        return {"answer": None, "debug_raw_response": data}