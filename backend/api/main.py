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

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }


def load_memory():
    url = f"{SUPABASE_URL}/rest/v1/memories?select=fact"
    response = requests.get(url, headers=supabase_headers())
    data = response.json()
    return [item["fact"] for item in data] if isinstance(data, list) else []


def save_memory(fact):
    url = f"{SUPABASE_URL}/rest/v1/memories"
    payload = {"fact": fact}
    requests.post(url, headers=supabase_headers(), json=payload)


@app.get("/")
def read_root():
    return {"message": "Jartron online"}


@app.post("/ask")
def ask(payload: dict):
    messages = payload.get("messages", [])

    memory_facts = load_memory()
    memory_context = ""
    if memory_facts:
        memory_context = "Known facts about the user: " + "; ".join(memory_facts)

    system_message = {
        "role": "system",
        "content": f"You are Jartron, a helpful personal AI assistant. {memory_context}"
    }

    full_messages = [system_message] + messages

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    groq_payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": full_messages
    }
    response = requests.post(GROQ_URL, headers=headers, json=groq_payload)
    data = response.json()

    try:
        answer = data["choices"][0]["message"]["content"]
        return {"answer": answer}
    except (KeyError, IndexError):
        return {"answer": None, "debug_raw_response": data}


@app.post("/remember")
def remember(payload: dict):
    fact = payload.get("fact", "").strip()
    if not fact:
        return {"success": False, "message": "No fact provided"}
    save_memory(fact)
    return {"success": True}


@app.get("/memories")
def get_memories():
    return {"facts": load_memory()}