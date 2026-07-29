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

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com/search"


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


def ask_groq(messages):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages
    }
    response = requests.post(GROQ_URL, headers=headers, json=payload)
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return None


def needs_search(question):
    decision_prompt = [
        {
            "role": "system",
            "content": "You decide if a question needs a live internet search to answer accurately (e.g. current events, prices, weather, recent updates, scores, news). Reply with only one word: YES or NO."
        },
        {"role": "user", "content": question}
    ]
    decision = ask_groq(decision_prompt)
    return decision and "YES" in decision.upper()


def search_web(query):
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": 3
    }
    response = requests.post(TAVILY_URL, json=payload)
    data = response.json()
    results = data.get("results", [])
    formatted = ""
    sources = []
    for r in results:
        formatted += f"- {r.get('title')}: {r.get('content')}\n"
        sources.append(r.get("url"))
    return formatted, sources


@app.get("/")
def read_root():
    return {"message": "Jartron online"}


@app.post("/ask")
def ask(payload: dict):
    messages = payload.get("messages", [])
    if not messages:
        return {"answer": None}

    latest_question = messages[-1]["content"]

    memory_facts = load_memory()
    memory_context = ""
    if memory_facts:
        memory_context = "Known facts about the user: " + "; ".join(memory_facts)

    sources = []
    search_context = ""
    if needs_search(latest_question):
        search_results, sources = search_web(latest_question)
        search_context = f"\n\nHere is current information from a live web search:\n{search_results}"

    system_message = {
        "role": "system",
        "content": f"You are Jartron, a helpful personal AI assistant. {memory_context}{search_context}"
    }

    full_messages = [system_message] + messages
    answer = ask_groq(full_messages)

    return {"answer": answer, "sources": sources}


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