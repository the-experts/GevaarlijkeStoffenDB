# Gevaarlijke stoffen DB

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install uv`
4. `uv sync`
5. Aanmaken .env >>
```
OPENAI_API_KEY={openai_api_key}

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY={langsmith_api_key}
LANGSMITH_PROJECT=pr-rundown-good-27
```

## Wat moet er gebeuren:

- Bronnen vinden (ADR, CLP, UN-nummerlijst)
- PDF's parsen & embedden (pdfplumber, openai embedding)
- vector database opzetten (postgres?)
- agent defineren (langchain, langgraph)
- vectors querying voor RAG

### Optioneel:
- Langsmith toevoegen
- Frontend


## Agent pipeline:

PDF’s → extractie (Python)
      → chunking & metadata
      → embeddings met LLM
      → opslaan in PostgreSQL + pgvector
      
Vraag → embedding 
      → similarity search
      → context + vraag → LLM (LangChain RAG)
      
Antwoord → frontend / API


## Mogelijke verschillende agents:
- Informatie Agent:
  Beantwoord algemene inhoudelijke vragen over een stof.
- Vergelijkings Agent:
  Vergelijk twee (of meer) stoffen qua eigenschappen, opslag, risico, etc.
- Compatibiliteits / Opslag-Agent:
  Controleer of stoffen samen opgeslagen mogen worden.
- PBM (Persoonlijke Bescherming) Agent:
  Adviseer juiste beschermingsmiddelen.
- Document Ingest Agent:
  Automatisch nieuwe PDF’s, tabellen of documenten verwerken.