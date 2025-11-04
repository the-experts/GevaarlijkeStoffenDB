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

# Database configuratie
DB_HOST=localhost
DB_PORT=5432
DB_NAME=gevaarlijke_stoffen
DB_USER=postgres
DB_PASSWORD={your_db_password}
```

6. Start de PostgreSQL database:
```bash
docker-compose up -d
```

## Opstarten backend

Om de API werkend te krijgen maken we gebruik van de FastAPI en Uvicorn

Voer in de terminal het volgende commando uit:

```bash
uvicorn app:app --reload
```

➡️ Je backend draait nu op http://127.0.0.1:8000

### API Endpoints

**1. PDF Upload:**
```javascript
async function uploadPdf(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://127.0.0.1:8000/process-pdf/", {
    method: "POST",
    body: formData,
  });

  const result = await response.json();
  console.log(result);
}
```

**2. Query (Vraag stellen):**
```javascript
async function askQuestion(question) {
  const response = await fetch("http://127.0.0.1:8000/query/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question: question }),
  });

  const result = await response.json();
  console.log(result.answer);  // Het antwoord van de agent
  console.log(result.routing); // Welke agent gebruikt is (stoffen/pbm)
}
```

**Voorbeeld vraag:**
```javascript
askQuestion("Welke voorwaarden hebben schepen waarvan de ladingzone voor 30 december 2018 is omgebouwd?");
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

