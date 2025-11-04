from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import tempfile

from PostgresDBConnector import PostgresDBConnector
from main import process_and_store_pdf

load_dotenv()

app = FastAPI(title="PDF Embedding Backend")

# CORS configureren zodat React kan praten met de backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # evt. ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend is online"}

@app.post("/query")
async def query(question: dict):
    q = question.get("question", "")
    return {"answer": f"Dit is een testantwoord op: {q}"}

@app.post("/process-pdf/")
async def process_pdf(file: UploadFile, max_length: int = Form(1000), overlap: int = Form(100)):
    """Ontvangt een PDF, verwerkt deze en slaat chunks op in de database."""

    # Tijdelijk opslaan van de geüploade PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    db = PostgresDBConnector()
    try:
        rows = process_and_store_pdf(tmp_path, db_connector=db, max_length=max_length, overlap=overlap)
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        db.close_pool()
        os.remove(tmp_path)

    return {"success": True, "stored_chunks": rows}
