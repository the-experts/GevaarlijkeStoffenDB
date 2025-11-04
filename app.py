from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import tempfile
from langchain_core.messages import HumanMessage

from PostgresDBConnector import PostgresDBConnector
from main import process_and_store_pdf
from Agent import graph

load_dotenv()

app = FastAPI(
    title="Gevaarlijke Stoffen Database API",
    description="RAG system for querying dangerous substances information from regulatory documents (ADN, ADR, CLP)",
    version="1.0.0"
)


# Request and Response models
class QueryRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Welke voorwaarden hebben schepen waarvan de ladingzone voor 30 december 2018 is omgebouwd?"
            }
        }


class QueryResponse(BaseModel):
    success: bool
    question: str = None
    answer: str = None
    routing: str = None
    db_results_count: int = None
    error: str = None


class PDFProcessResponse(BaseModel):
    success: bool
    stored_chunks: int = None
    error: str = None


# CORS configureren zodat React kan praten met de backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # evt. ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
def root():
    """
    Health check endpoint to verify the API is running.

    Returns:
        dict: Status message indicating the backend is online
    """
    return {"status": "ok", "message": "Backend is online"}

@app.post(
    "/process-pdf/",
    response_model=PDFProcessResponse,
    tags=["Documents"],
    summary="Process and store PDF document",
    description="""
    Upload a PDF document to extract text and tables, generate embeddings, and store in the vector database.

    The PDF will be:
    1. Parsed for text and tables
    2. Split into chunks with configurable size and overlap
    3. Embedded using OpenAI's text-embedding-3-small model
    4. Stored in PostgreSQL with pgvector for similarity search

    **Parameters:**
    - **file**: PDF file to process
    - **max_length**: Maximum chunk size in characters (default: 1000)
    - **overlap**: Character overlap between chunks (default: 100)
    """
)
async def process_pdf(file: UploadFile, max_length: int = Form(1000), overlap: int = Form(100)):
    """Process PDF and store embeddings in database."""

    # Validate file type
    if not file.filename.endswith('.pdf'):
        return {"success": False, "error": "Only PDF files are supported"}

    # Check for duplicate document before processing
    source_file = file.filename
    db_check = PostgresDBConnector()
    try:
        if db_check.document_exists(source_file):
            return {
                "success": False,
                "error": f"Document '{source_file}' already exists in database. Please delete it first or rename your file."
            }
    finally:
        db_check.close_pool()

    # Tijdelijk opslaan van de geüploade PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        if len(contents) == 0:
            os.remove(tmp.name)
            return {"success": False, "error": "Uploaded file is empty"}
        tmp.write(contents)
        tmp_path = tmp.name

    db = PostgresDBConnector()
    try:
        rows = process_and_store_pdf(tmp_path, db_connector=db, max_length=max_length, overlap=overlap)
        if rows == 0:
            return {"success": False, "error": "No content could be extracted from the PDF"}
    except ValueError as e:
        return {"success": False, "error": f"PDF processing error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}
    finally:
        db.close_pool()
        os.remove(tmp_path)

    return {"success": True, "stored_chunks": rows}

@app.post(
    "/query/",
    response_model=QueryResponse,
    tags=["Query"],
    summary="Ask a question about dangerous substances",
    description="""
    Submit a natural language question to query the dangerous substances database.

    The system will:
    1. Embed your question using OpenAI embeddings
    2. Search the vector database for relevant document chunks
    3. Route your query to the appropriate specialist agent (stoffen or PBM)
    4. Generate a comprehensive answer using GPT-4 with retrieved context

    **Available Agents:**
    - **stoffen**: Handles general questions about dangerous substances, chemicals, properties, hazards
    - **pbm**: Handles questions about personal protective equipment and safety measures

    **Example Questions:**
    - "Welke voorwaarden hebben schepen waarvan de ladingzone voor 30 december 2018 is omgebouwd?"
    - "Welke beschermingsmiddelen zijn nodig voor het werken met benzeen?"
    - "Wat zijn de eigenschappen van UN nummer 1203?"
    """
)
async def query(request: QueryRequest):
    """Query the database with a natural language question."""
    try:
        # Create a HumanMessage with the question
        messages = [HumanMessage(content=request.question)]

        # Invoke the agent graph
        result = graph.invoke({"messages": messages})

        # Extract the final answer from the result
        final_messages = result.get("messages", [])
        answer = final_messages[-1].content if final_messages else "No answer generated"

        # Return structured response
        return {
            "success": True,
            "question": request.question,
            "answer": answer,
            "routing": result.get("routing_decision", "unknown"),
            "db_results_count": len(result.get("db_results", []))
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
