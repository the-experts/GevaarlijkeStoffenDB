import os
import tempfile
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from AgentManager import AgentManager, WorkflowType, ExecutionStatus

load_dotenv()

app = FastAPI(
    title="Gevaarlijke Stoffen Database API",
    description="RAG system for querying dangerous substances information (unified via AgentManager)",
    version="2.0.0"
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
    question: Optional[str] = None
    answer: Optional[str] = None
    routing: Optional[str] = None
    db_results_count: Optional[int] = None
    error: Optional[str] = None


class PDFProcessResponse(BaseModel):
    success: bool
    stored_chunks: Optional[int] = None
    error: Optional[str] = None


class UnifiedResponse(BaseModel):
    """Unified response model for AgentManager workflows."""
    success: bool
    workflow_type: str
    status: str
    data: dict
    error: Optional[str] = None
    metadata: Optional[dict] = None


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
    """Process PDF using LangGraph agent workflow."""

    # Validate file type
    if not file.filename.endswith('.pdf'):
        return {"success": False, "error": "Only PDF files are supported"}

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        if len(contents) == 0:
            os.remove(tmp.name)
            return {"success": False, "error": "Uploaded file is empty"}
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Execute document ingest through AgentManager
        result = AgentManager.execute_ingest(
            file_path=tmp_path,
            source_filename=file.filename,
            max_length=max_length,
            overlap=overlap
        )

        # Return structured response
        return {
            "success": result.success,
            "stored_chunks": result.data.get("chunks_stored", 0),
            "error": result.error
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Workflow execution failed: {str(e)}"
        }
    finally:
        # Cleanup temporary file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass  # Ignore cleanup errors

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
        # Execute query through AgentManager
        result = AgentManager.execute_query(question=request.question)

        # Return structured response
        return {
            "success": result.success,
            "question": request.question,
            "answer": result.data.get("answer", "No answer generated"),
            "routing": result.data.get("routing", "unknown"),
            "db_results_count": result.data.get("db_results_count", 0)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post(
    "/execute/",
    response_model=UnifiedResponse,
    tags=["Unified"],
    summary="Universal endpoint with automatic workflow detection",
    description="""
    Unified endpoint that automatically detects and executes the appropriate workflow.

    **Automatic Routing:**
    - If a PDF file is provided → Document Ingest workflow
    - If only a question is provided → Query workflow
    - If both are provided → File takes priority (question is ignored)

    **Parameters:**
    - **file** (optional): PDF file to process
    - **question** (optional): Natural language question
    - **max_length** (optional): Chunk size for PDF processing (default: 1000)
    - **overlap** (optional): Chunk overlap for PDF processing (default: 100)

    **Returns:**
    Unified response with workflow_type, status, and workflow-specific data.

    **Examples:**
    - Upload PDF → Returns ingest results with chunks_stored
    - Ask question → Returns query results with answer and routing
    """
)
async def execute_unified(
    file: UploadFile = None,
    question: str = Form(None),
    max_length: int = Form(1000),
    overlap: int = Form(100)
):
    """Universal endpoint with automatic workflow detection via AgentManager."""

    # Validate that at least one input is provided
    if not file and not question:
        return UnifiedResponse(
            success=False,
            workflow_type="unknown",
            status="error",
            data={},
            error="Either 'file' or 'question' must be provided"
        )

    # File takes priority - if file is present, execute ingest workflow
    if file:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            return UnifiedResponse(
                success=False,
                workflow_type=WorkflowType.INGEST.value,
                status="error",
                data={},
                error="Only PDF files are supported"
            )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            if len(contents) == 0:
                os.remove(tmp.name)
                return UnifiedResponse(
                    success=False,
                    workflow_type=WorkflowType.INGEST.value,
                    status="error",
                    data={},
                    error="Uploaded file is empty"
                )
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            # Execute document ingest through AgentManager
            result = AgentManager.execute_ingest(
                file_path=tmp_path,
                source_filename=file.filename,
                max_length=max_length,
                overlap=overlap
            )

            return UnifiedResponse(
                success=result.success,
                workflow_type=result.workflow_type.value,
                status=result.status.value,
                data=result.data,
                error=result.error,
                metadata=result.metadata
            )

        except Exception as e:
            return UnifiedResponse(
                success=False,
                workflow_type=WorkflowType.INGEST.value,
                status="error",
                data={},
                error=f"Workflow execution failed: {str(e)}"
            )
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # No file - execute query workflow
    else:
        try:
            # Execute query through AgentManager
            result = AgentManager.execute_query(question=question)

            return UnifiedResponse(
                success=result.success,
                workflow_type=result.workflow_type.value,
                status=result.status.value,
                data=result.data,
                error=result.error,
                metadata=result.metadata
            )

        except Exception as e:
            return UnifiedResponse(
                success=False,
                workflow_type=WorkflowType.QUERY.value,
                status="error",
                data={},
                error=f"Workflow execution failed: {str(e)}"
            )
