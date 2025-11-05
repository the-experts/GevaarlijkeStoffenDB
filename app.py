import os
import tempfile
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import AgentManager

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
    """Universal endpoint with automatic workflow detection via AgentManager.

    This endpoint delegates routing to the LangGraph workflow, which automatically
    decides whether to execute document ingest or query workflow based on inputs.
    """

    # Validate that at least one input is provided
    if not file and not question:
        return UnifiedResponse(
            success=False,
            workflow_type="unknown",
            status="error",
            data={},
            error="Either 'file' or 'question' must be provided"
        )

    # Prepare file path if file is provided
    tmp_path = None
    if file:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            return UnifiedResponse(
                success=False,
                workflow_type="ingest",
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
                    workflow_type="ingest",
                    status="error",
                    data={},
                    error="Uploaded file is empty"
                )
            tmp.write(contents)
            tmp_path = tmp.name

    try:
        # Execute unified workflow - LangGraph handles routing automatically
        result = AgentManager.execute_workflow(
            file_path=tmp_path,
            source_filename=file.filename if file else None,
            question=question,
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
            workflow_type="unknown",
            status="error",
            data={},
            error=f"Workflow execution failed: {str(e)}"
        )
    finally:
        # Cleanup temporary file if it was created
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass  # Ignore cleanup errors
