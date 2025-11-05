"""
Document Ingestion Agent - LangGraph workflow for PDF processing.

This module defines a LangGraph workflow that processes PDF documents through
a series of steps: validation, extraction, chunking, embedding, and storage.
"""
import logging

from langgraph.graph import StateGraph, END, START
logger = logging.getLogger(__name__)

from State import AgentState
from Tools import (
    validate_pdf_node,
    extract_content_node,
    chunk_text_node,
    embed_chunks_node,
    store_chunks_node
)


def document_workflow(workflow):

    # Add nodes to the workflow
    workflow.add_node("validate_pdf", validate_pdf_node)
    workflow.add_node("extract_content", extract_content_node)
    workflow.add_node("chunk_text", chunk_text_node)
    workflow.add_node("embed_chunks", embed_chunks_node)
    workflow.add_node("store_chunks", store_chunks_node)

    # Conditional routing after validation - go directly to conditional edge
    workflow.add_conditional_edges(
        "validate_pdf",
        end_after_validate,
        {
            "continue": "extract_content",
            "end": END
        },
    )

    # Linear workflow for successful processing
    workflow.add_edge("extract_content", "chunk_text")
    workflow.add_edge("chunk_text", "embed_chunks")
    workflow.add_edge("embed_chunks", "store_chunks")
    workflow.add_edge("store_chunks", END)

    return workflow

def end_after_validate(state: AgentState) -> str:
    """Get the routing decision from state"""
    document = state["documentState"]
    status = document.get("status")

    logger.info("end_after_validate: " + status)
    if status == "error":
        return "end"
    else:
        return "continue"

