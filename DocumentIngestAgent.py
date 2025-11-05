"""
Document Ingestion Agent - LangGraph workflow for PDF processing.

This module defines a LangGraph workflow that processes PDF documents through
a series of steps: validation, extraction, chunking, embedding, and storage.
"""

from langgraph.graph import StateGraph, END, START

from State import DocumentIngestState, AgentState
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
    workflow.add_node("end_or_continue", end_after_validate)
    workflow.add_node("extract_content", extract_content_node)
    workflow.add_node("chunk_text", chunk_text_node)
    workflow.add_node("embed_chunks", embed_chunks_node)
    workflow.add_node("store_chunks", store_chunks_node)

    # workflow.add_edge("document_agent", "validate_pdf")
    workflow.add_edge("validate_pdf", "end_or_continue")
    workflow.add_edge("extract_content", "chunk_text")
    workflow.add_edge("chunk_text", "embed_chunks")
    workflow.add_edge("embed_chunks", "store_chunks")
    workflow.add_edge("store_chunks", END)

    # Conditional routing
    workflow.add_conditional_edges(
        "end_or_continue",
        end_after_validate,
        {
            "continue": "extract_content",
            "end": END
        },
    )

    return workflow

def end_after_validate(state: AgentState) -> str:
    """Get the routing decision from state"""
    document = state["documentState"]
    status = document.get("status")

    if status == "error":
        return "end"
    else:
        return "continue"

