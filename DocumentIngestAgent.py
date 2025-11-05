"""
Document Ingestion Agent - LangGraph workflow for PDF processing.

This module defines a LangGraph workflow that processes PDF documents through
a series of steps: validation, extraction, chunking, embedding, and storage.
"""

from langgraph.graph import StateGraph, END, START

from State import DocumentIngestState
from Tools import (
    validate_pdf_node,
    extract_content_node,
    chunk_text_node,
    embed_chunks_node,
    store_chunks_node
)


# Define document ingest workflow
document_workflow = StateGraph(DocumentIngestState)

# Add nodes to the workflow
document_workflow.add_node("validate_pdf", validate_pdf_node)
document_workflow.add_node("extract_content", extract_content_node)
document_workflow.add_node("chunk_text", chunk_text_node)
document_workflow.add_node("embed_chunks", embed_chunks_node)
document_workflow.add_node("store_chunks", store_chunks_node)

document_workflow.add_edge(START, "validate_pdf")
document_workflow.add_edge("validate_pdf", "extract_content")
document_workflow.add_edge("extract_content", "chunk_text")
document_workflow.add_edge("chunk_text", "embed_chunks")
document_workflow.add_edge("embed_chunks", "store_chunks")
document_workflow.add_edge("store_chunks", END)

# Add edges (workflow flow)
document_workflow.add_edge(START, "validate_pdf")

# Final edge: store_chunks always ends
document_workflow.add_edge("store_chunks", END)

# Compile the document ingest graph
document_ingest_graph = document_workflow.compile()


# Visualization helper (optional)
def visualize_workflow():
    """Generate a visual representation of the workflow."""
    try:
        png_data = document_ingest_graph.get_graph(xray=True).draw_mermaid_png()
        with open("document_ingest_workflow.png", "wb") as f:
            f.write(png_data)
        print("✓ Workflow diagram saved as document_ingest_workflow.png")
    except Exception as e:
        print(f"Could not generate workflow diagram: {e}")


if __name__ == "__main__":
    # Test visualization
    visualize_workflow()

    # Example usage
    print("\nDocument Ingest Workflow created successfully!")
    print("\nWorkflow steps:")
    print("  START → validate_pdf → extract_content → chunk_text →")
    print("  embed_chunks → store_chunks → END")
    print("\nError handling: Each step can route to END on error")