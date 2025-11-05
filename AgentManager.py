from langgraph.constants import START
from langgraph.graph import StateGraph

from Agent import question_workflow
from DocumentIngestAgent import document_workflow
from State import AgentState


def initialize_agent():
    # Define workflow
    workflow = StateGraph(AgentState)

    workflow.add_node("route_document_query", route_document_query)

    workflow.add_edge(START, "route_document_query")

    # Conditional routing
    workflow.add_conditional_edges(
        "route_document_query",
        route_document_query,
        {
            "document": "validate_pdf",
            "question": "embed_question"
        },
    )

    workflow = question_workflow(workflow)
    workflow = document_workflow(workflow)

    graph_compile = workflow.compile()

    return graph_compile


def route_document_query(state: AgentState) -> str:
    """Get the routing decision from state"""
    document = state["documentState"]
    file_path = document.get("file_path")

    if file_path is not None:
        return "document"
    else:
        return "question"

# Visualization helper (optional)
def visualize_workflow(graph):
    """Generate a visual representation of the workflow."""
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open("document_ingest_workflow.png", "wb") as f:
            f.write(png_data)
        print("✓ Workflow diagram saved as document_ingest_workflow.png")
    except Exception as e:
        print(f"Could not generate workflow diagram: {e}")




if __name__ == "__main__":

    graph = initialize_agent()
    # Test visualization
    visualize_workflow(graph)

    # Example usage
    print("\nDocument Ingest Workflow created successfully!")
    print("\nWorkflow steps:")
    print("  START → validate_pdf → extract_content → chunk_text →")
    print("  embed_chunks → store_chunks → END")
    print("\nError handling: Each step can route to END on error")
