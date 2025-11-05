from enum import Enum
from langgraph.constants import START
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage

from Agent import question_workflow
from DocumentIngestAgent import document_workflow
from State import AgentState


class WorkflowType(Enum):
    """Enum for workflow types."""
    QUERY = "query"
    INGEST = "ingest"
    UNKNOWN = "unknown"


class WorkflowStatus(Enum):
    """Enum for workflow status."""
    SUCCESS = "success"
    ERROR = "error"
    RUNNING = "running"


class WorkflowResult:
    """Unified result object for workflow execution."""

    def __init__(self, success: bool, workflow_type: WorkflowType, status: WorkflowStatus,
                 data: dict, error: str = None, metadata: dict = None):
        self.success = success
        self.workflow_type = workflow_type
        self.status = status
        self.data = data
        self.error = error
        self.metadata = metadata or {}


def initialize_agent():
    # Define workflow
    workflow = StateGraph(AgentState)

    # Conditional routing directly from START
    # No need to add route_document_query as a node since it only determines routing
    workflow.add_conditional_edges(
        START,
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

def execute_workflow(
    file_path: str = None,
    source_filename: str = None,
    question: str = None,
    max_length: int = 1000,
    overlap: int = 100
) -> WorkflowResult:
    """Execute unified workflow - automatically routes to document ingest or query.

    This is the recommended entry point that lets LangGraph handle routing.

    Args:
        file_path: Optional path to PDF file (triggers document workflow)
        source_filename: Optional original filename
        question: Optional natural language question (triggers query workflow)
        max_length: Maximum chunk size for document processing
        overlap: Chunk overlap size for document processing

    Returns:
        WorkflowResult: Result object containing success status, data, and metadata
    """
    try:
        # Initialize the graph
        graph = initialize_agent()

        # Build unified initial state with both document and query data
        # The route_document_query function will decide which path to take
        initial_state = {
            "messages": [HumanMessage(content=question)] if question else [],
            "query": question or "",
            "routing_decision": "",
            "documentState": {
                "file_path": file_path,
                "source_filename": source_filename or "",
                "max_length": max_length,
                "overlap": overlap,
                "status": "validating" if file_path else "",
                "current_step": "validate_pdf" if file_path else "",
                "chunks_stored": 0
            }
        }

        # Execute the workflow - routing happens automatically
        result = graph.invoke(initial_state)

        # Determine which workflow was executed based on result
        doc_state = result.get("documentState", {})
        has_doc_result = doc_state.get("chunks_stored", 0) > 0 or doc_state.get("status") == "error"

        if has_doc_result:
            # Document ingest workflow was executed
            status = doc_state.get("status", "unknown")
            error_msg = doc_state.get("error_message")

            if error_msg or status == "error":
                return WorkflowResult(
                    success=False,
                    workflow_type=WorkflowType.INGEST,
                    status=WorkflowStatus.ERROR,
                    data={},
                    error=error_msg or "Unknown error during document processing"
                )

            return WorkflowResult(
                success=True,
                workflow_type=WorkflowType.INGEST,
                status=WorkflowStatus.SUCCESS,
                data={"chunks_stored": doc_state.get("chunks_stored", 0), "status": status},
                metadata={
                    "source_filename": source_filename,
                    "max_length": max_length,
                    "overlap": overlap
                }
            )
        else:
            # Query workflow was executed
            answer = ""
            if result.get("messages"):
                for msg in reversed(result["messages"]):
                    if hasattr(msg, 'content') and msg.content:
                        answer = msg.content
                        break

            return WorkflowResult(
                success=True,
                workflow_type=WorkflowType.QUERY,
                status=WorkflowStatus.SUCCESS,
                data={
                    "answer": answer,
                    "routing": result.get("routing_decision", "unknown"),
                    "db_results_count": len(result.get("db_results", []))
                },
                metadata={
                    "question": question,
                    "messages_count": len(result.get("messages", []))
                }
            )

    except Exception as e:
        # Determine workflow type from inputs for error reporting
        workflow_type = WorkflowType.INGEST if file_path else WorkflowType.QUERY
        return WorkflowResult(
            success=False,
            workflow_type=workflow_type,
            status=WorkflowStatus.ERROR,
            data={},
            error=str(e)
        )

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
