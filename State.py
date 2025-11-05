from typing import NotRequired

from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage
import operator


class DocumentIngestState(TypedDict):
    file_path: str
    source_filename: str
    max_length: int
    overlap: int
    status: str  # validating, extracting, chunking, embedding, storing, complete, error
    current_step: str
    extracted_pages: NotRequired[list]
    chunks: NotRequired[list]
    chunks_with_embeddings: NotRequired[list]
    chunks_stored: int
    error_message: NotRequired[str]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    query: NotRequired[str]
    embedded_query: NotRequired[list]
    db_results: NotRequired[list]
    routing_decision: str
    documentState: NotRequired[DocumentIngestState]

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
