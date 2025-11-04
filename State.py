from typing import NotRequired

from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    query: NotRequired[str]
    embedded_query: NotRequired[list]
    db_results: NotRequired[list]
    routing_decision: str

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int