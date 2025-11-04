from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    query: str
    embedded_query: list
    db_results: list

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int