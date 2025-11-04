from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START  # ✅ Import END from here

from State import AgentState
from Tools import embed_question, router_agent, PBM_agent, stoffen_agent, route_query, query_db

# Initialize model
model = ChatOpenAI(model="gpt-4", temperature=0)



# Define workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("embed_question", embed_question)
workflow.add_node("query_db", query_db)
workflow.add_node("router_agent", router_agent)
workflow.add_node("stoffen_agent", stoffen_agent)
workflow.add_node("PBM_agent", PBM_agent)

# Add edges
workflow.add_edge(START, "embed_question")
workflow.add_edge("embed_question", "query_db")
workflow.add_edge("query_db", "router_agent")

# Conditional routing
workflow.add_conditional_edges(
    "router_agent",
    route_query,
    {
        "stoffen": "stoffen_agent",
        "pbm": "PBM_agent"
    },
)

# End edges - use END constant from langgraph.graph
workflow.add_edge("PBM_agent", END)
workflow.add_edge("stoffen_agent", END)

# Compile the graph
graph = workflow.compile()

# Optional: Visualize
# try:
#     from IPython.display import Image, display
#     png_data = graph.get_graph(xray=True).draw_mermaid_png()
#     with open("workflow_graph.png", "wb") as f:
#         f.write(png_data)
#     print("Graph saved as workflow_graph.png")
#
#     # Then open the file manually or:
#     import os
#
#     os.startfile("workflow_graph.png")  # Windows
# except:
#     print("Could not display graph")

# Test the workflow
messages = [HumanMessage(content="Welke voorwaarden hebben schepen waarvan de ladingzone is voor 30 december 2018 is omgebouwd?")]
result = graph.invoke({"messages": messages})
print(result)