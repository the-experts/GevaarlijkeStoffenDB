from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import OpenAI as OpenAILangchain
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI

from dotenv import load_dotenv

from PostgresDBConnector import PostgresDBConnector
from State import AgentState


from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

client = OpenAI()
db = PostgresDBConnector()

def load_pdf(file_path):
    """Inladen van de PDF"""
    loader = UnstructuredPDFLoader(file_path)  # <-- jouw bestand
    return loader.load()

def text_to_chunks(documents, size=1000, overlap=200):
    """split de tekst van de pdf in chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,  # aantal tekens per chunk
        chunk_overlap=overlap,  # overlap om context te behouden
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return splitter.split_documents(documents)

def embed_chunks(chunks):
    """Query de database"""
    client = OpenAI()
    embedded_chunks = []

    # Filter lege chunks
    texts = [c.page_content for c in chunks if c.page_content.strip()]

    # In batches van 100 (API ondersteunt tot 2048 inputs per call)
    batch_size = 100
    for batch in range(0, len(texts), batch_size):
        batch = texts[batch:batch + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        for e in response.data:
            embedded_chunks.append(e.embedding)


def embed_question(state: AgentState) -> AgentState:
    """Embed text question into vector representations."""
    """Embed the question"""
    messages = state["messages"]
    question = messages[-1].content if messages else ""

    print(f"📝 Embedding: {question}")

    try:
        # Correct way to create embeddings
        response = client.embeddings.create(
            model="text-embedding-3-small",  # or "text-embedding-ada-002"
            input=question
        )

        # Extract embedding from response
        embedded = response.data[0].embedding

    except Exception as e:
        print(f"Error creating embedding: {e}")
        embedded = []

    return {
        "query": question,
        "embedded_query": embedded,  # ← This gets added to state
        "messages": [],
        "db_results": []
    }


def query_db(state: AgentState) -> AgentState:
    """Query the database using the vectors from the embedded question."""

    # Get the embedding that was stored by embed_question
    embedded_query = state.get("embedded_query", [])  # ← Access from state
    query = state.get("query", "")

    print(f"🔍 Querying DB for: {query}")
    print(f"   Using embedding with {len(embedded_query)} dimensions")

    # Use the embeddings to query your vector database
    if embedded_query:
        # Example: similarity search with your vector DB
        results = db.search_similar_chunks(embedded_query, limit=3)
    else:
        print("⚠️  No embeddings available")
        results = []

    return {
        "db_results": results,  # ← Store results for next node
        "messages": []
    }


# Initialize LLM for routing
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def router_agent(state: AgentState) -> AgentState:
    """Use LLM to intelligently route the query"""
    query = state.get("query", "")

    print(f"🔀 Router analyzing: {query}")

    # Create routing prompt
    routing_prompt = f"""You are a routing agent for a dangerous substances database system.

Available agents:
1. "stoffen" - Handles queries about dangerous substances, chemicals, safety data, hazards
2. "pbm" - Handles queries about personal protective equipment (PBM), safety gear, protection measures

User query: {query}

Which agent should handle this query? Respond with ONLY "stoffen" or "pbm"."""

    response = router_llm.invoke([
        SystemMessage(content="You are a routing classifier."),
        HumanMessage(content=routing_prompt)
    ])

    # Parse response
    decision = response.content.lower().strip()

    # Store the routing decision in state
    return {
        "routing_decision": decision,
        "messages": []
    }


def route_query(state: AgentState) -> str:
    """Get the routing decision from state"""
    decision = state.get("routing_decision", "stoffen")
    print(f"→ Routing to: {decision}")
    return decision



def stoffen_agent(state: AgentState) -> AgentState:
    """Stoffen agent - processes dangerous substances queries"""
    # Initialize LLM (do this once at the top of your file)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    results = state.get("db_results", [])
    query = state.get("query", "")

    print(f"⚠️  Stoffen agent processing query: {query}")
    print(f"   Found {len(results)} database results")

    # Create prompt for the LLM
    system_prompt = """You are an expert on dangerous substances and chemical safety.
Your role is to provide accurate, helpful information about hazardous materials,
their properties, risks, and safety measures.

Always prioritize safety and provide clear, actionable information."""

    user_prompt = f"""User query: {query}

Database results found:
{results}

Please provide a comprehensive answer about the dangerous substances related to this query.
Include information about hazards, safety measures, and any relevant warnings."""

    # Call the LLM
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    print(f"✅ Generated response: {response.content[:100]}...")

    return {
        "messages": [AIMessage(content=response.content)]
    }



def PBM_agent(state: AgentState) -> AgentState:
    """PBM agent"""
    results = state.get("db_results", [])
    query = state.get("query", "")

    print(f"🔧 Processing PBM query")

    response = f"PBM information for '{query}': Found {len(results)} results - {', '.join(results)}"

    return {
        "messages": [AIMessage(content=response)]
    }