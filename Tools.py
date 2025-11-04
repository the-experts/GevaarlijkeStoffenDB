from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import OpenAI as OpenAILangchain
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI

from dotenv import load_dotenv

from State import AgentState

load_dotenv()

client = OpenAI()

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
        "embedded_query": embedded,
        "messages": []
    }


def query_db(state: AgentState) -> AgentState:
    """Query the database using the vectors from the embedded question."""
    messages = state["messages"]

    # Find the last message with embeddings
    embeddings = None
    for msg in reversed(messages):
        if hasattr(msg, 'additional_kwargs') and 'embeddings' in msg.additional_kwargs:
            embeddings = msg.additional_kwargs['embeddings']
            break

    # Query database
    # results = search_database(embeddings)
    results = "result"

    return {
        "messages": [AIMessage(content=f"Found {len(results)} results")]
    }


def router_agent(state: AgentState) -> AgentState:  # ← ADD state parameter!
    """Router agent - just passes through"""
    # This node doesn't do anything except pass state through
    # The actual routing logic is in the route_query function
    return {"messages": []}

def query_db(vectors):
    client = OpenAILangchain()

    for chunk in vectors:  # todo: change
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding

def stoffen_agent(state: AgentState) -> AgentState:
    """Stoffen agent"""
    results = state.get("db_results", [])
    query = state.get("query", "")


    response = f"Found {len(results)} dangerous substances matching '{query}': {', '.join(results)}"

    return {
        "messages": [AIMessage(content=response)]
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