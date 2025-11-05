from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import OpenAI as OpenAILangchain
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI

from dotenv import load_dotenv

from PostgresDBConnector import PostgresDBConnector
from State import AgentState, DocumentIngestState


from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Import PDF processing functions from main.py
from main import extract_text_and_tables, process_pages_to_chunks, embed_chunks_with_metadata
import os
import logging

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


# ============================================================================
# DOCUMENT INGESTION AGENT NODES
# ============================================================================

logger = logging.getLogger(__name__)


def validate_pdf_node(state: DocumentIngestState) -> DocumentIngestState:
    """Validate PDF file and check for duplicates."""

    file_path = state["file_path"]
    source_filename = state["source_filename"]

    logger.info(f"Validating PDF: {source_filename}")

    # Check file exists
    if not os.path.exists(file_path):
        return {
            "status": "error",
            "error_message": f"File not found: {file_path}"
        }

    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return {
            "status": "error",
            "error_message": "File is empty"
        }

    # Check for duplicates in database
    db_check = PostgresDBConnector()
    try:
        if db_check.document_exists(source_filename):
            return {
                "status": "error",
                "error_message": f"Document '{source_filename}' already exists in database"
            }
    finally:
        db_check.close_pool()

    logger.info(f"✓ Validated PDF: {source_filename} ({file_size} bytes)")

    return {
        "status": "extracting",
        "current_step": "Extracting text and tables from PDF"
    }


def extract_content_node(state: DocumentIngestState) -> DocumentIngestState:
    """Extract text and tables from PDF."""

    file_path = state["file_path"]

    logger.info("Extracting content from PDF...")

    try:
        # Use existing function from main.py
        pages = extract_text_and_tables(file_path)

        if not pages:
            return {
                "status": "error",
                "error_message": "No extractable content in PDF"
            }

        logger.info(f"✓ Extracted {len(pages)} pages")

        return {
            "extracted_pages": pages,
            "status": "chunking",
            "current_step": f"Chunking {len(pages)} pages into manageable segments"
        }

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {
            "status": "error",
            "error_message": f"Extraction failed: {str(e)}"
        }


def chunk_text_node(state: DocumentIngestState) -> DocumentIngestState:
    """Split pages into chunks with metadata."""

    pages = state["extracted_pages"]
    max_length = state.get("max_length", 1000)
    overlap = state.get("overlap", 100)

    logger.info(f"Chunking {len(pages)} pages...")

    try:
        # Use existing function from main.py
        chunks = process_pages_to_chunks(pages, max_length, overlap)

        logger.info(f"✓ Created {len(chunks)} chunks")

        return {
            "chunks": chunks,
            "status": "embedding",
            "current_step": f"Generating embeddings for {len(chunks)} chunks"
        }

    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        return {
            "status": "error",
            "error_message": f"Chunking failed: {str(e)}"
        }


def embed_chunks_node(state: DocumentIngestState) -> DocumentIngestState:
    """Generate embeddings for chunks."""

    chunks = state["chunks"]

    logger.info(f"Generating embeddings for {len(chunks)} chunks...")

    try:
        # Use existing function from main.py
        chunks_with_embeddings = embed_chunks_with_metadata(chunks)

        logger.info(f"✓ Generated {len(chunks_with_embeddings)} embeddings")

        return {
            "chunks_with_embeddings": chunks_with_embeddings,
            "status": "storing",
            "current_step": f"Storing {len(chunks_with_embeddings)} chunks in database"
        }

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return {
            "status": "error",
            "error_message": f"Embedding failed: {str(e)}"
        }


def store_chunks_node(state: DocumentIngestState) -> DocumentIngestState:
    """Store chunks with embeddings in database."""

    chunks_with_embeddings = state["chunks_with_embeddings"]
    source_filename = state["source_filename"]

    logger.info(f"Storing {len(chunks_with_embeddings)} chunks in database...")

    db_store = PostgresDBConnector()

    try:
        # Prepare batch data
        batch_data = [
            (
                source_filename,
                chunk['type'],
                chunk['page_number'],
                chunk['chunk_index'],
                chunk['content'],
                chunk['embedding']
            )
            for chunk in chunks_with_embeddings
        ]

        # Store in database
        rows_affected = db_store.store_document_chunks_batch(batch_data)

        logger.info(f"✓ Stored {rows_affected} chunks")

        return {
            "chunks_stored": rows_affected,
            "status": "complete",
            "current_step": "Document processing complete"
        }

    except Exception as e:
        logger.error(f"Storage failed: {e}")
        return {
            "status": "error",
            "error_message": f"Storage failed: {str(e)}"
        }
    finally:
        db_store.close_pool()