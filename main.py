from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from openai import OpenAI

def load_pdf(file_path):
    loader = UnstructuredPDFLoader(file_path)  # <-- jouw bestand
    return loader.load()

def text_to_chunks(documents, size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,  # aantal tekens per chunk
        chunk_overlap=200,  # overlap om context te behouden
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return splitter.split_documents(documents)

def embed_chunks(chunks):
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

    return embedded_chunks


# -- Main function --

if __name__ == "__main__":
    load_dotenv()

    # 1. PDF inladen
    documents = load_pdf("ADN+2023.pdf") # <-- jouw bestand
    print(f"PDF geladen met {len(documents)} document(en)")

    # 2. Tekst opdelen in chunks
    splitted_docs = text_to_chunks(documents)
    print(f"Opgesplitst in {len(splitted_docs)} chunks")

    # 3. embed chuncks
    embedded = embed_chunks(splitted_docs)

    # (optioneel) bekijken wat er in zit
    for i, doc in enumerate(splitted_docs[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print(doc.page_content[:500])
