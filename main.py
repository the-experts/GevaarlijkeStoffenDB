from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import pymupdf  # pymupdf


from openai import OpenAI


def extract_text_from_pdf(file_path):
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, max_length=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_length - overlap
    return chunks


def embed_chunks(chunks):
    client = OpenAI()

    embedded_chunks = []

    for chunk in chunks:
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding

        embedded_chunks.append(embedding)

    return embedded_chunks


if __name__ == "__main__":
    load_dotenv()

    text = extract_text_from_pdf("ADN+2023.pdf")

    chunks = split_text(text)

    embedded = embed_chunks(chunks)




