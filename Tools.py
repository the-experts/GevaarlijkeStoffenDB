from langchain_core.tools import tool
from langchain_openai import OpenAI


@tool
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


@tool
def query_db(vectors):
    client = OpenAI()

    embedded_chunks = []

    for chunk in chunks:
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding

        embedded_chunks.append(embedding)

    return embedded_chunks