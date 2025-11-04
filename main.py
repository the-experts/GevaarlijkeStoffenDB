from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import pymupdf  # pymupdf
import os
import pdfplumber

from openai import OpenAI
from PostgresDBConnector import PostgresDBConnector

def extract_text_and_tables(pdf_path):
    content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                content.append({
                    "type": "text",
                    "page": page_number,
                    "content": text
                })
            for table in page.extract_tables():
                cleaned_rows = []
                for row in table:
                    cleaned_row = [cell if cell is not None else "" for cell in row]
                    cleaned_rows.append(" | ".join(cleaned_row))
                table_text = "\n".join(cleaned_rows)
                if table_text.strip():
                    content.append({
                        "type": "table",
                        "page": page_number,
                        "content": table_text
                    })
    return content


def extract_text_from_pdf(file_path):
    """Extract text from PDF, preserving page numbers.

    Returns:
        list of tuples: [(page_number, text), ...]
    """
    doc = pymupdf.open(file_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        # Remove NUL characters that PostgreSQL cannot handle
        text = text.replace('\x00', '').replace('\u0000', '')
        if text.strip():  # Only include pages with content
            pages.append((page_num, text))
    doc.close()
    return pages

def split_text(text, max_length=1000, overlap=100):
    """Split text into overlapping chunks.

    Args:
        text: Text to split
        max_length: Maximum chunk length in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        list of strings: Text chunks
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_length - overlap
    return chunks


def process_pages_to_chunks(pages, max_length=1000, overlap=100, min_last_chunk=500):
    """Process pages into chunks with metadata.

    Args:
        pages: List of (page_number, text) tuples
        max_length: Maximum chunk length in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        list of dicts: [{'page_number': int, 'chunk_index': int, 'content': str}, ...]
    """
    result = []
    global_chunk_index = 0

    for item in pages:
      cleaned_content = item["content"].strip().replace('\x00', '').replace('\u0000', '')
      if item["type"] == "table":
        # Append tables directly
        result.append({
          'page_number': item["page"],
          'chunk_index': global_chunk_index,
          'type': 'table',
          'content': cleaned_content
        })
        global_chunk_index += 1
      elif item["type"] == "text":
        text = cleaned_content
        text_length = len(text)

        if text_length <= max_length:
          # Text fits in one chunk
          result.append({
            'page_number': item["page"],
            'chunk_index': global_chunk_index,
            'type': 'text',
            'content': text
          })
          global_chunk_index += 1
        else:
          # First, split using the provided function
          chunks = split_text(text, max_length, overlap)

          # Check if last chunk is too small
          if len(chunks) > 1 and len(chunks[-1]) < min_last_chunk:
            # Recalculate to distribute evenly
            num_chunks = len(chunks)
            adjusted_max_length = text_length // num_chunks
            chunks = split_text(text, adjusted_max_length, overlap)

          # Add each chunk to result with chunk index
          for chunk in enumerate(chunks):
            result.append({
              'page_number': item["page"],
              'chunk_index': global_chunk_index,
              'type': 'text',
              'content': chunk[1]
            })
            global_chunk_index += 1

    return result


def embed_chunks(chunks):
    """Generate embeddings for text chunks (legacy function).

    Args:
        chunks: List of text strings

    Returns:
        list: List of embeddings
    """
    client = OpenAI()

    embedded_chunks = []

    for chunk in chunks:
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding

        embedded_chunks.append(embedding)

    return embedded_chunks


def embed_chunks_with_metadata(chunks_with_metadata):
    """Generate embeddings for chunks with metadata.

    Args:
        chunks_with_metadata: List of dicts with 'page_number', 'chunk_index', 'content'

    Returns:
        list: List with 'embedding' field added to each valid chunk (empty chunks are filtered out)
    """
    client = OpenAI()
    valid_chunks = []
    skipped_count = 0

    for chunk_data in chunks_with_metadata:
        # Skip empty or whitespace-only chunks
        if not chunk_data['content'] or not chunk_data['content'].strip():
            print(f"⚠ Skipping empty chunk {chunk_data['chunk_index']} from page {chunk_data['page_number']}")
            skipped_count += 1
            continue

        print(f"Embedding chunk {chunk_data['chunk_index']} from page {chunk_data['page_number']}...")
        try:
            embedding = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk_data['content']
            ).data[0].embedding

            chunk_data['embedding'] = embedding
            valid_chunks.append(chunk_data)

        except Exception as e:
            print(f"✗ Error embedding chunk {chunk_data['chunk_index']}: {e}")
            skipped_count += 1
            continue

    if skipped_count > 0:
        print(f"⚠ Skipped {skipped_count} empty or invalid chunks")

    return valid_chunks


def process_and_store_pdf(pdf_path, db_connector, max_length=1000, overlap=100):
    """Complete pipeline: extract PDF, chunk, embed, and store in database.

    Args:
        pdf_path: Path to the PDF file
        db_connector: PostgresDBConnector instance
        max_length: Maximum chunk length in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        int: Number of chunks stored
    """
    print(f"\n{'='*60}")
    print(f"Processing PDF: {pdf_path}")
    print(f"{'='*60}\n")

    # Extract filename for database storage
    source_file = os.path.basename(pdf_path)

    # Step 1: Extract text from PDF
    print("Step 1: Extracting text from PDF...")
    pages = extract_text_and_tables(pdf_path)
    print(f"✓ Extracted {len(pages)} pages\n")

    # Step 2: Create chunks with metadata
    print("Step 2: Creating chunks with metadata...")
    chunks = process_pages_to_chunks(pages, max_length, overlap)
    print(f"✓ Created {len(chunks)} chunks\n")

    # Step 3: Generate embeddings
    print("Step 3: Generating embeddings...")

    chunks_with_embeddings = embed_chunks_with_metadata(chunks)
    print(f"✓ Generated {len(chunks_with_embeddings)} embeddings\n")

    # Step 4: Prepare data for batch insert
    print("Step 4: Storing in database...")
    batch_data = [
        (
            source_file,
            chunk['type'],
            chunk['page_number'],
            chunk['chunk_index'],
            chunk['content'],
            chunk['embedding']
        )
        for chunk in chunks_with_embeddings
    ]

    # Step 5: Store in database
    rows_affected = db_connector.store_document_chunks_batch(batch_data)
    print(f"✓ Stored {rows_affected} chunks in database\n")

    print(f"{'='*60}")
    print(f"✓ Processing complete!")
    print(f"{'='*60}\n")

    return rows_affected


if __name__ == "__main__":
    load_dotenv()

    # Initialize database connector
    db = PostgresDBConnector()

    try:
        # Process PDF and store in database
        pdf_path = ("ADN+2023+Small.pdf")
        chunks_stored = process_and_store_pdf(
            pdf_path=pdf_path,
            db_connector=db,
            max_length=1000,
            overlap=100
        )

        print(f"\n✓ Successfully processed and stored {chunks_stored} chunks from {pdf_path}")

        # Example: Search for similar chunks
        print("\n" + "="*60)
        print("Example: Searching for similar chunks...")
        print("="*60)

        # Create a test query embedding
        client = OpenAI()
        test_query = "Welke voorwaarden hebben schepen waarvan de ladingzone is voor 30 december 2018 is omgebouwd?"
        query_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=test_query
        ).data[0].embedding

        # Search for similar chunks
        results = db.search_similar_chunks(query_embedding, limit=3)

        print(f"\nQuery: '{test_query}'")
        print(f"Top {len(results)} similar chunks:\n")

        for i, (id, source_file, page_number, chunk_index, content, similarity) in enumerate(results, 1):
            print(f"{i}. [Page {page_number}, Chunk {chunk_index}] Similarity: {similarity:.4f}")
            print(f"   Source: {source_file}")
            print(f"   Content preview: {content[:200]}...")
            print()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up database connection
        db.close_pool()
