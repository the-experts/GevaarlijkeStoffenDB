from dotenv import load_dotenv
from Tools import load_pdf, text_to_chunks, embed_chunks

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
