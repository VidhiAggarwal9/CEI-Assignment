from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ingest import load_pdf, split_documents

def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def build_vectorstore(chunks, embeddings):
    """
    Takes text chunks + embedding model, creates a FAISS vector store,
    and saves it to disk so we don't have to re-embed every time.
    """
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

if __name__ == "__main__":
    # Step A: load + chunk the PDF (reusing our ingest.py functions)
    docs = load_pdf("data/resume.pdf")
    chunks = split_documents(docs)
    print(f"Number of chunks: {len(chunks)}")

    # Step B: load embedding model
    embeddings = get_embedding_model()

    # Step C: build and save the vector store
    vectorstore = build_vectorstore(chunks, embeddings)
    print("Vector store created and saved to 'faiss_index' folder!")

    # Step D: quick test - search for something relevant
    query = "What programming languages does this person know?"
    results = vectorstore.similarity_search(query, k=2)

    print("\n---- Top 2 relevant chunks for the query ----")
    for i, r in enumerate(results):
        print(f"\nResult {i+1}:")
        print(r.page_content)