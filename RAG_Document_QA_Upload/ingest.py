from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def split_documents(documents):
    """
    Splits documents into smaller chunks.
    chunk_size = max characters per chunk
    chunk_overlap = characters repeated between chunks, so context isn't lost at boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    file_path = "data/resume.pdf"
    docs = load_pdf(file_path)
    print(f"Number of pages loaded: {len(docs)}")

    chunks = split_documents(docs)
    print(f"Number of chunks created: {len(chunks)}")

    print("---- Preview of first chunk ----")
    print(chunks[0].page_content)

    print("\n---- Preview of second chunk ----")
    print(chunks[1].page_content)