import streamlit as st
import os
import tempfile
from ingest import load_pdf, split_documents
from vectorstore import get_embedding_model
from langchain_community.vectorstores import FAISS
from chatbot import retrieve_context, generate_answer

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄")

st.title(" Document Question Answering System")
st.write("Upload any PDF (notes, resume, research paper, book) and ask questions grounded in its content.")

# Load embedding model once, cache it (it doesn't change between files)
@st.cache_resource
def load_embedding_model():
    return get_embedding_model()

embeddings = load_embedding_model()

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# We use session_state to remember the vectorstore across reruns,
# and to detect when a NEW file is uploaded (so we know when to rebuild)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

if uploaded_file is not None:
    # Only rebuild the index if this is a different file than last time
    if uploaded_file.name != st.session_state.last_uploaded_filename:
        with st.spinner(f"Processing '{uploaded_file.name}'... (loading, chunking, embedding)"):
            # Save uploaded file to a temporary path so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            docs = load_pdf(tmp_path)
            chunks = split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, embeddings)

            st.session_state.vectorstore = vectorstore
            st.session_state.last_uploaded_filename = uploaded_file.name

            os.remove(tmp_path)  # clean up temp file

        st.success(f"'{uploaded_file.name}' processed! ({len(chunks)} chunks created)")

# Only show the question box once a document has been processed
if st.session_state.vectorstore is not None:
    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Retrieving relevant information..."):
            context = retrieve_context(st.session_state.vectorstore, query)

        with st.spinner("Generating answer..."):
            answer = generate_answer(query, context)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("See retrieved context (for transparency)"):
            st.write(context)
else:
    st.info(" Upload a PDF above to get started.")
