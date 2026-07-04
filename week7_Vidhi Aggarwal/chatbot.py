import os
from dotenv import load_dotenv
from groq import Groq
from vectorstore import get_embedding_model
from langchain_community.vectorstores import FAISS

# Load environment variables from .env
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_vectorstore():
    """
    Loads the FAISS index we already saved to disk in Step 5.
    """
    embeddings = get_embedding_model()
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True  # safe here since WE created this index ourselves
    )
    return vectorstore

def retrieve_context(vectorstore, query, k=3):
    """
    Finds the top-k most relevant chunks for the user's question.
    """
    results = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([r.page_content for r in results])
    return context

def generate_answer(query, context):
    """
    Sends the question + retrieved context to the LLM,
    instructing it to answer ONLY using that context.
    """
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    vectorstore = load_vectorstore()

    query = "What programming languages does this person know?"
    context = retrieve_context(vectorstore, query)

    print("---- Retrieved Context ----")
    print(context)

    answer = generate_answer(query, context)
    print("\n---- Generated Answer ----")
    print(answer)