from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── 1. Load the job database ───────────────────────────────────────────────
# Connects to the vector DB already built — no re-embedding needed
embeddings = OllamaEmbeddings(model="mistral")

vectordb = Chroma(
    persist_directory="/Users/kinjalsaxena/Documents/job-copilot/chroma_db",
    embedding_function=embeddings
    collection_name="job_postings"   # ← add this

)

# ── 2. Set up the retriever ────────────────────────────────────────────────
# On each question, this finds the 5 most semantically relevant jobs
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# ── 3. Load the local LLM ─────────────────────────────────────────────────
# Mistral runs fully on this machine — nothing sent to external servers
llm = OllamaLLM(model="mistral", temperature=0.1)  # low temp = factual answers

# ── 4. Define the assistant's behavior ────────────────────────────────────
# Keeps responses neutral, grounded, and always tied to real job data
prompt = PromptTemplate.from_template("""
You are a helpful job search assistant for a senior data scientist.
Answer using ONLY the job listings provided below — no guessing or adding context.
Always mention the job title and company when referencing a role.

Here are the relevant job listings:
{context}

Question: {question}

Answer (use bullet points when listing multiple roles):
""")

# ── 5. Format retrieved jobs into readable chunks ──────────────────────────
# Structures each job so the LLM can clearly parse title, company, and details
def format_jobs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        title   = doc.metadata.get("title",   "Unknown Title")
        company = doc.metadata.get("company", "Unknown Company")
        formatted.append(f"Job {i}: {title} at {company}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# ── 6. Wire everything into a RAG pipeline ────────────────────────────────
# Flow: question → retrieve jobs → format → prompt → Mistral → clean answer
rag_chain = (
    {
        "context":  retriever | format_jobs,  # semantic search + formatting
        "question": RunnablePassthrough()     # question passes through unchanged
    }
    | prompt             # both get injected into the prompt template
    | llm                # Mistral generates the answer
    | StrOutputParser()  # strips raw output into a clean string
)

# ── 7. Run a test query ───────────────────────────────────────────────────
if __name__ == "__main__":
    query = "Show me senior data scientist roles that mention causal inference"

    print(f"\n🔍 Searching for: {query}\n")
    print("=" * 50)

    answer = rag_chain.invoke(query)
    print(answer)

    print("=" * 50)
    print("Done. Powered by local job DB + Mistral.\n")
