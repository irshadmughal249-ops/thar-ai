import os
import re
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from google import genai
from google.genai import types
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

st.set_page_config(page_title="THAR-AI | Safety Assistant", page_icon="⛏️", layout="wide")
st.title("⛏️ THAR-AI | Safety & Operations Assistant")
st.caption("Multilingual Mining & HSE Co-Pilot (English | Roman Urdu | اردو | سنڌي)")

docs_folder = os.path.join(BASE_DIR, "mining_docs")
if not os.path.exists(docs_folder):
    os.makedirs(docs_folder)

pdf_files = [f for f in os.listdir(docs_folder) if f.endswith(".pdf")]

groq_key = os.getenv("GROQ_API_KEY", "").strip()
gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

with st.sidebar:
    st.header("Knowledge Base")
    st.write(f"Loaded PDFs: **{len(pdf_files)}**")
    for pdf in pdf_files:
        st.markdown(f"- `{pdf}`")
    st.divider()
    st.subheader("AI Engine Status")
    st.write(f"Groq API: {'✅ Connected' if groq_key else '❌ Missing in .env'}")
    st.write(f"Gemini API: {'✅ Connected' if gemini_key else '❌ Missing in .env'}")
    if st.button("Rebuild Index"):
        st.cache_resource.clear()
        st.rerun()

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource
def build_vector_store():
    if not pdf_files:
        return None
    loader = DirectoryLoader(docs_folder, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    if not documents:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=600)
    chunks = splitter.split_documents(documents)
    embeddings = load_embeddings()
    return FAISS.from_documents(chunks, embeddings)

def extract_single_qa(query, docs):
    """Offline fallback: Manual mein se sirf mutaliqa EK sawal aur uska jawab nikalta hai"""
    full_text = "\n".join([d.page_content for d in docs])
    full_text = re.sub(r'500 Mining Engineering.*?Page \d+ of \d+', ' ', full_text)
    qa_blocks = re.split(r'(?=(?:^|\n|\s)\d{1,3}\.\s+[A-Z])', full_text)

    stop_words = {
        "what", "is", "are", "a", "an", "the", "of", "in", "for", "to", "how", "why",
        "and", "or", "between", "difference", "define", "explain", "tell", "me", "about",
        "kia", "kya", "ha", "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "btao", "batao",
        "chha", "aahe", "ahe", "jo", "ji", "ja"
    }
    keywords = [w.lower() for w in re.findall(r'\w+', query) if w.lower() not in stop_words and len(w) > 2]

    best_block = None
    best_score = 0

    for block in qa_blocks:
        clean_b = block.strip()
        if not clean_b:
            continue
        first_line = clean_b.split("?")[0].lower() if "?" in clean_b else clean_b[:120].lower()
        score = sum(15 for kw in keywords if kw in first_line) + sum(2 for kw in keywords if kw in clean_b.lower())
        if score > best_score:
            best_score = score
            best_block = clean_b

    if best_block and "?" in best_block:
        q_part, ans_part = best_block.split("?", 1)
        sentences = re.split(r'(?<=[.!?])\s+', ans_part.strip())
        clean_sentences = []
        for idx, s in enumerate(sentences[:6]):
            if idx >= 3 and re.match(r'^(A|An|The)\s+[a-z\s-]+\s+(occurs|refers|comprises|involves|is a)\b', s) and not any(kw in s.lower() for kw in keywords):
                break
            clean_sentences.append(s)
        return f"**{q_part.strip()}?**\n\n" + " ".join(clean_sentences).strip()
    return best_block if best_block else docs[0].page_content[:800]

vectorstore = build_vector_store()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask any mining or HSE question (English / Roman Urdu / اردو / سنڌي)...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if vectorstore is None:
            reply = "⚠️ Please add a readable PDF to the `mining_docs` folder and click **Rebuild Index**."
            st.warning(reply)
        else:
            with st.spinner("Understanding your question & translating from manual..."):
                relevant_docs = vectorstore.similarity_search(user_query, k=5)

                context_text = "\n\n---\n\n".join(
                    [
                        f"[Document: {os.path.basename(d.metadata.get('source', 'Unknown'))} | Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
                        for d in relevant_docs
                    ]
                )

                system_instruction = (
                    "You are THAR-AI, a Senior Mining & HSE Operations Co-Pilot for SECMC Thar Block II.\n"
                    "Read the user's question and the retrieved English manual sections, then answer ONLY the user's specific question.\n\n"
                    "STRICT RULES:\n"
                    "1. **Ignore Unrelated Questions:** The retrieved chunks contain multiple numbered questions. Extract ONLY the single rule/definition that directly answers the user's question and completely ignore all other questions in the chunk.\n"
                    "2. **Exact Language Match (CRITICAL):**\n"
                    "   - If the user writes in **Roman Urdu** (e.g., 'overburden kia ha'), write the ENTIRE explanation in natural, clear **Roman Urdu** while keeping key mining terms (e.g., Overburden, Topsoil, Backfilling, Acid-forming) in brackets or clear form.\n"
                    "   - If the user writes in **Urdu script**, reply in Urdu.\n"
                    "   - If the user writes in **Sindhi**, reply in Sindhi.\n"
                    "   - If the user writes in **English**, reply in English.\n"
                    "3. **Clean Structure & Citation:** Use bullet points for key management/safety steps, and cite the exact Document Name, Question Number, and Page Number."
                )

                user_prompt = f"Manual Context:\n{context_text}\n\nUser Question: {user_query}"
                reply = None

                # 1. Primary Engine: Groq (Auto-detects active model from your Groq account)
                if groq_key:
                    try:
                        groq_client = Groq(api_key=groq_key)
                        available_models = [
                            m.id for m in groq_client.models.list().data
                            if "whisper" not in m.id and "guard" not in m.id and "tts" not in m.id
                        ]
                        # Best available chat model pick karein
                        preferred = [m for m in available_models if "70b" in m or "llama-3" in m or "llama-4" in m or "qwen" in m]
                        selected_model = preferred[0] if preferred else available_models[0]

                        chat_completion = groq_client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": system_instruction},
                                {"role": "user", "content": user_prompt},
                            ],
                            model=selected_model,
                            temperature=0.1,
                            max_tokens=900,
                        )
                        reply = chat_completion.choices[0].message.content
                    except Exception as e:
                        st.caption(f"Groq fallback triggered: {e}")
                # 2. Secondary Engine: Google Gemini
                if not reply and gemini_key:
                    try:
                        gemini_client = genai.Client(
                            api_key=gemini_key,
                            http_options=types.HttpOptions(timeout=25000)
                        )
                        resp = gemini_client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=f"{system_instruction}\n\n{user_prompt}",
                        )
                        reply = resp.text
                    except Exception:
                        reply = None

                # 3. Offline Clean Single-Q&A Fallback
                if not reply:
                    src = os.path.basename(relevant_docs[0].metadata.get("source", "Unknown"))
                    pg = relevant_docs[0].metadata.get("page", 0) + 1
                    exact_qa = extract_single_qa(user_query, relevant_docs)
                    reply = f"**[{src} — Page {pg}]**\n\n{exact_qa}"

                st.markdown(reply)
                with st.expander("View Retrieved Reference Clauses"):
                    st.text(context_text)

    st.session_state.messages.append({"role": "assistant", "content": reply})