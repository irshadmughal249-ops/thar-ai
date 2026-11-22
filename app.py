import os
import re
import base64
import streamlit as st
from groq import Groq
from google import genai
from google.genai import types
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

st.set_page_config(
    page_title="Assistant",
    page_icon="⛏️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container {
        max-width: 720px;
        padding-top: 1.5rem;
        padding-bottom: 5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    @media (max-width: 640px) {
        h1 {
            font-size: 1.6rem !important;
        }
        .stChatMessage {
            padding: 0.6rem !important;
            font-size: 0.95rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)
st.title("AI Assistant")
st.caption("Mining")

docs_folder = os.path.join(BASE_DIR, "mining_docs")
if not os.path.exists(docs_folder):
    os.makedirs(docs_folder)

pdf_files = [f for f in os.listdir(docs_folder) if f.lower().endswith(".pdf")]

groq_key = ""
gemini_key = ""
try:
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    gemini_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    pass

groq_key = (groq_key or os.getenv("GROQ_API_KEY", "")).strip()
gemini_key = (gemini_key or os.getenv("GEMINI_API_KEY", "")).strip()

with st.sidebar:
    st.header("Knowledge Base")
    st.write(f"Loaded PDFs: **{len(pdf_files)}**")
    for pdf in pdf_files:
        st.markdown(f"- `{pdf}`")
    st.divider()
    st.subheader("AI Engine Status")
    st.write(f"Groq API: {'✅ Connected' if groq_key else '❌ Missing Key'}")
    st.write(f"Gemini API: {'✅ Connected' if gemini_key else '❌ Missing Key'}")
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

user_query = st.chat_input("Ask me anything about Mining")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if vectorstore is None:
            reply = "⚠️ Please add a readable PDF to the `mining_docs` folder and click **Rebuild Index**."
            st.warning(reply)
        else:
            with st.spinner("Searching manual & generating answer..."):
                relevant_docs = vectorstore.similarity_search(user_query, k=5)

                context_text = "\n\n---\n\n".join(
                    [
                        f"[Document: {os.path.basename(d.metadata.get('source', 'Unknown'))} | Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
                        for d in relevant_docs
                    ]
                )

                system_instruction = (
                    "You are THAR-AI, a strict document-grounded Mining & HSE Operations Co-Pilot for SECMC Thar Block II.\n"
                    "You MUST answer ONLY using the exact facts, definitions, and numbers written inside the provided 'Manual Context' below.\n\n"
                    "STRICT ANTI-HALLUCINATION RULES:\n"
                    "1. **ONLY Use Manual Context:** Do NOT use your outside/general AI knowledge. Every single point in your answer must come directly from the provided Manual Context.\n"
                    "2. **If Not Found in PDF:** If the user's question is NOT answered in the Manual Context, say clearly in the user's language: 'Yeh maloomat uploaded manual/PDF mein mojood nahi hai' (do NOT guess or make up an answer).\n"
                    "3. **Extract Single Relevant Q&A:** Ignore unrelated numbered questions in the chunk and translate/explain ONLY the specific Q&A that matches the user's query.\n"
                    "4. **Language Match:** Reply in the exact language the user asked in (Roman Urdu, Urdu, Sindhi, or English), keeping key technical terms clear.\n"
                    "5. **Mandatory Citation:** Always end with the exact Document Name, Question Number, and Page Number from the context."
                )

                user_prompt = f"Manual Context:\n{context_text}\n\nUser Question: {user_query}"
                reply = None

                if groq_key:
                    try:
                        groq_client = Groq(api_key=groq_key)
                        available_models = [
                            m.id for m in groq_client.models.list().data
                            if "whisper" not in m.id and "guard" not in m.id and "tts" not in m.id
                        ]
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

                if not reply:
                    src = os.path.basename(relevant_docs[0].metadata.get("source", "Unknown"))
                    pg = relevant_docs[0].metadata.get("page", 0) + 1
                    exact_qa = extract_single_qa(user_query, relevant_docs)
                    reply = f"**[{src} — Page {pg}]**\n\n{exact_qa}"

                st.markdown(reply)
                with st.expander("View Retrieved Reference Clauses"):
                    st.text(context_text)

    st.session_state.messages.append({"role": "assistant", "content": reply})
