import os
import re
import base64
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(BASE_DIR, "mining_docs")
os.makedirs(DOCS_FOLDER, exist_ok=True)

st.set_page_config(
    page_title="Assistant | Mining",
    page_icon="⛏️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def set_mining_background():
    img_path = os.path.join(BASE_DIR, "mining_bg.jpg")
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        bg_css = f'background: linear-gradient(rgba(10, 15, 28, 0.55), rgba(10, 15, 28, 0.72)), url("data:image/jpeg;base64,{encoded}") no-repeat center center fixed !important; background-size: cover !important;'
    else:
        bg_css = 'background-color: #0F172A !important;'

    st.html(f"""
    <style>
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            {bg_css}
        }}
        [data-testid="stHeader"], [data-testid="stBottom"], [data-testid="stBottom"] > div {{
            background: transparent !important;
        }}
        .block-container {{
            max-width: 720px;
            padding-top: 2rem;
            padding-bottom: 5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }}
        h1, .stCaption, p, li, label {{
            color: #F8FAFC !important;
            text-shadow: 0 1px 4px rgba(0, 0, 0, 0.85);
        }}
        .stChatMessage {{
            background-color: rgba(15, 23, 42, 0.88) !important;
            border: 1px solid rgba(245, 158, 11, 0.55) !important;
            border-radius: 12px !important;
            padding: 0.85rem !important;
        }}
        @media (max-width: 640px) {{
            h1 {{ font-size: 1.6rem !important; }}
            .stChatMessage {{ font-size: 0.95rem !important; }}
        }}
    </style>
    """)

set_mining_background()
st.title("⛏️ THAR-AI | Mining Assistant")
st.caption("Ask any question from the Mining & HSE Knowledge Base")

groq_key = ""
try:
    groq_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    pass
groq_key = (groq_key or os.getenv("GROQ_API_KEY", "")).strip()

with st.sidebar:
    st.header("📂 Manage PDF Files")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for up_file in uploaded_files:
            with open(os.path.join(DOCS_FOLDER, up_file.name), "wb") as f:
                f.write(up_file.getbuffer())
        st.cache_resource.clear()
        st.success("PDF(s) indexed!")

    pdf_files = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith(".pdf")]
    st.write(f"Loaded PDFs: **{len(pdf_files)}**")
    for pdf in pdf_files:
        st.markdown(f"- `{pdf}`")
    if st.button("Rebuild Index"):
        st.cache_resource.clear()
        st.rerun()

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

@st.cache_resource
def build_vector_store(file_list):
    if not file_list:
        return None
    all_docs = []
    for fname in file_list:
        loader = PyMuPDFLoader(os.path.join(DOCS_FOLDER, fname))
        all_docs.extend(loader.load())
    if not all_docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=250)
    chunks = splitter.split_documents(all_docs)
    return FAISS.from_documents(chunks, load_embeddings())

vectorstore = build_vector_store(tuple(sorted(pdf_files)))

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask me anything about mining")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if vectorstore is None:
            reply = "⚠️ No PDF file loaded. Please upload a PDF from the sidebar."
            st.warning(reply)
        else:
            follow_up_words = {"summarize", "summary", "short", "shorten", "chota", "mukhtasar", "brief", "bullet", "points", "explain", "isay", "isko"}
            query_tokens = set(re.findall(r'\w+', user_query.lower()))
            is_followup = len(st.session_state.messages) >= 2 and bool(query_tokens & follow_up_words) and len(query_tokens) <= 7

            if is_followup:
                last_bot_reply = st.session_state.messages[-2]["content"]
                context_text = f"Previous Answer to refine:\n{last_bot_reply}"
            else:
                results = vectorstore.similarity_search_with_score(user_query, k=4)
                stop_words = {"what", "is", "are", "a", "an", "the", "of", "in", "for", "to", "how", "why", "define", "explain", "tell", "me", "about", "kia", "kya", "ha", "hai", "btao"}
                keywords = [w for w in query_tokens if w not in stop_words and len(w) > 2]
                best_hits = max((sum(1 for kw in keywords if kw in d.page_content.lower()) for d, _ in results), default=0)

                if best_hits == 0 and results[0][1] > 1.40:
                    context_text = ""
                else:
                    context_text = "\n\n---\n\n".join(
                        [f"[Source: {os.path.basename(d.metadata.get('source', 'PDF'))} | Page {d.metadata.get('page', 0)+1}]\n{d.page_content}" for d, _ in results[:3]]
                    )

            if not context_text:
                reply = "I don't know about it."
                st.markdown(reply)
           elif groq_key:
                try:
                    client = Groq(api_key=groq_key)
                    models = [
                        m.id for m in client.models.list().data
                        if not any(x in m.id.lower() for x in ["whisper", "guard", "tts", "orpheus", "playai", "distil"])
                    ]
                    selected = "llama-3.3-70b-versatile" if "llama-3.3-70b-versatile" in models else (
                        next((m for m in models if "70b" in m or "llama-3" in m), models[0])
                    )

                    sys_prompt = (
                        "You are a strict PDF-locked Mining Assistant.\n"
                        "RULES:\n"
                        "1. Answer ONLY using the provided PDF Context and the Recent Chat History.\n"
                        "2. NEVER use outside knowledge. If the answer is not in the context or chat history, reply ONLY with: I don't know about it.\n"
                        "3. Keep answers concise, direct, and to-the-point (3-5 lines or crisp bullet points) unless asked for detail.\n"
                        "4. If the user asks to summarize, shorten, or merge previous answers, combine/summarize ONLY from the Recent Chat History.\n"
                        "5. End with the exact Source file and Page number when referencing PDF content."
                    )

                    recent_history = "\n".join(
                        [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages[-6:-1]]
                    )

                    res = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": f"Recent Chat History:\n{recent_history}\n\nPDF Context:\n{context_text}\n\nUser Request: {user_query}"}
                        ],
                        model=selected,
                        temperature=0.0,
                        max_tokens=450
                    )
                    reply = res.choices[0].message.content
                    st.markdown(reply)
                except Exception as e:
                    reply = f"⚠️ Engine error: {e}"
                    st.error(reply)
            else:
                reply = context_text[:600] + "..."
                st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
