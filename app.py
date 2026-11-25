import os
import re
import base64
from datetime import datetime
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
    page_title="THAR-AI | Mining Assistant",
    page_icon="⛏️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def set_mining_background():
    img_path = os.path.join(BASE_DIR, "mining_bg.jpg")
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        bg_css = f'background: linear-gradient(rgba(10, 15, 28, 0.58), rgba(10, 15, 28, 0.76)), url("data:image/jpeg;base64,{encoded}") no-repeat center center fixed !important; background-size: cover !important;'
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
            max-width: 760px;
            padding-top: 1.8rem;
            padding-bottom: 5.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }}
        h1, .stCaption, p, li, label {{
            color: #F8FAFC !important;
            text-shadow: 0 1px 4px rgba(0, 0, 0, 0.85);
        }}
        .stChatMessage {{
            background-color: rgba(15, 23, 42, 0.90) !important;
            border: 1px solid rgba(245, 158, 11, 0.55) !important;
            border-radius: 12px !important;
            padding: 0.9rem !important;
        }}
        .stButton > button {{
            background-color: rgba(15, 23, 42, 0.85) !important;
            color: #FBBF24 !important;
            border: 1px solid rgba(245, 158, 11, 0.6) !important;
            border-radius: 8px !important;
            font-size: 0.82rem !important;
            padding: 0.3rem 0.6rem !important;
        }}
        .stButton > button:hover {{
            background-color: rgba(245, 158, 11, 0.2) !important;
            border-color: #FBBF24 !important;
        }}
        @media (max-width: 640px) {{
            h1 {{ font-size: 1.55rem !important; }}
            .stChatMessage {{ font-size: 0.95rem !important; }}
        }}
    </style>
    """)

set_mining_background()
st.title("⛏️ THAR-AI | Mining Assistant")
st.caption("Advanced RAG Knowledge Engine • Surface & Underground Mining | HSE")

groq_key = ""
try:
    groq_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    pass
groq_key = (groq_key or os.getenv("GROQ_API_KEY", "")).strip()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📂 Knowledge Base")
    pdf_files = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith(".pdf")]
    st.write(f"Loaded Manuals: **{len(pdf_files)}**")
    for pdf in pdf_files:
        st.markdown(f"- `{pdf}`")

    if st.session_state.messages:
        st.divider()
        chat_export = "\n\n".join(
            [f"[{m['role'].upper()}]:\n{m['content']}" for m in st.session_state.messages]
        )
        st.download_button(
            label="📥 Download Chat Notes",
            data=chat_export,
            file_name=f"THAR_AI_Notes_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=300)
    chunks = splitter.split_documents(all_docs)
    return FAISS.from_documents(chunks, load_embeddings()), chunks

store_data = build_vector_store(tuple(sorted(pdf_files)))
vectorstore, all_chunks = store_data if store_data else (None, [])

def check_greeting_or_smalltalk(text):
    clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
    words = clean.split()
    if not words or len(words) > 7:
        return None

    salam_patterns = ["assalam", "asalam", "salam", "aoa", "assalamu"]
    if any(any(w.startswith(p) for p in salam_patterns) for w in words):
        return "Walaikum Assalam! 👋 Main **THAR-AI** hoon, aapka Mining Engineering & HSE Knowledge Assistant. Mining manuals ya safety se mutalliq koi bhi sawaal poochein!"

    hi_words = {"hi", "hello", "hey", "hlo", "hy", "greetings", "morning", "evening", "afternoon"}
    if words[0] in hi_words or (len(words) <= 3 and any(w in hi_words for w in words)):
        return "Hello! 👋 Welcome to **THAR-AI** (Mining & HSE Knowledge Assistant). How can I help you with your mining manuals today?"

    if any(p in clean for p in ["how are you", "kya haal", "kia hal", "kesy ho", "kaise ho"]):
        return "Alhamdulillah, main bilkul theek hoon! Aap batayein, mining engineering ya HSE ke kis topic par maloomat chahiye?"

    if any(p in clean for p in ["thank you", "thanks", "shukriya", "thx", "jazakallah"]):
        return "You're welcome! Agar kisi aur mining topic par madad chahiye ho to zaroor poochein. ⛏️"

    if any(p in clean for p in ["who are you", "what are you", "tum kon ho", "ap kon ho", "introduce yourself"]):
        return "Main **THAR-AI** hoon — ek specialized Mining Engineering & HSE Assistant jo aapke uploaded mining manuals se exact, verified answers faraham karta hai."

    return None

def get_groq_client_and_model():
    client = Groq(api_key=groq_key)
    models = [
        m.id for m in client.models.list().data
        if not any(x in m.id.lower() for x in ["whisper", "guard", "tts", "orpheus", "playai", "distil"])
    ]
    selected = "llama-3.3-70b-versatile" if "llama-3.3-70b-versatile" in models else (
        next((m for m in models if "70b" in m or "llama-3" in m), models[0])
    )
    return client, selected

def rewrite_query_with_history(client, model, raw_query, history):
    if not history:
        return raw_query, False, False
    recent = "\n".join([f"{m['role'].upper()}: {m['content'][:600]}" for m in history[-4:]])
    prompt = (
        "Analyze the recent chat history and the user's latest message.\n"
        "1. If the user asks to reformat, make it more attractive/clearer, continue an incomplete sentence (like 'including?'), or summarize the previous answer without introducing a new topic, set REFORMAT_PREVIOUS: yes and set QUERY to the main mining topic of the previous assistant answer.\n"
        "2. Otherwise, rewrite the user's message into a standalone mining search query.\n"
        "3. Set DETAILED: yes if the user asks for details, continuation, or attractive structured formatting.\n\n"
        "Respond strictly in 3 lines:\n"
        "QUERY: <standalone search query>\n"
        "DETAILED: <yes or no>\n"
        "REFORMAT_PREVIOUS: <yes or no>\n\n"
        f"Recent Chat:\n{recent}\n\nLatest Message: {raw_query}"
    )
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=120
        )
        out = res.choices[0].message.content
        q_match = re.search(r"QUERY:\s*(.+)", out)
        d_match = re.search(r"DETAILED:\s*(yes|no)", out, re.I)
        r_match = re.search(r"REFORMAT_PREVIOUS:\s*(yes|no)", out, re.I)
        rewritten = q_match.group(1).strip() if q_match else raw_query
        is_detailed = (d_match.group(1).lower() == "yes") if d_match else False
        is_reformat = (r_match.group(1).lower() == "yes") if r_match else False
        return rewritten, is_detailed, is_reformat
    except Exception:
        low = raw_query.lower()
        is_ref = any(w in low for w in ["attractive", "format", "including", "continue", "complete", "summarize", "short", "chota"])
        return raw_query, True, is_ref

def hybrid_retrieve(query, k=6):
    sem_results = vectorstore.similarity_search_with_score(query, k=k)
    stop_words = {
        "what", "is", "are", "a", "an", "the", "of", "in", "for", "to", "how", "why",
        "define", "explain", "tell", "me", "about", "give", "detailed", "answer",
        "more", "summarize", "short", "kia", "kya", "ha", "hai", "btao", "kro", "karo"
    }
    tokens = [w for w in re.findall(r'\w+', query.lower()) if w not in stop_words and len(w) > 2]

    scored_docs = []
    seen = set()
    for doc, dist in sem_results:
        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])
        if key not in seen:
            seen.add(key)
            kw_hits = sum(1 for t in tokens if t in doc.page_content.lower())
            scored_docs.append((kw_hits, -dist, doc))

    if tokens:
        for chunk in all_chunks:
            text_l = chunk.page_content.lower()
            hits = sum(1 for t in tokens if t in text_l)
            if hits >= max(1, len(tokens) // 2):
                key = (chunk.metadata.get("source"), chunk.metadata.get("page"), chunk.page_content[:80])
                if key not in seen:
                    seen.add(key)
                    scored_docs.append((hits, -1.0, chunk))

    scored_docs.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top_docs = scored_docs[:k]
    if not top_docs:
        return "", []
    if top_docs[0][0] == 0 and (-top_docs[0][1]) > 1.45:
        return "", []

    blocks = []
    sources = []
    for _, _, d in top_docs:
        src = f"{os.path.basename(d.metadata.get('source', 'PDF'))} | Page {d.metadata.get('page', 0) + 1}"
        blocks.append(f"[{src}]\n{d.page_content}")
        if src not in sources:
            sources.append(src)
    return "\n\n---\n\n".join(blocks), sources

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

quick_prompt = None
if st.session_state.messages:
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("📝 Summarize", use_container_width=True):
        quick_prompt = "Summarize the previous answer in 3 crisp bullet points."
    if c2.button("📖 Detailed", use_container_width=True):
        quick_prompt = "Give a complete, detailed technical explanation of the previous topic from the PDF."
    if c3.button("📌 Key Points", use_container_width=True):
        quick_prompt = "Extract key technical points, parameters, and safety rules from the previous answer."
    if c4.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

typed_query = st.chat_input("Ask me anything about mining")
user_query = quick_prompt or typed_query

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        greeting_reply = check_greeting_or_smalltalk(user_query)
        if greeting_reply:
            reply = greeting_reply
            st.markdown(reply)
        elif vectorstore is None:
            reply = "⚠️ No PDF manual found in mining_docs."
            st.warning(reply)
        elif not groq_key:
            reply = "⚠️ GROQ_API_KEY is missing in Streamlit Secrets."
            st.error(reply)
        else:
            try:
                client, selected_model = get_groq_client_and_model()
                standalone_query, wants_detail, is_reformat = rewrite_query_with_history(
                    client, selected_model, user_query, st.session_state.messages[:-1]
                )
                if any(w in user_query.lower() for w in ["detail", "detailed", "explain", "more", "elaborate", "tafseel", "attractive", "including", "continue"]):
                    wants_detail = True

                context_text, sources = hybrid_retrieve(standalone_query, k=6 if wants_detail else 4)
                recent_history = "\n".join(
                    [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages[-6:-1]]
                )

                if not context_text and not recent_history:
                    reply = "I don't know about it."
                    st.markdown(reply)
                else:
                    style_rule = (
                        "Present the answer in an attractive, clean, well-organized format using bold headings, crisp bullet points, and complete sentences. NEVER stop mid-sentence or end with a trailing colon like 'including:'."
                        if (wants_detail or is_reformat) else
                        "Keep the answer concise, direct, and complete (3-5 full lines or crisp bullet points). NEVER cut off mid-sentence."
                    )
                    sys_prompt = (
                        "You are THAR-AI, an expert Mining Engineering & HSE Knowledge Assistant.\n"
                        "STRICT RULES:\n"
                        "1. Base your answer strictly on the provided PDF Context and Recent Chat History.\n"
                        "2. NEVER invent external facts not supported by the PDF Context or Recent Chat History.\n"
                        "3. If the requested topic is completely absent from both the PDF Context and Recent Chat History, reply ONLY with: I don't know about it.\n"
                        "4. If the user asks to make the previous answer 'more attractive', complete a cut-off line (like 'including?'), or reformat it, rewrite and complete the previous answer cleanly using the PDF Context.\n"
                        f"5. {style_rule}\n"
                        "6. Match the user's language (English, Roman Urdu, Urdu, or Sindhi) while keeping mining technical terms accurate.\n"
                        "7. Always cite the exact Source file and Page number(s) at the end."
                    )

                    res = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {
                                "role": "user",
                                "content": (
                                    f"Standalone Topic: {standalone_query}\n\n"
                                    f"Recent Chat History:\n{recent_history}\n\n"
                                    f"PDF Context:\n{context_text}\n\n"
                                    f"User Request: {user_query}"
                                )
                            }
                        ],
                        model=selected_model,
                        temperature=0.0,
                        max_tokens=1500 if wants_detail else 900
                    )
                    reply = res.choices[0].message.content.strip()
                    if not reply:
                        reply = "I don't know about it."
                    st.markdown(reply)

                    if sources and "I don't know about it" not in reply:
                        with st.expander("📚 Verified PDF Reference Pages"):
                            for s in sources:
                                st.markdown(f"- `{s}`")

            except Exception as e:
                reply = f"⚠️ Engine error: {e}"
                st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
