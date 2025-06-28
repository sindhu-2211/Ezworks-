# app.py
import os
import re
import json
import string
from datetime import datetime
from typing import List, Tuple, Optional, Set

import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from openai import OpenAI
import difflib

# ────────────────────────────────────────────────
# ⚠️  HARD‑CODED OPENAI KEY — replace for production
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ────────────────────────────────────────────────

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
summarizer  = pipeline("summarization",       model="facebook/bart-large-cnn")
embeddings  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

NUM_Q = 3   # number of quiz questions

# ─────────── Helper functions ───────────
def extract_text(uploaded) -> str:
    ext = uploaded.name.split(".")[-1].lower()
    if ext == "pdf":
        pdf = PdfReader(uploaded)
        return "\n".join(p.extract_text() or "" for p in pdf.pages)
    return uploaded.read().decode("utf-8", errors="ignore")

def chunk_text(text: str, size=500) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paras:
        if len(p) <= size:
            chunks.append(p)
        else:
            sents, buf = re.split(r"(?<=[.!?]) +", p), ""
            for s in sents:
                if len(buf) + len(s) <= size:
                    buf += " " + s
                else:
                    if buf: chunks.append(buf.strip())
                    buf = s
            if buf: chunks.append(buf.strip())
    return chunks

def build_store(chunks):
    return FAISS.from_texts(chunks, embedding=embeddings)

def gen_summary(text: str) -> str:
    try:
        return summarizer(text[:4000], max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
    except Exception:
        # fallback first 3 sentences
        return " ".join(re.split(r"(?<=[.!?]) +", text)[:3])

# ─────────── QA functions ───────────
def ask_question(q: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    db = st.session_state.vector_store
    docs = db.similarity_search(q, k=5)
    best_a, best_ctx, best_s = None, None, 0.0
    for d in docs:
        try:
            r = qa_pipeline(question=q, context=d.page_content)
            if r["score"] > best_s:
                best_s, best_a = r["score"], r["answer"]
                best_ctx = d.page_content.replace(r["answer"], f"<mark>{r['answer']}</mark>", 1)
        except Exception:
            pass
    return best_a, best_ctx, best_s

def _add_history(q, a, ctx, conf):
    st.session_state.chat_history.append(
        {
            "question": q,
            "answer": a,
            "source": ctx,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "confidence": conf,
        }
    )

def handle_multipart(q: str) -> bool:
    parts = re.split(r"\band\b|\bor\b", q, flags=re.IGNORECASE)
    if len(parts) <= 1:
        return False
    st.info("Multi‑part question detected—breaking it down.")
    for i, part in enumerate(parts, 1):
        part = part.strip()
        if not part.endswith("?"):
            part += "?"
        st.write(f"*Part {i}:* {part}")
        a, ctx, conf = ask_question(part)
        if a:
            _add_history(part, a, ctx, conf)
            st.write(f"*Answer:* {a}")
            if ctx:
                st.markdown(ctx, unsafe_allow_html=True)
        else:
            st.warning("No answer found.")
    return True

# ─────────── Quiz functions ───────────
def generate_quiz(text: str, n=NUM_Q):
    prompt = (
        f"Generate exactly {n} question‑answer pairs as JSON "
        f"[{{'question': '...', 'answer': '...'}}, ...] based on the text below.\n\n{text[:4000]}"
    )
    raw = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.7
    ).choices[0].message.content
    raw = re.sub(r"^```json|```$", "", raw, flags=re.I).strip()
    try:
        data = json.loads(raw)
        return [d for d in data if "question" in d and "answer" in d][:n]
    except json.JSONDecodeError:
        return None

# ─────────── Evaluation helpers ───────────
def _clean(t): return " ".join(t.lower().translate(str.maketrans("", "", string.punctuation)).split())
_stop = {"the","and","or","but","in","on","at","to","for","of","with","by","is","are","was","were","be","been",
         "have","has","had","do","does","did","will","would","could","should","may","might","can","this","that",
         "these","those","a","an","as","it","its"}
def _keywords(t): return {w for w in _clean(t).split() if len(w)>=3 and w not in _stop}
def _numbers(t):  return set(re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", t or ""))

def eval_answer(user: str, correct: str):
    if not user.strip(): return 0, "❌ No answer"
    if correct.startswith("Refer") or not correct.strip(): return 1, "✅ Accepted (no ref answer)"
    uc, cc = _clean(user), _clean(correct)
    sim = difflib.SequenceMatcher(None, uc, cc).ratio()
    kw = len(_keywords(user)&_keywords(correct))/max(1,len(_keywords(correct)))
    num = len(_numbers(user)&_numbers(correct))/max(1,len(_numbers(correct))) if _numbers(correct) else 1
    phr = lambda w:{' '.join(w[i:i+l]) for i in range(len(w)) for l in (2,3) if i+l<=len(w)}
    ph = len(phr(uc.split())&phr(cc.split()))/max(1,len(phr(cc.split())))
    score = 0.4*kw + 0.3*num + 0.2*ph + 0.1*sim
    if score>=.8: return 1, "✅ Excellent"
    if score>=.6: return .8, f"🟡 Good – ref: {correct}"
    if score>=.4: return .5, f"🔶 Partial – ref: {correct}"
    if score>=.2: return .2, f"🔸 Few points – ref: {correct}"
    return 0, f"❌ Incorrect – ref: {correct}"

# ─────────── Streamlit UI ───────────
st.set_page_config("📄 Doc Chat & Challenge", layout="wide")
st.title("📄 Document Chat & Challenge")

# Initial session state
for k,v in {"summary":"", "chat_history":[], "vector_store":None,
            "doc_text":"", "quiz":None, "last_filename":None}.items():
    if k not in st.session_state: st.session_state[k]=v

# Upload & process
uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf","txt"])
if uploaded and st.button("Process Document"):
    if uploaded.name != st.session_state.last_filename:           # new file
        txt = extract_text(uploaded)
        st.session_state.doc_text     = txt
        st.session_state.vector_store = build_store(chunk_text(txt))
        st.session_state.summary      = gen_summary(txt)
        st.session_state.chat_history = []     # reset history only now
        st.session_state.quiz         = None
        st.session_state.last_filename= uploaded.name
        st.success("New document processed — history cleared.")
    else:
        st.info("Same document re‑processed; history preserved.")

# ── Sidebar: always show chat history ──
with st.sidebar:
    st.header("💬 Chat History")
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:      # chronological
            with st.expander(f"{chat['timestamp']} – {chat['question'][:40]}"):
                st.write(chat["answer"])
                if chat["source"]:
                    st.markdown(chat["source"], unsafe_allow_html=True)
    else:
        st.info("No history yet.")
    if st.button("Clear History"):
        st.session_state.chat_history = []

# Main interface after summary
if st.session_state.summary:
    st.subheader("📑 Summary")
    st.info(st.session_state.summary)

    mode = st.radio("Choose mode", ["Ask Anything", "Challenge Me"], horizontal=True)

    # ── Ask Anything Mode ──
    if mode=="Ask Anything":
        q = st.text_input("Your question")
        if q:
            if not handle_multipart(q):
                a, ctx, conf = ask_question(q)
                if a:
                    _add_history(q,a,ctx,conf)
                    st.subheader("✅ Answer")
                    st.write(a)
                    if ctx:
                        st.subheader("📄 Source Context")
                        st.markdown(ctx, unsafe_allow_html=True)
                    st.caption(f"Confidence: {conf:.2f}")
                else:
                    st.warning("No confident answer found.")

    # ── Challenge Me Mode ──
    else:
        st.subheader("🏆 Challenge Me")
        if not st.session_state.quiz:
            if st.button("Generate Questions"):
                quiz = generate_quiz(st.session_state.doc_text, NUM_Q)
                if quiz:
                    st.session_state.quiz = quiz
                    st.success("Questions generated!")
                else:
                    st.error("Could not generate quiz, try again.")
        else:
            inputs=[]
            for i,qa in enumerate(st.session_state.quiz):
                st.markdown(f"**Q{i+1}: {qa['question']}**")
                inputs.append(st.text_area("Your answer", key=f"ans_{i}"))
            if st.button("Submit Answers"):
                total=0
                for inp,qa in zip(inputs,st.session_state.quiz):
                    st.markdown(f"**Q:** {qa['question']}")
                    st.markdown(f"*Your answer:* {inp if inp else '—'}")
                    sc,fb=eval_answer(inp,qa['answer'])
                    total+=sc
                    st.markdown(fb)
                    st.write("---")
                pct=total/len(st.session_state.quiz)*100
                st.markdown(f"## 🎯 Final Score: {total:.1f}/{len(st.session_state.quiz)} ({pct:.1f}%)")
                if st.button("🔥 Retake / New Quiz"): st.session_state.quiz=None
