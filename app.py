from __future__ import annotations
import io, json, os, re
from pathlib import Path
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from tutor_engine import Chunk, TfidfRetriever, chunk_text, extractive_answer, make_cloze_quiz

BASE = Path(__file__).parent
DEMO_PATH = BASE / "data" / "demo_notes.md"
st.set_page_config(page_title="AI Study Tutor", page_icon="📚", layout="wide")
st.markdown("""<style>
.block-container{max-width:1280px;padding-top:1.3rem}
.hero{padding:1.5rem;border-radius:18px;background:linear-gradient(120deg,#1d2451,#7357f6);color:white;margin-bottom:1rem}
.hero h1{margin:0 0 .3rem 0}.hero p{color:#e8e6ff;margin:0}
[data-testid="stMetric"]{background:#f6f6fc;border:1px solid #e9e8f7;padding:12px;border-radius:12px}
</style>""",unsafe_allow_html=True)

def split_markdown(text: str, filename: str) -> list[Chunk]:
    title=Path(filename).stem.replace("_"," ").title(); body=[]; out=[]
    for line in text.splitlines():
        match=re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$",line)
        if match:
            if "\n".join(body).strip(): out.extend(chunk_text("\n".join(body),filename,f"Section: {title}"))
            title=match.group(1).strip(); body=[]
        else: body.append(line)
    if "\n".join(body).strip(): out.extend(chunk_text("\n".join(body),filename,f"Section: {title}"))
    return out

def extract_upload(upload) -> list[Chunk]:
    name=upload.name; raw=upload.getvalue(); suffix=Path(name).suffix.lower()
    if suffix==".pdf":
        out=[]
        for page_no,page in enumerate(PdfReader(io.BytesIO(raw)).pages,start=1):
            out.extend(chunk_text(page.extract_text() or "",name,f"Page {page_no}"))
        return out
    text=raw.decode("utf-8",errors="replace")
    return split_markdown(text,name) if suffix in {".md",".markdown"} else chunk_text(text,name,"Text document")

def settings(key_ui: str):
    key=key_ui.strip() or os.getenv("OPENAI_API_KEY","")
    model=os.getenv("OPENAI_MODEL",""); base=os.getenv("OPENAI_BASE_URL","")
    try:
        key=key or str(st.secrets.get("OPENAI_API_KEY","")).strip()
        model=model or str(st.secrets.get("OPENAI_MODEL","gpt-4o-mini")).strip()
        base=base or str(st.secrets.get("OPENAI_BASE_URL","https://api.openai.com/v1")).strip()
    except Exception: pass
    return key,model or "gpt-4o-mini",base or "https://api.openai.com/v1"

def llm(messages, key, model, base):
    from openai import OpenAI
    response=OpenAI(api_key=key,base_url=base).chat.completions.create(model=model,messages=messages,temperature=.2)
    return (response.choices[0].message.content or "").strip()

def make_context(results):
    parts=[]; sources=[]
    for i,(chunk,score) in enumerate(results,1):
        parts.append(f"[S{i}] Source: {chunk.citation}\nRelevance: {score:.3f}\nPassage: {chunk.text}")
        sources.append({"id":f"S{i}","source":chunk.citation,"score":score,"text":chunk.text})
    return "\n\n".join(parts),sources

def answer(question,results,key,model,base):
    if not results: return "I couldn't find relevant passages in the loaded materials. Try rephrasing or upload notes about this topic.","retrieval"
    if not key: return extractive_answer(question,results),"source-only"
    context,_=make_context(results)
    system=("You are a careful study tutor. Answer only from the supplied source passages. If evidence is insufficient, say so. "
            "Explain clearly and cite factual claims with exact IDs such as [S1]. Use only source IDs in the context; never invent references.")
    try:
        return llm([{"role":"system","content":system},{"role":"user","content":f"QUESTION:\n{question}\n\nSOURCES:\n{context}\n\nAnswer with inline source IDs."}],key,model,base),"llm-rag"
    except Exception as exc:
        return extractive_answer(question,results)+f"\n\nLLM unavailable ({type(exc).__name__}); showing a source-only answer instead.","source-only-fallback"

def generate_quiz(results,count,key,model,base):
    if key and results:
        context,_=make_context(results)
        prompt=f"""Create {count} conceptual multiple-choice questions using only these source passages. Return JSON array with question, options (exactly 4 strings), answer (one exact option), explanation, source_id (S1 etc). Make plausible distractors and do not invent facts.\n\n{context}"""
        try:
            raw=llm([{"role":"system","content":"Return valid JSON only, no Markdown."},{"role":"user","content":prompt}],key,model,base)
            raw=re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$","",raw,flags=re.I)
            parsed=json.loads(raw); valid={f"S{i}" for i in range(1,len(results)+1)}; out=[]
            for q in parsed:
                opts=q.get("options",[]); ans=str(q.get("answer","")); sid=q.get("source_id","S1")
                if isinstance(opts,list) and len(opts)==4 and ans in opts:
                    if sid not in valid: sid="S1"
                    chunk=results[int(sid[1:])-1][0]
                    out.append({"question":str(q.get("question","Review the material.")),"options":[str(x) for x in opts],"answer":ans,"explanation":str(q.get("explanation","")),"source":chunk.citation,"source_text":chunk.text})
            if out: return out[:count],"Generated conceptual questions from retrieved source passages."
        except Exception as exc:
            return make_cloze_quiz(results,count),f"LLM quiz generation failed ({type(exc).__name__}); using source-derived cloze questions."
    return make_cloze_quiz(results,count),"No API key configured: these are source-derived fill-in-the-blank questions."

def render_sources(sources):
    if not sources: return
    with st.expander("View retrieved source passages",expanded=True):
        for s in sources:
            st.markdown(f"**[{s['id']}] {s['source']}** · relevance {s['score']:.3f}")
            st.write(s["text"]); st.divider()

st.markdown('<div class="hero"><h1>📚 AI Study Tutor</h1><p>Learn from your own notes with source-linked answers, practice quizzes, and study progress.</p></div>',unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## Your study space")
    source_mode=st.radio("Study material",["Built-in demo notes","Upload my documents"])
    uploads=st.file_uploader("Upload PDF, TXT, or Markdown notes",type=["pdf","txt","md","markdown"],accept_multiple_files=True) or [] if source_mode=="Upload my documents" else []
    top_k=st.slider("Passages to retrieve",2,8,5)
    min_score=st.slider("Minimum relevance score",0.0,0.5,0.0,0.05,help="TF-IDF scores are relevance signals, not probabilities.")
    st.markdown("---"); st.markdown("### Optional generative AI")
    key_ui=st.text_input("OpenAI API key",type="password",help="Optional; never save your key in GitHub.")
key,model,base=settings(key_ui)
try:
    if source_mode=="Built-in demo notes":
        chunks=split_markdown(DEMO_PATH.read_text(encoding="utf-8"),DEMO_PATH.name); summary="Built-in Materials Science demo notes"
    else:
        if not uploads: st.info("Upload notes in the sidebar or switch to built-in demo notes."); st.stop()
        total=0; chunks=[]
        for upload in uploads:
            total+=len(upload.getvalue())
            if total>35*1024*1024: st.error("Combined upload limit is 35 MB."); st.stop()
            extracted=extract_upload(upload)
            if not extracted: st.warning(f"No extractable text found in {upload.name}; scanned PDFs may need OCR.")
            chunks.extend(extracted)
        summary=", ".join(x.name for x in uploads)
    if not chunks: st.error("No searchable text found."); st.stop()
    signature=summary+"|"+"|".join(c.text for c in chunks)
    import hashlib
    signature=hashlib.sha256(signature.encode("utf-8",errors="replace")).hexdigest()
    if st.session_state.get("signature")!=signature:
        st.session_state.signature=signature; st.session_state.chunks=chunks
        st.session_state.retriever=TfidfRetriever(chunks); st.session_state.messages=[]
        st.session_state.quiz=[]; st.session_state.quiz_history=[]; st.session_state.quiz_submitted=False
except Exception as exc:
    st.error(f"Could not process study material: {exc}"); st.stop()

chunks=st.session_state.chunks; retriever=st.session_state.retriever
col1,col2,col3,col4=st.columns(4)
col1.metric("Documents",len(set(c.source for c in chunks))); col2.metric("Passages",len(chunks))
col3.metric("Retrieval","TF-IDF"); col4.metric("Answer mode","LLM + RAG" if key else "Source-only")
st.caption(f"Loaded: {summary}. Verify cited passages before relying on answers.")
tab_chat,tab_quiz,tab_progress,tab_docs=st.tabs(["💬 Ask your notes","📝 Practice quiz","📈 Study progress","📄 Document library"])
with tab_chat:
    st.subheader("Ask a question")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"]=="assistant": render_sources(message.get("sources",[]))
    question=st.chat_input("Ask something from your study notes…")
    if question:
        results=[x for x in retriever.search(question,top_k) if x[1]>=min_score]
        _,sources=make_context(results)
        st.session_state.messages.append({"role":"user","content":question})
        reply,mode=answer(question,results,key,model,base)
        st.session_state.messages.append({"role":"assistant","content":reply,"sources":sources,"mode":mode})
        st.rerun()
    with st.expander("Try a sample question"):
        st.markdown("- Compare BCC and FCC crystal structures.\n- Explain elastic versus plastic deformation.\n- What does tempering do to quenched steel?\n- How do dislocations affect strength?")
    if st.session_state.messages:
        st.download_button("Download conversation CSV",pd.DataFrame([{"role":m["role"],"message":m["content"]} for m in st.session_state.messages]).to_csv(index=False).encode(),"study_conversation.csv","text/csv")
with tab_quiz:
    st.subheader("Practice with your material")
    topic=st.text_input("Topic or learning objective",placeholder="e.g. crystal structures, heat treatment")
    count=st.slider("Number of questions",3,10,5,key="quiz_count")
    if st.button("Generate practice quiz",type="primary"):
        query=topic.strip() or "important concepts definitions mechanisms and comparisons"
        results=[x for x in retriever.search(query,max(top_k,count)) if x[1]>=min_score]
        quiz,note=generate_quiz(results,count,key,model,base)
        st.session_state.quiz=quiz; st.session_state.quiz_note=note; st.session_state.quiz_submitted=False; st.session_state.quiz_answers={}
    if st.session_state.get("quiz_note"): st.info(st.session_state.quiz_note)
    quiz=st.session_state.get("quiz",[])
    if quiz:
        with st.form("quiz_form"):
            for i,q in enumerate(quiz):
                st.markdown(f"**Question {i+1}.** {q['question']}")
                st.radio("Select answer",q["options"],key=f"answer_{i}",index=None,label_visibility="collapsed")
                st.caption("Source: "+q["source"])
            submitted=st.form_submit_button("Check answers",type="primary")
        if submitted:
            answers=[st.session_state.get(f"answer_{i}") for i in range(len(quiz))]
            score=sum(a==q["answer"] for a,q in zip(answers,quiz))
            st.session_state.quiz_answers={str(i):a for i,a in enumerate(answers)}
            st.session_state.quiz_submitted=True
            st.session_state.quiz_history.append({"attempt":len(st.session_state.quiz_history)+1,"score":score,"total":len(quiz),"percent":round(score/len(quiz)*100,1),"topic":topic or "General"})
        if st.session_state.get("quiz_submitted"):
            answers=st.session_state.quiz_answers
            st.metric("Your score",f"{sum(answers.get(str(i))==q['answer'] for i,q in enumerate(quiz))}/{len(quiz)}")
            for i,q in enumerate(quiz):
                if answers.get(str(i))==q["answer"]: st.success(f"Question {i+1}: Correct")
                else: st.error(f"Question {i+1}: Correct answer — {q['answer']}")
                st.write(q.get("explanation") or q.get("source_text",""))
        st.download_button("Download quiz and answers",pd.DataFrame([{"question":q["question"],"options":" | ".join(q["options"]),"answer":q["answer"],"source":q["source"]} for q in quiz]).to_csv(index=False).encode(),"study_quiz.csv","text/csv")
    else: st.info("Generate a quiz to start practising.")
with tab_progress:
    history=st.session_state.get("quiz_history",[])
    if not history: st.info("Complete a quiz to see session progress. Progress is not stored on a server.")
    else:
        hist=pd.DataFrame(history); a,b,c=st.columns(3)
        a.metric("Attempts",len(hist)); b.metric("Latest score",f"{hist.iloc[-1]['percent']:.1f}%"); c.metric("Average score",f"{hist['percent'].mean():.1f}%")
        st.dataframe(hist,use_container_width=True,hide_index=True); st.line_chart(hist.set_index("attempt")["percent"])
        st.download_button("Download progress CSV",hist.to_csv(index=False).encode(),"study_progress.csv","text/csv")
with tab_docs:
    rows=[{"Source":c.source,"Location":c.locator,"Chunk":c.chunk_id,"Characters":len(c.text),"Preview":c.text[:200].replace("\n"," ")} for c in chunks]
    df=pd.DataFrame(rows); st.dataframe(df,use_container_width=True,hide_index=True)
    selected=st.selectbox("Inspect a source",sorted(set(c.source for c in chunks)))
    for c in [x for x in chunks if x.source==selected][:12]:
        with st.expander(f"{c.locator} · passage {c.chunk_id+1}"): st.write(c.text)
    st.download_button("Download passage index",df.to_csv(index=False).encode(),"document_passage_index.csv","text/csv")
st.divider(); st.caption("AI Study Tutor · Source-grounded learning companion. Verify citations and generated quiz answers against the original materials.")
