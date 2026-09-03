from typing import Any, Dict, List
import streamlit as st

from backend.core import run_llm


# def format_sources(context_docs: List[Any]) -> List[str]:
#     # Fixed: Correct default value syntax meta.get("source", "Unknown")
#     return [
#         str(meta.get("source", "Unknown"))
#         for doc in (context_docs or [])
#         if (meta := (getattr(doc, "metadata", None) or {})) is not None
#     ]

def format_sources(context_docs: List[Any]) -> List[str]:
    sources = []
    for doc in context_docs or []:
        metadata = getattr(doc, "metadata", {}) or {}
        
        # Search across common metadata keys
        source = (
            metadata.get("source")
            or metadata.get("url")
            or metadata.get("title")
            or metadata.get("link")
        )
        
        if source:
            sources.append(str(source))
        elif metadata:
            # Fallback: display available metadata keys if "source" isn't found
            sources.append(str(metadata))
            
    return sources if sources else ["No Metadata Available"]


st.set_page_config(
    page_title="LangChain Documentation Helper",
    page_icon="🤖",
    layout="centered",
)

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear Session", use_container_width=True):
        # Fixed: Fixed typo 'mesages' -> 'messages'
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! What can I help you with today about LangChain?",
            "sources": ["www.langchain.com"],
        }
    ]

# Render existing chat message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Fixed: Changed msg.get[...] to msg.get(...)
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# Chat Input & Assistant Response Loop
if prompt := st.chat_input("Ask a question about LangChain..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate & render assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = run_llm(prompt)
            answer = res["answer"]
            # Extract flattened list of document artifacts from nested tool output list
            raw_docs = [doc for sublist in res.get("context", []) for doc in sublist]
            sources = format_sources(raw_docs)            

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in set(sources):  # Deduplicate sources for clean display
                        st.markdown(f"- {s}")

            # Save assistant response to session state
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )