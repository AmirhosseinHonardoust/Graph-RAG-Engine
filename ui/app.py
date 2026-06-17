import os

import requests
import streamlit as st

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
# You can override the API URL via environment variable or in the sidebar.
DEFAULT_API = os.environ.get("GRAPH_RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Graph RAG MVP", layout="wide")

# Sidebar configuration
st.sidebar.title("Settings")
api_url = st.sidebar.text_input(
    "Backend API URL", value=DEFAULT_API, help="Example: http://localhost:8000"
)
if st.sidebar.button("Check API health"):
    try:
        r = requests.get(f"{api_url}/health", timeout=10)
        st.sidebar.success(f"API OK: {r.json()}")
    except Exception as e:
        st.sidebar.error(f"API not reachable: {e}")

st.title("Graph-Powered Recommendation & Reasoning (MVP)")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def post_json(url: str, payload: dict, timeout: int = 60):
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_json(url: str, timeout: int = 60):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔎 Ask", "🧭 Recommend"])

# ----------------------------- ASK ---------------------------------
with tab1:
    st.subheader("Ask a question about the corpus")
    q = st.text_input("Question", value="What is FAISS?")
    answer_mode = st.selectbox(
        "Answer mode",
        options=["extractive", "llm"],
        index=0,
        help=(
            "Extractive mode returns retrieved source passages directly. "
            "LLM mode asks the backend to synthesize a grounded answer when "
            "optional LLM credentials are configured."
        ),
    )
    ask_col1, ask_col2 = st.columns([1, 3])

    with ask_col1:
        if st.button("Ask", use_container_width=True):
            if not q.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking…"):
                    try:
                        data = post_json(f"{api_url}/ask", {"question": q, "mode": answer_mode})
                    except Exception as e:
                        st.error(f"Error calling /ask: {e}")
                    else:
                        st.markdown("### Answer")
                        st.caption(f"Answer mode: `{data.get('answer_mode', answer_mode)}`")

                        llm_error = data.get("llm_error")
                        if llm_error:
                            st.warning(f"LLM mode fell back to extractive mode: {llm_error}")

                        st.markdown(data.get("answer", "_No answer returned_"))

                        citations = data.get("citations", [])
                        if citations:
                            with st.expander("Citations"):
                                for c in citations:
                                    title = c.get("doc_title", "Untitled")
                                    url = c.get("url", "")
                                    st.markdown(f"- [{title}]({url})")

                        paths = data.get("paths", [])
                        if paths:
                            with st.expander("Why these? (graph paths)"):
                                st.json(paths)

    with ask_col2:
        st.info(
            "Tip: Try questions that relate concepts across docs, e.g. "
            "`How do embeddings relate to FAISS in modern RAG?`"
        )
        st.markdown(
            "**Mode guide:** `extractive` is source-first and always available. "
            "`llm` requires the backend LLM configuration and will safely "
            "fall back when unavailable."
        )

# --------------------------- RECOMMEND ------------------------------
with tab2:
    st.subheader("Get similar documents")
    try:
        docs = get_json(f"{api_url}/docs_list").get("items", [])
    except Exception as e:
        st.error(f"Error calling /docs_list: {e}")
        docs = []

    if not docs:
        st.warning("No documents found. Did you run the ingest step?")
        st.code("python -m ingest.ingest_docs", language="bash")
    else:
        options = {d["title"]: d["doc_id"] for d in docs}
        title = st.selectbox("Choose a document", list(options.keys()))
        if st.button("Recommend similar"):
            with st.spinner("Finding related documents…"):
                try:
                    rec = post_json(f"{api_url}/recommend", {"doc_id": options[title]})
                except Exception as e:
                    st.error(f"Error calling /recommend: {e}")
                else:
                    items = rec.get("items", [])
                    if not items:
                        st.info("No recommendations found.")
                    else:
                        for item in items:
                            doc_title = item.get("title", "Untitled")
                            score = item.get("score", 0.0)
                            url = item.get("url", "")
                            pr = item.get("pagerank", 0.0)
                            st.markdown(
                                f"- **[{doc_title}]({url})**  \n"
                                f"  Score: `{score:.3f}` · PageRank: `{pr:.3f}`"
                            )

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.caption(
    "MVP uses FAISS for vector search and an in-memory graph for reasoning. "
    "Swap in Neo4j and an LLM for richer answers."
)
