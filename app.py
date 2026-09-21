import streamlit as st

from research_agent import run_research

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

st.title("🔎 AI Research Agent")
st.caption("A single CrewAI agent that searches the web and writes you a report. Powered by Groq.")

# The key comes only from Streamlit secrets.
api_key = st.secrets.get("GROQ_API_KEY", "")

if not api_key:
    st.error(
        "No Groq API key found.\n\n"
        "Add it under **Manage app → Settings → Secrets** in Streamlit Cloud as:\n\n"
        "```toml\nGROQ_API_KEY = \"gsk_your_key_here\"\n```"
    )
    st.stop()

with st.sidebar:
    st.header("Settings")
    st.success("Groq API key loaded")
    max_results = st.slider("Search results per query", 3, 10, 5)
    temperature = st.slider("Creativity", 0.0, 1.0, 0.4, 0.1,
                            help="Lower is more factual, higher is more speculative.")
    st.divider()
    st.caption("Model: `openai/gpt-oss-120b`  \nSearch: DuckDuckGo (free)")

topic = st.text_input("Research topic", placeholder="e.g. The impact of AI tutors on student outcomes")

if st.button("Run Research", type="primary", use_container_width=True):
    if not topic.strip():
        st.error("Please enter a topic to research.")
    else:
        with st.spinner("Searching the web and writing your report. This usually takes 30-90 seconds..."):
            try:
                report = run_research(topic.strip(), api_key, max_results, temperature)
                st.session_state["report"] = report
                st.session_state["topic"] = topic.strip()
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")

if "report" in st.session_state:
    st.divider()
    st.markdown(st.session_state["report"])
    safe_name = "".join(c if c.isalnum() else "_" for c in st.session_state["topic"])[:50]
    st.download_button(
        "Download report (.md)",
        data=st.session_state["report"],
        file_name=f"{safe_name}_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
