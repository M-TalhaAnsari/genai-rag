import os
import hashlib
import tempfile
import streamlit as st
from typing import List, Dict

# Your custom imports remain exactly the same
from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from config import constants, settings
from utils.logging import logger

EXAMPLES = {
    "Google 2024 Environmental Report": {
        "question": "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022. Also retrieve regional average CFE in Asia pacific in 2023",
        "file_paths": ["examples/google-2024-environmental-report.pdf"]
    },
    "DeepSeek-R1 Technical Report": {
        "question": "Summarize DeepSeek-R1 model's performance evaluation on all coding tasks against OpenAI o1-mini model",
        "file_paths": ["examples/DeepSeek Technical Report.pdf"]
    }
}

# --- Backend Initialization ---
# st.cache_resource ensures these are only initialized once per server run
@st.cache_resource
def load_components():
    return DocumentProcessor(), RetrieverBuilder(), AgentWorkflow()

def _get_file_hashes(file_paths: List[str]) -> frozenset:
    """Generate SHA-256 hashes for a list of file paths."""
    hashes = set()
    for path in file_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                hashes.add(hashlib.sha256(f.read()).hexdigest())
    return frozenset(hashes)

def main():
    # --- Page Config ---
    st.set_page_config(page_title="DocChat 🐥", page_icon="🐥", layout="wide")
    
    # --- State Initialization ---
    if "file_hashes" not in st.session_state:
        st.session_state.file_hashes = frozenset()
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    processor, retriever_builder, workflow = load_components()

    # --- UI Layout: Sidebar ---
    with st.sidebar:
        st.title("DocChat 🐥")
        st.caption("Powered by Docling & LangGraph")
        st.divider()
        
        st.header("1. Document Setup 📂")
        
        # Example Selector
        example_options = ["-- Custom Upload --"] + list(EXAMPLES.keys())
        selected_mode = st.selectbox("Choose Input Method", example_options)
        
        uploaded_files = []
        current_file_paths = []

        if selected_mode == "-- Custom Upload --":
            # Streamlit accepts file extensions without the dot (e.g., 'pdf', not '.pdf')
            # Stripping dots dynamically in case your constants have them
            allowed = [ext.replace(".", "") for ext in constants.ALLOWED_TYPES]
            uploaded_files = st.file_uploader(
                "Upload your documents", 
                type=allowed, 
                accept_multiple_files=True
            )
            
            if uploaded_files:
                # Save uploaded files temporarily so processor can access them by path
                temp_dir = tempfile.mkdtemp()
                for uf in uploaded_files:
                    temp_path = os.path.join(temp_dir, uf.name)
                    with open(temp_path, "wb") as f:
                        f.write(uf.getbuffer())
                    current_file_paths.append(temp_path)
        else:
            st.success(f"Loaded: {selected_mode}")
            current_file_paths = EXAMPLES[selected_mode]["file_paths"]
            
            # Button to auto-fill the example question
            if st.button("Use Example Question 📝"):
                st.session_state.question_input = EXAMPLES[selected_mode]["question"]

    # --- Document Processing Logic ---
    if current_file_paths:
        current_hashes = _get_file_hashes(current_file_paths)
        
        # If hashes changed or retriever is empty, rebuild it
        if st.session_state.retriever is None or current_hashes != st.session_state.file_hashes:
            with st.spinner("Processing documents and building index... ⚙️"):
                try:
                    chunks = processor.process(current_file_paths)
                    st.session_state.retriever = retriever_builder.build_hybrid_retriever(chunks)
                    st.session_state.file_hashes = current_hashes
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                    logger.error(f"Processing error: {str(e)}")

    # --- UI Layout: Main Content ---
    st.markdown("<h2 style='text-align: center;'>Ask questions about your documents ✨</h2>", unsafe_allow_html=True)
    
    # We use a container to manage the input form
    with st.container():
        question = st.text_area(
            "❓ Enter your question below:", 
            value=st.session_state.question_input, 
            height=100
        )
        
        # Placing submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_pressed = st.button("Submit 🚀", type="primary", use_container_width=True)

    st.divider()

    # --- Query Execution ---
    if submit_pressed:
        if not current_file_paths:
            st.warning("⚠️ Please upload documents or select an example from the sidebar first.")
        elif not question.strip():
            st.warning("⚠️ Please enter a question to ask.")
        elif st.session_state.retriever is None:
            st.error("⚠️ Documents are still processing. Please try again in a moment.")
        else:
            with st.spinner("Agents are researching and drafting the answer... 🐥"):
                try:
                    result = workflow.full_pipeline(
                        question=question,
                        retriever=st.session_state.retriever
                    )
                    
                    # Display Answer
                    st.markdown("### 🐥 Answer")
                    st.info(result.get("draft_answer", "No answer generated."))
                    
                    # Hide Verification Report in an expandable section
                    with st.expander("✅ Verification Report (Click to expand)"):
                        st.write(result.get("verification_report", "No verification report provided."))
                        
                except Exception as e:
                    st.error(f"❌ Error during workflow execution: {e}")
                    logger.error(f"Workflow error: {str(e)}")

if __name__ == "__main__":
    main()