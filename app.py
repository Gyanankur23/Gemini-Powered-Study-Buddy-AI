# =============================================================================
# PROJECT 2: Study Buddy — RAG (Retrieval Augmented Generation) Chatbot
# Workshop: Building with the Free Gemini API
# =============================================================================
# WHAT THIS PROJECT TEACHES:
#   1. What RAG is — grounding AI answers in YOUR documents
#   2. How to extract text from a PDF using pypdf
#   3. How to inject document content into a prompt (Context Injection)
#   4. How to manage chat history in Streamlit using session_state
#   5. How to handle API rate limits gracefully with retry logic
#
# HOW TO RUN:
#   1. pip install -r requirements.txt
#   2. streamlit run study_buddy.py
# =============================================================================

import time
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

# ✅ FIXED: Using the current stable model string from Google's official docs
MODEL_NAME = "gemini-2.5-flash"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 15

SYSTEM_PROMPT = """
You are a focused and helpful Study Buddy assistant.
Your job is to answer questions ONLY based on the document content provided to you.
Rules you must always follow:
  - If the answer is in the document, provide a clear and concise answer.
  - If the answer is NOT in the document, say: "I couldn't find that in the document."
  - Never make up information or use outside knowledge.
  - Keep answers student-friendly: clear, simple, and to the point.
"""


# =============================================================================
# SECTION 2: PDF PROCESSING
# =============================================================================

def extract_text_from_pdf(uploaded_file) -> str:
    """
    Reads a PDF file and extracts all its text content as a single string.

    This is the core of RAG — we pull raw text out of the document
    so the AI can reference it when answering questions.

    Args:
        uploaded_file: The file object from Streamlit's file_uploader widget.

    Returns:
        A single string containing all the text from the PDF.
    """
    try:
        reader = PdfReader(uploaded_file)
        extracted_text = ""

        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text()

            # Some pages (e.g. image-only pages) may return None — skip those
            if page_text:
                extracted_text += f"\n--- Page {page_number + 1} ---\n"
                extracted_text += page_text

        return extracted_text

    except Exception as e:
        st.error(f"❌ Failed to read PDF: {e}")
        return ""


# =============================================================================
# SECTION 3: GEMINI SETUP
# =============================================================================

@st.cache_resource
def load_model(api_key: str):
    """
    Configures the Gemini API and returns a GenerativeModel instance.

    @st.cache_resource means this runs ONCE per session — efficient and
    avoids re-authenticating on every message send.

    Args:
        api_key: The user's Gemini API key.

    Returns:
        A configured GenerativeModel object.
    """
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
    )

    return model


# =============================================================================
# SECTION 4: CORE RAG FUNCTION
# =============================================================================

def ask_question(model, question: str, document_text: str, chat_history: list) -> str:
    """
    Sends a question to Gemini, grounded by the PDF document content.

    HOW RAG WORKS HERE (teach this to students):
      Normal chatbot:  User Question → AI Answer (uses training data)
      RAG chatbot:     User Question + Document Content → AI Answer (uses YOUR doc)

    The document text is injected directly into the prompt so Gemini
    only references that content — not its general training knowledge.

    Args:
        model:         The loaded GenerativeModel instance.
        question:      The user's current question.
        document_text: The full extracted text from the uploaded PDF.
        chat_history:  List of previous (question, answer) tuples for memory.

    Returns:
        The AI's answer as a plain string.
    """

    # --- Build the Prompt ---
    # This is "Augmentation" — we inject the document into every prompt
    prompt = f"""
Here is the document you should use to answer questions:

=== DOCUMENT START ===
{document_text}
=== DOCUMENT END ===

Based ONLY on the document above, please answer this question:
{question}
"""

    # --- Build Conversation History ---
    # Alternating user/model messages so the AI remembers earlier turns
    messages = []

    for past_question, past_answer in chat_history:
        messages.append({"role": "user",  "parts": [past_question]})
        messages.append({"role": "model", "parts": [past_answer]})

    # Add the current question as the latest message
    messages.append({"role": "user", "parts": [prompt]})

    # --- Send to Gemini with Retry Logic ---
    # Free tier can hit rate limits — this retries automatically
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(messages)
            return response.text

        except Exception as e:
            error_message = str(e)

            if "429" in error_message or "quota" in error_message.lower():
                if attempt < MAX_RETRIES:
                    st.warning(
                        f"⏳ Rate limit hit. Waiting {RETRY_DELAY_SECONDS}s "
                        f"before retry {attempt}/{MAX_RETRIES - 1}..."
                    )
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    return (
                        "❌ Rate limit exceeded after multiple retries. "
                        "Please wait a minute and try again. "
                        "Tip: Each student should use their own API key!"
                    )
            else:
                return f"❌ Unexpected error: {error_message}"


# =============================================================================
# SECTION 5: STREAMLIT UI
# =============================================================================

def main():
    """Main function — renders the full Streamlit application."""

    st.set_page_config(
        page_title="Study Buddy — RAG Chatbot",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Study Buddy")
    st.caption("RAG Chatbot | Workshop Project 2")
    st.markdown(
        "Upload any PDF (textbook chapter, notes, research paper) and ask "
        "questions about it. The AI answers **only** from your document — "
        "no hallucinations, no outside knowledge."
    )
    st.divider()

    # -------------------------------------------------------------------------
    # SIDEBAR
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Setup")

        api_key = st.text_input(
            label="Gemini API Key",
            type="password",
            placeholder="Paste your key here...",
            help="Get a free key at: aistudio.google.com",
        )

        st.divider()

        st.subheader("📄 Upload Your Document")
        uploaded_file = st.file_uploader(
            label="Choose a PDF file",
            type="pdf",
            help="The AI will only answer from this document.",
        )

        if uploaded_file:
            with st.spinner("Reading your PDF..."):
                # Only re-process if a new file is uploaded
                if "document_text" not in st.session_state or \
                   st.session_state.get("last_file") != uploaded_file.name:

                    document_text = extract_text_from_pdf(uploaded_file)

                    if document_text:
                        st.session_state.document_text = document_text
                        st.session_state.last_file = uploaded_file.name
                        st.session_state.chat_history = []
                        st.success(
                            f"✅ PDF loaded! ({len(document_text):,} characters extracted)"
                        )
                    else:
                        st.error("Could not extract text. Is this a scanned PDF?")

        st.divider()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.info(
            "💡 **How RAG works:**\n\n"
            "Your PDF text is injected directly into every prompt. "
            "Gemini reads it fresh each time and answers only from that content. "
            "No vector database needed — this is the simplest form of RAG!"
        )

    # -------------------------------------------------------------------------
    # MAIN CHAT AREA
    # -------------------------------------------------------------------------

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Guard: no document uploaded yet
    if "document_text" not in st.session_state:
        st.info("👈 Start by entering your API key and uploading a PDF in the sidebar.")
        return

    # Guard: no API key
    if not api_key:
        st.warning("🔑 Please enter your Gemini API Key in the sidebar.")
        return

    # Display past messages
    for question, answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)

    # Chat input pinned to bottom
    user_question = st.chat_input("Ask something about your document...")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                model = load_model(api_key)
                answer = ask_question(
                    model=model,
                    question=user_question,
                    document_text=st.session_state.document_text,
                    chat_history=st.session_state.chat_history,
                )
            st.markdown(answer)

        st.session_state.chat_history.append((user_question, answer))


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
