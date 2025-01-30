import streamlit as st
from streamlit import session_state
import time
import base64
import os
from vector import ManageEmbeddings  # Import the ManageEmbeddings class
from chatbot import ChatbotManager  # Import the ChatbotManager class
import pandas as pd
from docx import Document

# Function to display the PDF of a given file
def displayPDF(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def displayText(file):
    content = file.read().decode("utf-8")
    st.text_area("Text Content", value=content, height=300)


# Function to display .docx files
def displayDocx(file):
    doc = Document(file)
    content = "\n".join([para.text for para in doc.paragraphs if para.text])
    st.text_area("Document Content", value=content, height=300)


# Function to display CSV files
def displayCSV(file):
    df = pd.read_csv(file)  # Read CSV with semicolon delimiter
    st.dataframe(df)  # Display the dataframe in Streamlit

# Initialize session_state variables
if 'temp_file_path' not in st.session_state:
    st.session_state['temp_file_path'] = None

if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Set the page configuration
st.set_page_config(
    page_title="DocBuddy",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Navigation
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("### RAG | LLAMA3.2 | QDRANT")
    st.markdown("---")

    # Navigation menu as radio buttons
    choice = st.selectbox("Navigate", ["Home", "Chatbot", "Contact"])

# Home Page
if choice == "Home":
    st.title("DocBuddy App")
    st.markdown("""
    Welcome to the **Document Buddy App**.

    Built using an open-source stack (Llama 3.2, BGE Embeddings, and Qdrant running locally within a Docker container).

    - **Upload Documents**: Easily upload your documents.
    - **Summarize**: Get concise summaries of your documents.
    - **Chat**: Interact with your documents through our intelligent chatbot.

    Enhance your document management experience with DocBuddy!
    """)

# Chatbot Page
elif choice == "Chatbot":
    st.title("DocBuddy: Your Personal Document Assistant")
    st.markdown("---")

    # Upload Section
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "docx", "pptx", "csv"])
    if uploaded_file:
        st.success("File uploaded successfully!")
        st.markdown(f"**Filename:** {uploaded_file.name}")
        st.markdown(f"**File Size:** {uploaded_file.size} bytes")

        # Display based on file type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".pdf":
            st.markdown("### PDF Preview")
            displayPDF(uploaded_file)
        elif file_extension == ".txt":
            st.markdown("### File Content Preview")
            displayText(uploaded_file)
        elif file_extension == ".docx":
            st.markdown("### Document Content Preview")
            displayDocx(uploaded_file)
        elif file_extension == ".csv":
            st.markdown("### CSV Content Preview")
            displayCSV(uploaded_file)
        else:
            st.warning("Preview not available for this file type.")

        # Save to session state
        temp_file_path = "temp" + file_extension
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state['temp_file_path'] = temp_file_path


    # Embeddings Section
    st.header("Create Embeddings")
    create_embeddings = st.button("Create Embeddings")
    if create_embeddings:
        if st.session_state['temp_file_path'] is None:
            st.warning("Please upload a document first.")
        else:
            try:
                embeddings_manager = ManageEmbeddings(
                    model_name="BAAI/bge-small-en",
                    device="cpu",
                    encode_kwargs={"normalize_embeddings": True},
                    qdrant_host="http://localhost:6333",
                    db_name="Vector_Database"
                )

                with st.spinner("Clearing old embeddings..."):
                    # result = embeddings_manager.clear_existing_embeddings()
                    time.sleep(2)
                # st.success(result)

                with st.spinner("Creating new embeddings..."):
                    embed_result = embeddings_manager.embed(st.session_state['temp_file_path'])
                    time.sleep(1)
                st.success(embed_result)

                # Initialize chatbot
                if st.session_state['chatbot_manager'] is None:
                    st.session_state['chatbot_manager'] = ChatbotManager(
                        model_name="BAAI/bge-small-en",
                        device="cpu",
                        encode_kwargs={"normalize_embeddings": True},
                        llm_model="llama3.2:3b",
                        llm_temperature=0.6,
                        qdrant_host="http://localhost:6333",
                        db_name="Vector_Database"
                    )

            except Exception as e:
                st.error(f"Error: {e}")

    # Chatbot Section
    st.header("Chat with Document")
    if st.session_state['chatbot_manager'] is None:
        st.info("Please upload a document and create embeddings to start chatting.")
    else:
        for msg in st.session_state['messages']:
            st.chat_message(msg['role']).markdown(msg['content'])

        # User input
        if user_input := st.chat_input("Type your message here..."):
            st.chat_message("user").markdown(user_input)
            st.session_state['messages'].append({"role": "user", "content": user_input})

            with st.spinner("Responding..."):
                try:
                    answer = st.session_state['chatbot_manager'].get_response(user_input)
                    time.sleep(1)
                except Exception as e:
                    answer = f"An error occurred: {e}"

            st.chat_message("assistant").markdown(answer)
            st.session_state['messages'].append({"role": "assistant", "content": answer})

# Contact Page
elif choice == "Contact":
    st.title("Contact Me")
    st.markdown("""
    I'd love to hear from you! Whether you have a question, feedback, or want to contribute, feel free to reach out.

    - **Email:** [guptaanakull@gmail.com](mailto:guptaanakull@gmail.com)
    - **GitHub:** [Contribute on GitHub](https://github.com/Nakul2401)
    """)
