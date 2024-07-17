import os
import pickle
import hashlib
import streamlit as st
import faiss
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize ChatGroq
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Create ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Function to perform vector embedding
def embed_pdfs(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = []

    for doc in docs:
        chunks = text_splitter.split_documents([doc])
        final_documents.extend(chunks)

    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors, final_documents

# Function to hash chunks for tracking
def hash_chunk(chunk):
    return hashlib.sha256(chunk.page_content.encode()).hexdigest()

# Function to delete a PDF
def delete_pdf(file_name):
    global uploaded_files_history, questions_history

    # Remove from history
    uploaded_files_history = [file for file in uploaded_files_history if file != file_name]
    with open("shared_storage/uploaded_files_history.pkl", "wb") as f:
        pickle.dump(uploaded_files_history, f)

    # Reload documents
    with open("shared_storage/documents.pkl", "rb") as f:
        documents = pickle.load(f)

    # Filter out documents related to the file
    documents = [doc for doc in documents if doc.metadata["source"] != file_name]
    with open("shared_storage/documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    # Recreate FAISS index
    vectors, _ = embed_pdfs(documents)
    faiss.write_index(vectors.index, "shared_storage/vectors.index")

# Function to rename a PDF
def rename_pdf(old_name, new_name):
    global uploaded_files_history

    # Update history
    uploaded_files_history = [new_name if file == old_name else file for file in uploaded_files_history]
    with open("shared_storage/uploaded_files_history.pkl", "wb") as f:
        pickle.dump(uploaded_files_history, f)

    # Reload documents
    with open("shared_storage/documents.pkl", "rb") as f:
        documents = pickle.load(f)

    # Update document metadata
    for doc in documents:
        if doc.metadata["source"] == old_name:
            doc.metadata["source"] = new_name

    with open("shared_storage/documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    # Recreate FAISS index
    vectors, _ = embed_pdfs(documents)
    faiss.write_index(vectors.index, "shared_storage/vectors.index")

# Function to update a PDF
def update_pdf(old_name, new_file):
    global uploaded_files_history

    # Remove old file and update history
    uploaded_files_history = [new_file.name if file == old_name else file for file in uploaded_files_history]
    with open("shared_storage/uploaded_files_history.pkl", "wb") as f:
        pickle.dump(uploaded_files_history, f)

    # Reload documents
    with open("shared_storage/documents.pkl", "rb") as f:
        documents = pickle.load(f)

    # Remove old document and add new document
    documents = [doc for doc in documents if doc.metadata["source"] != old_name]
    pdf_reader = PdfReader(new_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    new_doc = Document(page_content=text, metadata={"source": new_file.name})
    documents.append(new_doc)

    with open("shared_storage/documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    # Recreate FAISS index
    vectors, _ = embed_pdfs(documents)
    faiss.write_index(vectors.index, "shared_storage/vectors.index")

# Load previously uploaded files
uploaded_files_history = []
if os.path.exists("shared_storage/uploaded_files_history.pkl"):
    with open("shared_storage/uploaded_files_history.pkl", "rb") as f:
        uploaded_files_history = pickle.load(f)

# Load the questions history
questions_history = {}
if os.path.exists("shared_storage/questions_history.pkl"):
    with open("shared_storage/questions_history.pkl", "rb") as f:
        questions_history = pickle.load(f)

# Streamlit interface
st.title("University Study Assistant")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the section", ["Manage PDFs", "Ask Questions"])

if app_mode == "Manage PDFs":
    st.header("Upload, Manage, and Update PDFs")

    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

    if uploaded_files:
        st.write("Processing PDFs...")
        all_docs = []
        new_files = []

        for uploaded_file in uploaded_files:
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            doc = Document(page_content=text, metadata={"source": uploaded_file.name})
            all_docs.append(doc)
            new_files.append(uploaded_file.name)

        vectors, final_documents = embed_pdfs(all_docs)

        # Save vectors and documents in shared storage
        if not os.path.exists("shared_storage"):
            os.makedirs("shared_storage")

        faiss.write_index(vectors.index, "shared_storage/vectors.index")
        with open("shared_storage/documents.pkl", "wb") as f:
            pickle.dump(final_documents, f)

        uploaded_files_history.extend(new_files)
        with open("shared_storage/uploaded_files_history.pkl", "wb") as f:
            pickle.dump(uploaded_files_history, f)

        st.success("PDFs processed and vectors saved.")

    st.subheader("History of Uploaded PDFs")
    search_query = st.text_input("Search for a PDF")
    if uploaded_files_history:
        filtered_files = [file for file in uploaded_files_history if search_query.lower() in file.lower()]
        if filtered_files:
            for file_name in filtered_files:
                st.write(file_name)
                col1, col2, col3, col4 = st.columns([1, 2, 1, 2])
                with col1:
                    if st.button(f"Delete", key=f"delete_{file_name}"):
                        delete_pdf(file_name)
                        st.experimental_rerun()
                with col2:
                    new_name = st.text_input(f"Rename {file_name}", key=f"rename_{file_name}")
                with col3:
                    if st.button(f"Rename", key=f"rename_btn_{file_name}"):
                        if new_name:
                            rename_pdf(file_name, new_name)
                            st.experimental_rerun()
                with col4:
                    new_file = st.file_uploader(f"Update {file_name}", type=["pdf"], key=f"update_{file_name}")
                    if st.button(f"Update", key=f"update_btn_{file_name}"):
                        if new_file:
                            update_pdf(file_name, new_file)
                            st.experimental_rerun()
        else:
            st.write("No matching PDFs found.")
    else:
        st.write("No PDFs uploaded yet.")

elif app_mode == "Ask Questions":
    st.header("Query Interface")

    st.subheader("Ask a Question")
    question = st.text_input("Enter your question here:")

    if st.button("Get Answer"):
        # Load vectors and documents from shared storage
        index = faiss.read_index("shared_storage/vectors.index")
        with open("shared_storage/documents.pkl", "rb") as f:
            documents = pickle.load(f)

        docstore = InMemoryDocstore({i: doc for i, doc in enumerate(documents)})
        vectors = FAISS(embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), docstore=docstore, index=index, index_to_docstore_id={i: i for i in range(len(documents))})

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': question})
        answer = response['answer']

        # Track the question
        question_hash = hashlib.sha256(question.encode()).hexdigest()
        if question_hash not in questions_history:
            questions_history[question_hash] = question
            with open("shared_storage/questions_history.pkl", "wb") as f:
                pickle.dump(questions_history, f)

        # Calculate progress
        total_chunks = len(documents)
        queried_chunk_hashes = set(questions_history.keys())
        covered_chunks = len([chunk for chunk in documents if hash_chunk(chunk) in queried_chunk_hashes])
        progress = (covered_chunks / total_chunks)

        st.write("Answer:", answer)
        st.write("Study Progress:", f"{progress * 100:.2f}%")
        st.progress(progress)
