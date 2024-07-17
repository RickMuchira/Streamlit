# whatsapp.py

import os
import pickle
import logging
from flask import Flask, request, abort
from langchain_text_splitters import RecursiveCharacterTextSplitter
from twilio.twiml.messaging_response import MessagingResponse
import faiss
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize Flask app
app = Flask(__name__)

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

# Function to generate response
def generate_response(question, vectors):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': question})
    return response['answer']

@app.route('/whatsapp', methods=['POST'])
def whatsapp_reply():
    """ Respond to incoming messages with the query results. """
    try:
        msg = request.form.get('Body')
        if not msg:
            abort(400, description="Invalid request: 'Body' parameter missing")

        logger.info(f"Received message: {msg}")

        # Load vectors and documents from shared storage
        index = faiss.read_index("shared_storage/vectors.index")
        with open("shared_storage/documents.pkl", "rb") as f:
            documents = pickle.load(f)

        docstore = InMemoryDocstore({i: doc for i, doc in enumerate(documents)})
        vectors = FAISS(embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                        docstore=docstore, index=index,
                        index_to_docstore_id={i: i for i in range(len(documents))})

        logger.info("Generating response...")
        answer = generate_response(msg, vectors)
        logger.info(f"Generated response: {answer}")

        response = MessagingResponse()
        response.message(answer)

        return str(response)
    except Exception as e:
        logger.error(f"Error processing the request: {e}")
        response = MessagingResponse()
        response.message("Sorry, there was an error processing your request. Please try again later.")
        return str(response)

if __name__ == '__main__':
    app.run(debug=True)
