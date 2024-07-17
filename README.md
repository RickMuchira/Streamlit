# WhatsApp Question and Answering PDF Chatbot

This project is a WhatsApp chatbot that can answer questions based on the content of uploaded PDF documents. The chatbot uses various NLP and machine learning tools to process the documents and generate responses.

## Features

- Upload PDFs and manage them via a Streamlit interface.
- Ask questions through WhatsApp and get responses based on the content of the uploaded PDFs.
- Efficient vector-based document retrieval and response generation.
- Caching for improved performance.

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Twilio account for WhatsApp integration
- Ngrok for exposing the Flask server to the internet

## Installation

### Step 1: Clone the Repository
git clone https://github.com/RickMuchira/Streamlit.git

### Step 2: Create and Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Step 3: Install Dependencies
pip install -r requirements.txt

## Step 4: Set Up Environment Variables
Create a .env file in the root directory with the following content:

- GROQ_API_KEY=your_groq_api_key
- GOOGLE_API_KEY=your_google_api_key

### Running the Applications
Flask Application for WhatsApp
Start the Flask App
