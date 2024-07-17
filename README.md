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

# Running the Applications
### Flask Application for WhatsApp
## Step 1:Start the Flask App
run on the termial - python whatsapp.py
_This will start the Flask server on the default port 5000. You can access it via http://127.0.0.1:5000._

## Step 2:Expose the Flask App to the Internet:
If you want to test the app with WhatsApp, you'll need to expose your Flask server to the internet. You can use a tool like ngrok for this:
run - **ngrok http 5000**
_Note the https URL provided by ngrok and use it to configure your Twilio WhatsApp webhook._

# Running the Streamlit App
## Step 1:Start the Streamlit App:
Run the Streamlit app for managing PDFs and asking questions:
- run **streamlit run streamlit_app.py** on the terminal
_This will start the Streamlit server on the default port 8501. You can access it via http://localhost:8501_

## Step 2:Test the Streamlit App:
- Open the Streamlit app in your browser.
- Use the "Manage PDFs" section to upload and manage PDF files.
- Use the "Ask Questions" section to ask questions and get responses based 
  on the uploaded PDFs.

# Step-by-Step Guide to Configure Twilio Webhook

## 1. Sign in to Twilio
- Go to Twilio's website(https://www.twilio.com/en-us) and sign in to your account.
- If you don't have an account, you can create one by following the 
  registration process.

## 2. Set Up a Twilio WhatsApp Sandbox
### Navigate to the Twilio Console:
- After logging in, go to the Twilio Console.
- Navigate to the "Messaging" section and then to "Try it out" under 
  "Programmable Messaging".
### Configure the WhatsApp Sandbox:
- Find the WhatsApp Sandbox configuration under "Try WhatsApp".
- You will see a sandbox number and a code to join the sandbox. Follow the 
  instructions to join the sandbox by sending the code via WhatsApp to the 
  given number
### Set Up Webhook URLs:
- In the "Sandbox Configuration" section, you will see fields for "When a message comes in" and "When a message is delivered".
- Enter your ngrok URL followed by the /whatsapp endpoint in the "When a message comes in" field.
- For example, if your ngrok URL is https://abcd1234.ngrok.io, then the webhook URL should be https://abcd1234.ngrok.io/whatsapp.

# Update the Twilio Webhook URL
## step 1:Go Back to Twilio Console:
- In the WhatsApp Sandbox configuration, update the "When a message comes in" field with your ngrok URL and the __https://abcd1234.ngrok.io/whatsapp__ _ endpoint.
- Click "Save" to update the configuration.

# Send a Message via whatsapp
- Send a message to your Twilio sandbox number, and observe the response from your chatbot



  






