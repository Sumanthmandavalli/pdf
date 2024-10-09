import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter
import pdfplumber
from flask import Flask , render_template , session , jsonify , request
from flask_cors import CORS  
import pdfplumber
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import logging
# Load environment variables
load_dotenv()
MODEL = "mistral"  # Specify Ollama's Mixture of Experts model

# Initialize the model and embeddings
model = Ollama(model=MODEL, temperature=0)
embeddings = OllamaEmbeddings()
parser = StrOutputParser()

def extract_text_from_pdfs(pdf_paths):
    combined_text = ''
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                combined_text += page.extract_text() + '\n'
    return combined_text

# List of PDF paths
pdf_paths = ["Crowley.pdf"]

# Extract text from all PDFs
combined_text = extract_text_from_pdfs(pdf_paths)

# Define the prompt template
template = """
Answer the question based on the {context}. 
Context: {context}
Question: {question}
"""

prompt = PromptTemplate.from_template(template)
chain = prompt | model | parser

text_splitter = CharacterTextSplitter(
    chunk_size=50,  # The maximum number of characters per chunk
    chunk_overlap=20  # Overlap of characters between chunks for context
)

chunks = text_splitter.split_text(combined_text)

# 3. Prepare documents to store in DocArrayInMemorySearch
documents = [Document(page_content=chunk) for chunk in chunks]

vectorstore = DocArrayInMemorySearch.from_documents(documents, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 50})


app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def format_answer(answer):
    paragraphs = answer.split('\n')
    formatted_answer = ''
    for para in paragraphs:
        if para.strip():
            formatted_answer += f'<p>{para.strip()}</p>'
    return formatted_answer

# Function for generating LLM response
def generate_response(question):
    try:
        documents = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in documents])
        answer = chain.invoke({"context": context, "question": question})
        return format_answer(answer)
    except Exception as e:
        logging.error("Error in generate_response: %s", str(e))
        return "An error occurred while generating the response."

@app.route('/')
def index():
    if 'messages' not in session:
        session['messages'] = [{"role": "assistant", "content": ""}]
    return render_template('index.html', messages=session['messages'])

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('message')
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    if 'messages' not in session:
        session['messages'] = []

    try:
        session['messages'].append({"role": "user", "content": user_input})
        response = generate_response(user_input)
        session['messages'].append({"role": "assistant", "content": response})
        return jsonify({"message": response})
    except Exception as e:
        logging.error("Error in chat endpoint: %s", str(e))
        return jsonify({"error": "An error occurred"}), 500

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    try:
        return jsonify(session.get('messages', []))
    except Exception as e:
        logging.error("Error in get_chat_history endpoint: %s", str(e))
        return jsonify({"error": "An error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
