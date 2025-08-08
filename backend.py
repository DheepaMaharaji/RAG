from chromadb.config import Settings
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import chromadb
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_together import ChatTogether
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables to store the RAG chain
rag_chain = None
collection = None


# --- RAG Pipeline Functions (from your original code) ---

def load_and_split_pdfs(pdf_paths):
    """Load PDF documents from a list of paths and split them into chunks."""
    all_documents = []
    for pdf_path in pdf_paths:
        try:
            if os.path.exists(pdf_path):
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                all_documents.extend(documents)
                logger.info(f"Loaded {len(documents)} pages from {pdf_path}")
            else:
                logger.warning(f"PDF file not found: {pdf_path}")
        except Exception as e:
            logger.error(f"Error loading {pdf_path}: {e}")

    if not all_documents:
        logger.warning("No documents loaded. Please check PDF paths and permissions.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunked_documents = text_splitter.split_documents(all_documents)
    logger.info(f"Split into {len(chunked_documents)} chunks.")
    return chunked_documents


def create_chroma_collection(collection_name="my_pdf_collection", persist_directory="./chroma_db",
                             model_name="all-MiniLM-L6-v2"):
    """Initialize ChromaDB client and create a collection."""
    try:
        client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            chroma_db_impl="duckdb+parquet"
        ))
        embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
        collection = client.get_or_create_collection(name=collection_name)
        logger.info(f"Successfully created or connected to collection '{collection_name}'.")
        return collection, embedding_function
    except Exception as e:
        logger.error(f"Error creating ChromaDB collection: {e}")
        raise


def prepare_data_for_chromadb(chunked_documents):
    """Prepare LangChain Documents for ingestion into ChromaDB."""
    texts = [doc.page_content for doc in chunked_documents]
    metadatas = [doc.metadata for doc in chunked_documents]
    ids = [f"doc_chunk_{i}" for i in range(len(chunked_documents))]
    return texts, metadatas, ids


def ingest_data_into_chromadb(collection, texts, metadatas, ids,embedding_function):
    """Add data to ChromaDB collection."""
    try:
        # Generate embeddings for the texts
        embeddings = embedding_function.embed_documents(texts)

        # Add documents with embeddings
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully added {len(texts)} documents to collection '{collection.name}'.")

    except Exception as e:
        logger.error(f"Error adding documents to ChromaDB: {e}")
        raise

def setup_rag_pipeline(collection, embedding_function, model_name="meta-llama/Llama-3-8b-chat-hf"):
    """Set up the RAG pipeline by connecting retriever, prompt, and LLM."""
    try:
        # Create the LangChain VectorStore object
        vector_store = Chroma(
            persist_directory="./chroma_db",
            collection_name=collection.name,
            embedding_function=embedding_function
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Define the LLM using LangChain's wrapper for Together AI
        llm = ChatTogether(
            model=model_name,
            temperature=0.7,
            max_tokens=512,
            together_api_key="ce8bca09a6df6e7f1f23a900281cf1d71922e045121324bbae50a8e2e612b0c6"
        )

        # Define the prompt template
        template = """
        You are an AI assistant for Northeastern University's Global Services office.
        Your goal is to answer questions based on the provided context only.
        If the context doesn't contain the answer, say "I am sorry, but I cannot find that information in the provided documents. Please contact the Office of Global Services directly for more assistance."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Create the RAG chain
        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        logger.info("RAG pipeline setup complete.")
        return rag_chain

    except Exception as e:
        logger.error(f"Error setting up RAG pipeline: {e}")
        raise


def initialize_rag_system():
    """Initialize the entire RAG system."""
    global rag_chain, collection

    try:
        # Update these paths to match your PDF locations
        # For development, you might want to use a smaller subset
        all_pdf_files = [
            'resources / African Students Access Scholarship.pdf',
            'resources / Double Husky Scholarship.pdf',
            'resources/Fellowship Opportunities.pdf',
            'resources/Full Circle Scholarship.pdf',
            'resources/Graduate Scholarships _ Student Financial Services.pdf',
            'resources/Masters Apply - Khoury College of Computer Sciences.pdf',
            'resources / Parent and Family Scholarship.pdf'
            # Add more PDF paths as needed
        ]

        # Check if we have any valid PDF files
        valid_pdfs = [pdf for pdf in all_pdf_files if os.path.exists(pdf)]

        if not valid_pdfs:
            logger.warning("No PDF files found. Using fallback initialization.")
            # Create empty collection for demo purposes
            collection, embedding_function = create_chroma_collection()
            rag_chain = setup_rag_pipeline(collection, embedding_function)
            return

        # Load and split PDFs
        chunked_documents = load_and_split_pdfs(valid_pdfs)

        if chunked_documents:
            # Create ChromaDB collection
            collection, embedding_function = create_chroma_collection()

            # Ingest data if collection is empty
            if collection.count() == 0:
                texts, metadatas, ids = prepare_data_for_chromadb(chunked_documents)
                ingest_data_into_chromadb(collection, texts, metadatas, ids, embedding_function)
            else:
                logger.info(f"Collection already has {collection.count()} documents. Skipping ingestion.")

            # Setup RAG pipeline
            rag_chain = setup_rag_pipeline(collection, embedding_function)
            logger.info("RAG system initialized successfully!")
        else:
            logger.error("No documents available for RAG pipeline setup.")

    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        logger.error(traceback.format_exc())


# --- Flask Routes ---

@app.route('/')
def index():
    """Serve the frontend HTML."""
    # You can serve the HTML file directly or return it as a string
    # For this example, we'll return a simple template
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NU Global Services Assistant</title>
    </head>
    <body>
        <h1>NU Global Services Assistant</h1>
        <p>The chat interface is available. You can integrate the frontend HTML here.</p>
        <p>API endpoint available at: <code>/api/chat</code></p>
    </body>
    </html>
    """


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages from the frontend."""
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        user_message = data['message'].strip()

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        # Check if RAG chain is initialized
        if rag_chain is None:
            logger.error("RAG chain not initialized")
            return jsonify({
                'response': "I'm sorry, but the system is not properly initialized. Please contact the administrator."
            }), 500

        logger.info(f"Processing question: {user_message}")

        # Get response from RAG chain
        response = rag_chain.invoke(user_message)

        logger.info(f"Generated response: {response[:100]}...")

        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'response': "I'm sorry, but I encountered an error while processing your request. Please try again later."
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        status = {
            'status': 'healthy',
            'rag_initialized': rag_chain is not None,
            'collection_count': collection.count() if collection else 0
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    try:
        if collection is None:
            return jsonify({'error': 'Collection not initialized'}), 500

        stats = {
            'document_count': collection.count(),
            'collection_name': collection.name if hasattr(collection, 'name') else 'unknown'
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting NU Global Services Assistant Backend...")

    # Initialize the RAG system
    initialize_rag_system()

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=8000)
