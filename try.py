import os
import re
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import openai
import streamlit as st
import pymupdf4llm
import fitz

# Set page config as the first Streamlit command
st.set_page_config(page_title="Multi-PDF RAG Bot", layout="wide")

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "multimodal"

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("Please set OPENAI_API_KEY and PINECONE_API_KEY in your environment variables.")
    st.stop()

openai.api_key = OPENAI_API_KEY  # Set OpenAI API key
pc = Pinecone(api_key=PINECONE_API_KEY)  # Initialize Pinecone client

# Check if Pinecone index exists and create it if not
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(PINECONE_INDEX_NAME)  # Connect to the Pinecone index

# Initialize Streamlit session state for uploaded PDFs
if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = {}

# Function to generate embeddings
def generate_embeddings(text: str) -> List[float]:
    """Generate embeddings for a given text using OpenAI."""
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return []

# Function to extract keywords from the answer
def extract_keywords(answer: str) -> List[str]:
    """Extract keywords or phrases from the answer using regex."""
    keywords = re.findall(r'\b\w+\b', answer)  # Extract individual words
    return keywords

# PDFProcessor class
class PDFProcessor:
    def process_document(self, file_path: str, file_name: str) -> List[Dict]:
        """Process PDF and extract text with proper metadata structure."""
        documents = []
        try:
            with fitz.open(file_path) as pdf_doc:
                # Get document metadata
                metadata = pdf_doc.metadata
                total_pages = len(pdf_doc)

                # Process each page
                for page_num in range(total_pages):
                    page = pdf_doc[page_num]
                    text = page.get_text()  # Extract all text from the page
                    
                    if text.strip():
                        documents.append({
                            "text": text,
                            "metadata": {
                                "file_path": file_path,
                                "file_name": file_name,  # Unique identifier for the PDF
                                "page_num": page_num + 1,  # 1-based page numbers
                                "total_pages": total_pages,
                                "type": "pdf"
                            }
                        })

                print(f"Processed {len(documents)} pages from {file_name}")
                return documents
                
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return []

    def highlight_keywords_on_page(self, file_path: str, page_num: int, keywords: List[str]) -> bytes:
        """Highlight only the keywords on a PDF page."""
        try:
            doc = fitz.open(file_path)
            if page_num < 1 or page_num > len(doc):
                st.error(f"Invalid page number: {page_num}")
                return None
                
            page = doc[page_num - 1]  # Convert to 0-based index
            
            # Search for each keyword on the page
            for keyword in keywords:
                text_instances = page.search_for(keyword)
                
                if not text_instances:
                    print(f"Keyword '{keyword}' not found on page {page_num}")
                    continue
                
                # Add highlights for each instance of the keyword
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=(1, 0.8, 0))  # Bright yellow
                    highlight.update()
            
            # Render page
            zoom = 2  # Higher quality
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            img_bytes = pix.tobytes()
            doc.close()
            return img_bytes
            
        except Exception as e:
            st.error(f"Error highlighting page: {e}")
            return None

# Initialize PDFProcessor
pdf_processor = PDFProcessor()

# Function to process and ingest PDF content into Pinecone
def upload_and_process_pdf(file_path: str, file_name: str):
    """Extract text, generate embeddings, and store in Pinecone."""
    try:
        documents = pdf_processor.process_document(file_path, file_name)  # Extract text from PDF
        
        # Generate embeddings and prepare vectors
        vectors = []
        for doc in documents:
            embedding = generate_embeddings(doc["text"])  # Generate embedding for page text
            if embedding:
                vectors.append({
                    "id": f"{file_name}_page_{doc['metadata']['page_num']}",
                    "values": embedding,
                    "metadata": {
                        "text": doc["text"],  # Ensure the 'text' key is included in metadata
                        "file_path": file_path,
                        "file_name": file_name,  # Unique identifier for the PDF
                        "page_num": doc['metadata']['page_num']
                    }
                })
        
        # Upsert vectors to Pinecone
        if vectors:
            index.upsert(vectors=vectors)
            st.success(f"Ingested {len(vectors)} pages from {file_name} into Pinecone!")
        
        # Track uploaded PDFs
        st.session_state.uploaded_pdfs[file_name] = file_path
        return True
    except Exception as e:
        st.error(f"Error processing PDF {file_name}: {e}")
        return False

# Query documents function
def query_documents(query: str, file_name: str) -> Dict:
    """Query documents with fixed metadata handling."""
    try:
        # Generate query embedding
        query_embedding = generate_embeddings(query)
        if not query_embedding:
            return {
                "answer": "Error generating query embedding.",
                "context": "",
                "highlights": []
            }
        
        # Query Pinecone with metadata filter
        st.info("Searching documents...")
        query_response = index.query(
            vector=query_embedding,
            top_k=10,  # Increase the number of matches for better context
            include_metadata=True,
            filter={"file_name": file_name}  # Filter by file_name
        )
        
        matches = query_response.get('matches', [])
        if not matches:
            return {
                "answer": "No relevant information found in the documents.",
                "context": "",
                "highlights": []
            }
        
        # Extract context and prepare highlights
        contexts = []
        highlights = []
        
        for match in matches:
            metadata = match.get('metadata', {})
            text = metadata.get('text', '')
            
            # Verify required metadata fields exist
            if text and all(key in metadata for key in ['file_path', 'file_name', 'page_num']):
                contexts.append({
                    'text': text,
                    'score': match['score'],
                    'metadata': metadata
                })
                
                highlights.append({
                    "file_path": metadata['file_path'],
                    "file_name": metadata['file_name'],
                    "page_num": int(metadata['page_num']),  # Ensure page_num is int
                    "text": text
                })
        
        if not contexts:
            return {
                "answer": "Could not extract context from matched documents.",
                "context": "",
                "highlights": []
            }
        
        # Sort contexts by score
        contexts.sort(key=lambda x: x['score'], reverse=True)
        
        # Prepare context text with scores
        context_text = "\n\n".join([
            f"[Page {c['metadata']['page_num']}, Score: {c['score']:.2f}] {c['text']}"
            for c in contexts
        ])
        
        st.info(f"Found {len(contexts)} relevant pages")
        
        # Generate answer
        prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        Context: {context_text}
        
        Question: {query}
        
        Answer:"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300  # Increase token limit for more detailed answers
        )
        
        answer = response.choices[0].message['content'].strip()
        
        # Extract keywords from the answer
        keywords = extract_keywords(answer)
        print(f"Extracted keywords: {keywords}")  # Debugging
        
        return {
            "answer": answer,
            "context": context_text,
            "highlights": highlights,
            "keywords": keywords  # Add keywords to the response
        }
        
    except Exception as e:
        st.error(f"Error querying documents: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return {"answer": f"Error: {str(e)}", "context": "", "highlights": []}

# Streamlit app main function
def main():
    st.title("Multi-PDF RAG Bot")
    
    # Sidebar for uploading PDFs
    st.sidebar.header("Upload PDFs")
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save the uploaded file to a permanent location
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            if upload_and_process_pdf(file_path, uploaded_file.name):
                st.sidebar.success(f"Uploaded and processed {uploaded_file.name} successfully!")
    
    # Sidebar for displaying uploaded PDFs
    st.sidebar.header("Uploaded PDFs")
    if st.session_state.uploaded_pdfs:
        selected_file = st.sidebar.selectbox(
            "Select a PDF to query",
            list(st.session_state.uploaded_pdfs.keys())
        )
    else:
        st.sidebar.info("No PDFs uploaded yet.")
    
    # Query interface
    st.header("Query Your PDFs")
    query = st.text_input("Enter your question:")
    if query and st.session_state.uploaded_pdfs:
        response = query_documents(query, selected_file)  # Pass selected_file to query_documents
        st.subheader("Answer")
        st.write(response["answer"])

        with st.expander("Show Context"):
            st.write(response["context"])

        if response.get("highlights"):
            st.subheader("Relevant Pages")
            for highlight in response["highlights"]:
                with st.expander(f"Page {highlight['page_num']} from {highlight['file_name']}"):
                    # Display highlighted page
                    file_path = st.session_state.uploaded_pdfs.get(highlight['file_name'])
                    if file_path:
                        highlighted_page = pdf_processor.highlight_keywords_on_page(
                            file_path,
                            highlight['page_num'],
                            response.get("keywords", [])  # Pass keywords for highlighting
                        )
                        if highlighted_page:
                            st.image(highlighted_page)
                    else:
                        st.error(f"File {highlight['file_name']} not found.")

if __name__ == "__main__":
    main()