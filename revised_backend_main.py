from fastapi import FastAPI, File, UploadFile, HTTPException, Header, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
import os
import time
from moviepy.editor import VideoFileClip
from PIL import Image
import openai
import uvicorn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import boto3
import logging
from botocore.exceptions import NoCredentialsError
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel
import re
import fitz
from typing import List, Dict
import redis

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class QueryVideoRequest(BaseModel):
    query: str

class QueryPDFRequest(BaseModel):
    query: str

# Ensure OPENAI_API_KEY is set
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
openai.api_key = os.getenv("OPENAI_API_KEY")

# S3 Configuration
S3_BUCKET = 'standalone-data-bucket'
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Pinecone Configuration
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'multimodal')

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(PINECONE_INDEX_NAME)

# Redis Configuration for Session Management
redis_client = redis.Redis(host="localhost", port=6379, db=0)

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
        logger.error(f"Error generating embeddings: {e}")
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

                logger.debug(f"Processed {len(documents)} pages from {file_name}")
                return documents
                
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return []

    def highlight_keywords_on_page(self, file_path: str, page_num: int, keywords: List[str]) -> bytes:
        """Highlight only the keywords on a PDF page."""
        try:
            doc = fitz.open(file_path)
            if page_num < 1 or page_num > len(doc):
                logger.error(f"Invalid page number: {page_num}")
                return None
                
            page = doc[page_num - 1]  # Convert to 0-based index
            
            # Search for each keyword on the page
            for keyword in keywords:
                text_instances = page.search_for(keyword)
                
                if not text_instances:
                    logger.debug(f"Keyword '{keyword}' not found on page {page_num}")
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
            logger.error(f"Error highlighting page: {e}")
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
            logger.info(f"Ingested {len(vectors)} pages from {file_name} into Pinecone!")
        
        return True
    except Exception as e:
        logger.error(f"Error processing PDF {file_name}: {e}")
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
        logger.info("Searching documents...")
        query_response = index.query(
            vector=query_embedding,
            top_k=3,  # Reduce the number of matches
            include_metadata=True,
            filter={"file_name": file_name}
        )
        
        matches = query_response.get('matches', [])
        if not matches:
            return {
                "answer": "No relevant information found in the documents.",
                "context": "",
                "highlights": []
            }
        
        # Filter matches based on the score
        score_threshold = 0.7  # Adjust this threshold as needed
        filtered_matches = [match for match in matches if match['score'] >= score_threshold]

        if not filtered_matches:
            return {
                "answer": "No relevant information found in the documents.",
                "context": "",
                "highlights": []
            }

        contexts = []
        highlights = []

        for match in filtered_matches:
            metadata = match.get('metadata', {})
            text = metadata.get('text', '')
            
            if text and all(key in metadata for key in ['file_path', 'file_name', 'page_num']):
                contexts.append({
                    'text': text,
                    'score': match['score'],
                    'metadata': metadata
                })
                
                highlights.append({
                    "file_path": metadata['file_path'],
                    "file_name": metadata['file_name'],
                    "page_num": int(metadata['page_num']),
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
        
        logger.info(f"Found {len(contexts)} relevant pages")
        
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
            max_tokens=300
        )
        
        answer = response.choices[0].message['content'].strip()
        
        # Extract keywords from the answer
        keywords = extract_keywords(answer)
        logger.debug(f"Extracted keywords: {keywords}")  # Debugging
        
        return {
            "answer": answer,
            "context": context_text,
            "highlights": highlights,
            "keywords": keywords
        }
        
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        return {"answer": f"Error: {str(e)}", "context": "", "highlights": []}

# Video-related functions
def upload_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name
    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="The file was not found")
    except NoCredentialsError:
        raise HTTPException(status_code=403, detail="Credentials not available")

def download_from_s3(file_name, bucket):
    try:
        logging.debug(f"Downloading file: {file_name} from bucket: {bucket}")
        s3_client.download_file(bucket, file_name, file_name)
        logging.debug(f"File downloaded successfully: {file_name}")
    except NoCredentialsError:
        raise HTTPException(status_code=403, detail="Credentials not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

def extract_frames(video_path, interval=1):
    frames = []
    with VideoFileClip(video_path) as video:
        duration = video.duration
        for i in range(0, int(duration), interval):
            frame = video.get_frame(i)
            frames.append((i, Image.fromarray(frame)))
    return frames

def transcribe_audio(video_path):
    with VideoFileClip(video_path) as video:
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path)
        with open(audio_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        os.remove(audio_path)
        return transcript["text"]

def split_transcript(transcript, words_per_segment=50):
    words = transcript.split()
    segments = []
    start_time = 0
    words_per_second = 2.5
    segment_duration = words_per_segment / words_per_second
    for i in range(0, len(words), words_per_segment):
        segment_text = " ".join(words[i:i + words_per_segment])
        end_time = start_time + segment_duration
        segments.append((start_time, end_time, segment_text))
        start_time = end_time
    return segments

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def store_embeddings_in_pinecone(segments):
    vectors = []
    for i, segment in enumerate(segments):
        embedding = get_embedding(segment[2])
        logging.debug(f"Generated embedding for segment {i}: {segment[2]}")
        vectors.append((f"segment_{i}", embedding, {"start_time": segment[0], "end_time": segment[1], "text": segment[2]}))
    index.upsert(vectors)
    logging.debug(f"Stored {len(vectors)} embeddings in Pinecone")

def find_relevant_segment(query):
    query_embedding = get_embedding(query)
    logging.debug(f"Generated query embedding for: {query}")
    results = index.query(vector=query_embedding, top_k=1, include_metadata=True)
    if results.matches:
        match = results.matches[0]
        logging.debug(f"Found relevant segment: {match.metadata}")
        return match.metadata["start_time"], match.metadata["end_time"], match.metadata["text"]
    logging.warning("No relevant segment found")
    return None

def generate_response(query, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{context}\n\n{query}"},
        ],
        max_tokens=150,
    )
    return response.choices[0].message["content"].strip()

def generate_subclip(video_path, start_time, end_time, margin=3):
    try:
        with VideoFileClip(video_path) as video:
            duration = video.duration
            start_time = max(start_time - margin, 0)
            end_time = min(end_time + margin, duration)
            logging.debug(f"Generating subclip from {start_time} to {end_time}")
            subclip = video.subclip(start_time, end_time)
            subclip_path = f"subclip_{int(start_time)}-{int(end_time)}.mp4"
            subclip.write_videofile(subclip_path, audio_codec="aac")
            logging.debug(f"Subclip saved to: {subclip_path}")
            return subclip_path
    except Exception as e:
        logging.error(f"Error generating subclip: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating subclip: {str(e)}")

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...), session_id: str = Header(...)):
    try:
        logger.debug(f"Received file: {file.filename}")
        logger.debug(f"Session ID: {session_id}")

        if not file.filename:
            raise HTTPException(status_code=400, detail="No file name provided")

        safe_filename = file.filename.replace(" ", "_")
        s3_object_path = f"video/{safe_filename}"

        logger.debug(f"Uploading file to S3: {s3_object_path}")
        logger.debug(f"Bucket: {S3_BUCKET}")
        logger.debug(f"Credentials: {AWS_ACCESS_KEY_ID}, {AWS_SECRET_ACCESS_KEY}")

        # Upload the file to S3
        s3_client.upload_fileobj(
            file.file,
            S3_BUCKET,
            s3_object_path
        )

        # Verify the upload
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_object_path)
        except Exception as e:
            logger.error(f"Failed to verify file upload: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to verify file upload: {str(e)}")

        # Store the uploaded file in Redis
        redis_client.hset(session_id, safe_filename, s3_object_path)

        return JSONResponse(
            content={
                "message": "Video uploaded successfully",
                "video_path": s3_object_path
            }
        )
    except NoCredentialsError:
        logger.error("AWS credentials not available")
        raise HTTPException(status_code=403, detail="AWS credentials not available")
    except FileNotFoundError:
        logger.error("The file was not found")
        raise HTTPException(status_code=404, detail="The file was not found")
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")
        
@app.post("/query-video/")
async def query_video(request: QueryVideoRequest, session_id: str = Header(...)):
    query = request.query

    try:
        results = []
        for video_name, video_path in redis_client.hgetall(session_id).items():
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            download_from_s3(video_path, S3_BUCKET)
            transcript = transcribe_audio(video_path)

            segments = split_transcript(transcript)
            store_embeddings_in_pinecone(segments)

            relevant_segment = find_relevant_segment(query)
            if relevant_segment:
                start_time, end_time, relevant_text = relevant_segment
                context = f"Transcript: {transcript}\nRelevant text: {relevant_text}"
                response = generate_response(query, context)
                subclip_path = generate_subclip(video_path, start_time, end_time)
                results.append({
                    "video_name": video_name.decode("utf-8"),
                    "response": response,
                    "relevant_text": relevant_text,
                    "start_time": start_time,
                    "end_time": end_time,
                    "subclip_path": subclip_path
                })

        if results:
            return JSONResponse(content=results)
        else:
            raise HTTPException(status_code=404, detail="No relevant segment found in any video")
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), session_id: str = Header(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file name provided")

        # Sanitize the file name by replacing spaces with underscores
        safe_filename = file.filename.replace(" ", "_")

        # Save the uploaded file to a temporary location
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, safe_filename)

        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Upload the PDF to S3
        s3_object_path = f"pdf/{safe_filename}"  # Store PDFs in the 'pdf' folder in S3
        s3_client.upload_fileobj(
            file.file,
            S3_BUCKET,
            s3_object_path
        )

        # Verify that the file exists in S3
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_object_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to verify file upload: {str(e)}")

        # Process the PDF and store embeddings in Pinecone
        if upload_and_process_pdf(file_path, safe_filename):
            # Store the uploaded file in Redis
            redis_client.hset(session_id, safe_filename, s3_object_path)

            return JSONResponse(content={
                "message": "PDF uploaded and processed successfully",
                "file_name": safe_filename,
                "s3_path": s3_object_path  # Return the S3 object path
            })
        else:
            raise HTTPException(status_code=500, detail="Error processing PDF")
    except Exception as e:
        logging.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")

@app.post("/query-pdf/")
async def query_pdf(request: QueryPDFRequest, session_id: str = Header(...)):
    query = request.query

    try:
        results = []
        for file_name, file_path in redis_client.hgetall(session_id).items():
            response = query_documents(query, file_name.decode("utf-8"))
            results.append({
                "file_name": file_name.decode("utf-8"),
                "response": response
            })

        if results:
            return JSONResponse(content=results)
        else:
            raise HTTPException(status_code=404, detail="No relevant information found in any PDF")
    except Exception as e:
        logging.error(f"Error querying PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying PDF: {str(e)}")

@app.get("/get-subclip/")
async def get_subclip(subclip_path: str):
    def iterfile():
        with open(subclip_path, mode="rb") as file_like:
            yield from file_like
    return StreamingResponse(iterfile(), media_type="video/mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)