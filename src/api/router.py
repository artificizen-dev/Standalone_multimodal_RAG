import os, json
import traceback
import pandas as pd
from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Depends, HTTPException, Form, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

router = APIRouter()
root_dir = Path.cwd()
load_dotenv()

ALLOWED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "word",
    "application/msword": "word"
}

class Message(BaseModel):
    content: str

@router.get('/health_check')
async def index_page():
    return {"message": "Standalone Bot Heath Success"}

@router.post("/chroma/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    document_name: str = Form(...)
):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
            
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=400, detail="Only PDF or Word files are allowed")
        
        document_type = ALLOWED_CONTENT_TYPES[file.content_type]

        content_type = file.content_type
        file_content = await file.read()

        try:
            pass
        except Exception as e:
            print(f"Error processing file: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
        
        return {
            "message": f"{document_name} ({document_type.upper()}) uploaded successfully",
            "file_url": ""
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))