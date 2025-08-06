from pydantic import BaseModel, field_validator, Field
from fastapi import UploadFile
from typing import Annotated

class FileUpload(BaseModel):
    '''
    This is file upload validation to allow for txt and pdf only
    '''
    file: Annotated[UploadFile,Field(..., description="Upload the file you want to retrive context")]
    
    @field_validator('file')
    def validate_pdf_txt(cls, value: UploadFile):
        if not value.filename.lower().endswith((".pdf",".txt")):
            raise ValueError("Only pdf and txt allowed")
        return value