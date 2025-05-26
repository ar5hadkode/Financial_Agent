from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from financial_analysis import run_financial_assistant , upload_data_handler , remove_data_handler

app = FastAPI()

class QueryInput(BaseModel):
    query: str

class DataDeleteInput(BaseModel):
    identifier: str
    
@app.post("/query")
async def handle_query(data: QueryInput):
    response = run_financial_assistant(data.query)
    return {"response": response}
@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    try:
        result = upload_data_handler(file)
        return {"message": "File uploaded and processed successfully", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/remove-data")
async def remove_data(data: DataDeleteInput):
    try:
        result = remove_data_handler(data.identifier)
        return {"message": "Data removed successfully", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))