# Importing required libraries
from fastapi_offline import FastAPIOffline
from fastapi import Depends, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from sparknlp.base import LightPipeline, DocumentAssembler
from sparknlp.annotator import SentenceDetectorDLModel
from pyspark.ml import PipelineModel
from typing import List
import uvicorn
import asyncio
import logging

# Initialize the FastAPI app
app = FastAPIOffline(title="Sentence Detection API", version="0.3")

# Global variables for Spark session and model

spark: SparkSession = None
sd_model = None
model_initialized = False
timeout_occurred = False 

# Pydantic models for request and response
class RequestModel(BaseModel):
    text: str = Field(..., description="The input text to be processed by the pipeline.")

class SentenceAnnotation(BaseModel):
    begin: int
    end: int
    result: str

class ResponseModel(BaseModel):
    sentences: List[SentenceAnnotation]

# Helper function to convert annotation to dictionary
def annotation_to_dict(anno):
    return {'begin': anno.begin, 'end': anno.end, 'result': anno.result}

# Function to reinitialize Spark session and model
def reinitialize():
    global spark, sd_model, model_initialized, timeout_occurred
    try:
        if spark:
            spark.stop()

        spark = SparkSession.builder \
                 .appName("SentenceDetectionAPI") \
                 .config("spark.driver.memory", "8G") \
                 .config("spark.driver.maxResultSize", "0") \
                 .config("spark.nlp.cuda.allocator", "ON") \
                 .getOrCreate()

        documenter = DocumentAssembler().setInputCol("text").setOutputCol("document")
        sentencerDL = SentenceDetectorDLModel.load("model/").setInputCols(["document"]).setOutputCol("sentences")
        sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))
        model_initialized = True
        timeout_occurred = False
        logging.info("Reinitialization successful")
    except Exception as e:
        logging.error(f"Error during reinitialization: {e}")

# Async function to process text with timeout
async def process_text(text: str, timeout=300):
    global timeout_occurred
    if not text.strip():
        raise ValueError("Input text is empty.")

    try:
        result = await asyncio.wait_for(asyncio.to_thread(sd_model.fullAnnotate, text), timeout)
        return result
    except asyncio.TimeoutError:
        logging.error("Processing timed out")
        timeout_occurred = True
        reinitialize()
        raise HTTPException(status_code=408, detail="Processing timed out")
    except Exception as e:
        logging.error(f"Error during text processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Function to filter sentences based on criteria
def filter_sentences(sentences: List[dict]) -> List[dict]:
    return [s for s in sentences if len(s['result'].strip()) > 3 and ' ' in s['result'].strip()]

# FastAPI event and route handlers
@app.on_event("startup")
def startup_event():
    global spark, sd_model, model_initialized
    try:
        spark = SparkSession.builder \
            .appName("SentenceDetectionAPI") \
            .config("spark.driver.memory", "8G") \
            .config("spark.driver.maxResultSize", "0") \
            .config("spark.nlp.cuda.allocator", "ON") \
            .getOrCreate()

        documenter = DocumentAssembler().setInputCol("text").setOutputCol("document")
        sentencerDL = SentenceDetectorDLModel.load("model/").setInputCols(["document"]).setOutputCol("sentences")
        sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))
        model_initialized = True
    except Exception as e:
        logging.error(f"Error during startup: {e}")
        raise e

@app.get("/startup-check", tags=["Health and Startup Checks"])
async def startup_check():
    if model_initialized:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="initializing")

@app.get("/health", tags=["Health and Startup Checks"])
async def health_check():
    if not timeout_occurred:
        return {"status": "healthy"}
    else:
        raise HTTPException(status_code=503, detail="unhealthy")

@app.get("/{full_path:path}", include_in_schema=False)
def redirect_to_docs(full_path: str):
    return RedirectResponse(url='/docs')

@app.post("/process", response_model=ResponseModel, tags=["Sentence Detection"])
async def process(request_data: RequestModel):
    if not sd_model:
        raise HTTPException(status_code=500, detail="Model not initialized yet")
    try:
        result = await process_text(request_data.text)
        json_result = [{k: [annotation_to_dict(vv) for vv in v] if v else None for k, v in res.items()} for res in result][0]
        filtered_sentences = filter_sentences(json_result.get("sentences", []))
        return {"sentences": filtered_sentences}
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

# Main entry point for running the app
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)
    