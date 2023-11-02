from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from sparknlp.base import LightPipeline, DocumentAssembler
from sparknlp.annotator import SentenceDetectorDLModel
from pyspark.ml import PipelineModel
from typing import List

app = FastAPI(title="Sentence Detection API v0.1")

# Global variables to store Spark session and model
spark: SparkSession = None
sd_model = None

# Pydantic models
class RequestModel(BaseModel):
    text: str = Field(..., description="The input text to be processed by the pipeline.")

class SentenceAnnotation(BaseModel):
    annotatorType: str
    begin: int
    end: int
    result: str
    metadata: dict

class ResponseModel(BaseModel):
    sentences: List[SentenceAnnotation]

# Helper functions
def process_text(text):
    return sd_model.fullAnnotate(text)

def annotation_to_dict(anno):
    return {
'annotatorType': anno.annotatorType,
'begin': anno.begin,
'end': anno.end,
'result': anno.result,
'metadata': dict(anno.metadata)  # Convert JavaMap to Python dictionary
}

# FastAPI startup event to initialize Spark and load the model
@app.on_event("startup")
def startup_event():
    global spark, sd_model
    spark = (SparkSession.builder
             .config("spark.driver.memory", "16G")
             .config("spark.driver.maxResultSize", "0")
             .config("spark.nlp.cuda.allocator", "ON")
             .getOrCreate())

    documenter = DocumentAssembler() \
                 .setInputCol("text") \
                 .setOutputCol("document")

    sentencerDL = SentenceDetectorDLModel \
                  .load("model/") \
                  .setInputCols(["document"]) \
                  .setOutputCol("sentences")

    sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))

@app.post("/process", response_model=ResponseModel, tags=["Sentence Detection"],
          summary="Detect sentences in given text",
          description="This endpoint detects sentences in given text Spark NLP pipeline and returns the annotations (enriched objects with sentences).")
def process(request_data: RequestModel):
    if not sd_model:
        raise HTTPException(status_code=500, detail="Model not initialized yet")

    text = request_data.text
    result = process_text(text)

    # Convert Annotation objects to dictionaries
    json_result = [{k: [annotation_to_dict(vv) for vv in v] if v else None for k, v in res.items()} for res in result][0]  # Assuming a single result

    return json_result

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

