# External Libraries
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from sparknlp.base import LightPipeline, DocumentAssembler
from sparknlp.annotator import SentenceDetectorDLModel
from pyspark.ml import PipelineModel
import os

# Initialization
app = Flask(__name__)

# Spark NLP Configuration
spark = (SparkSession.builder \
    .appName("Spark NLP Server") \
    .config("spark.jars", "/tmp/spark-nlp-assembly-gpu-5.1.4.jar") \
    .config("spark.nlp.cuda.allocator", "ON") \
    .getOrCreate())

# Text Processing Functions
def process_text(text):
    documenter = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    sentencerDL = SentenceDetectorDLModel \
        .load("model/") \
        .setInputCols(["document"]) \
        .setOutputCol("sentences")

    sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))

    return sd_model.fullAnnotate(text)

def annotation_to_dict(anno):
    return {
        'annotatorType': anno.annotatorType,
        'begin': anno.begin,
        'end': anno.end,
        'result': anno.result,
        'metadata': dict(anno.metadata)  # Convert JavaMap to Python dictionary
    }

# Flask Routes
@app.route('/process', methods=['POST'])
def process():
    data = request.json
    text = data.get('text', '')
    result = process_text(text)

    # Convert Annotation objects to dictionaries
    json_result = [{k: [annotation_to_dict(vv) for vv in v] if v else None for k, v in res.items()} for res in result]

    return jsonify(json_result)

# Run the Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
