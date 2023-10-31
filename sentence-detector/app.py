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
         .config("spark.driver.memory", "16G") \
         .config("spark.driver.maxResultSize", "0") \
         .config("spark.kryoserializer.buffer.max", "2000M") \
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


def run():
    app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    run()
