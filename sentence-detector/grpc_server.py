from concurrent import futures
import grpc
import time
import sentence_detection_pb2
import sentence_detection_pb2_grpc
from pyspark.sql import SparkSession
from sparknlp.base import LightPipeline, DocumentAssembler
from sparknlp.annotator import SentenceDetectorDLModel
from pyspark.ml import PipelineModel

# Global variables
spark: SparkSession = None
sd_model = None

def annotation_to_dict(anno):
    """Converts annotation to dictionary."""
    return {
        'begin': anno.begin,
        'end': anno.end,
        'result': anno.result,
        }

def process_text(text: str):
    """Processes text using the global model."""
    return sd_model.fullAnnotate(text)

class SentenceDetectionService(sentence_detection_pb2_grpc.SentenceDetectionServiceServicer):
    def ProcessText(self, request, context):
        if not sd_model:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details('Model not initialized yet')
            return sentence_detection_pb2.ProcessTextResponse()

        start_time = time.time()
        text = request.text
        result = process_text(text)
        end_time = time.time()
        print(f"Processing took {end_time - start_time} seconds")
        json_result = [{k: [annotation_to_dict(vv) for vv in v] if v else None for k, v in res.items()} for res in result][0]

        annotations = [sentence_detection_pb2.SentenceAnnotation(
            begin=anno['begin'],
            end=anno['end'],
            result=anno['result'],
        ) for anno in json_result['sentences']]

        return sentence_detection_pb2.ProcessTextResponse(sentences=annotations)

def serve():
    """Starts the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sentence_detection_pb2_grpc.add_SentenceDetectionServiceServicer_to_server(SentenceDetectionService(), server)
    server.add_insecure_port('[::]:5001')
    server.start()

    print("gRPC server started and listening on [::]:5001")
    server.wait_for_termination()

def startup_event():
    """Initializes Spark and loads the model during server startup."""
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

if __name__ == '__main__':
    startup_event()  # Initialize Spark and the sentence detection model
    serve()  # Start the gRPC server
    