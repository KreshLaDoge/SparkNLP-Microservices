# NLP service [Sentence detection][Multilingual]

This project utilizes a custom spark-cuda image to create an NLP service. It uses Spark NLP and a specific pre-trained Sentence Detector model.

## Docker Configuration

The Dockerfile uses a base image `spark-cuda:latest` (base image of this repo - you might need to create it by yourself locally) and downloads a Sentence Detector model from a provided link. It then sets up an environment to run a Flask application serving the NLP functionality.

### Dockerfile Content

```Dockerfile
FROM spark-cuda:latest

RUN wget https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip \
    && unzip sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip -d model \
    && rm sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip

# Copy the application file into the container
COPY app.py .

# Set the entrypoint to run the Flask application
ENTRYPOINT ["python3", "app.py"]
```

## Application Overview

The Flask application provides an endpoint `/process` which accepts text and returns processed results after applying Spark NLP transformations.

### External Libraries

- Flask
- PySpark
- Spark NLP

### Initialization

The application initializes Spark with specific configurations to work seamlessly with GPU-based resources. It also sets up necessary NLP pipelines to process incoming text.

### Text Processing

Text processing is achieved using Spark NLP's LightPipeline and a pre-trained Sentence Detector model.

### Flask Routes

The main endpoint `/process` requires a POST request with a JSON body containing the text to be processed. The response is the processed result in a structured JSON format.

### Running the Server

The Flask server can be started using Docker and listens on `0.0.0.0` on port `5000`.

## Getting Started

1. **Build the Docker image**:

   ```bash
   docker build -t sentence-detector-service .
   ```

2. **Run the Docker container with GPU support**:

   Ensure you have NVIDIA Docker runtime installed. If not, you can find instructions [here](https://github.com/NVIDIA/nvidia-docker).

   Once NVIDIA Docker is set up, run the container with GPU enabled:

   ```bash
   docker run --gpus all -p 5000:5000 sentence-detector-service
   ```

   The `--gpus all` flag enables all available GPUs for the container. If you only want to allocate a specific number of GPUs, replace `all` with the desired number, e.g., `--gpus 2` for two GPUs.

3. **Test the server**:

   Send a POST request to `http://localhost:5000/process` with text data using tools like `curl` or Postman.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "Your sample text here."}' http://localhost:5000/process
```

## Contribution

Feel free to contribute to this project by opening issues or pull requests.
