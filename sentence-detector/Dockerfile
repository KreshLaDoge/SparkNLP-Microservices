# Use custom spark-cuda base image
FROM spark-cuda:latest

# Install FastAPI, Uvicorn and Gunicorn
RUN pip install fastapi uvicorn gunicorn pydantic

RUN wget https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip \
    && unzip sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip -d model \
    && rm sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip

# Copy the application file into the container
COPY app.py .

# Set the entrypoint to run the FastAPI application with Uvicorn and Gunicorn
ENTRYPOINT ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:5000", "app:app"]
