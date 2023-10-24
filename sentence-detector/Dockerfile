# Use the apache/spark-py image as the base image
FROM apache/spark-py

# Set the working directory
WORKDIR /usr/src/app

USER root

# Install necessary packages and download model
RUN apt-get update && \
    apt-get install -y python3-pip unzip wget && \
    pip3 install numpy pyspark==3.5 flask spark-nlp==5.1.3 && \
    wget https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip && \
    unzip sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip -d model && \
    rm sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip && \
    wget -O  /tmp/spark-nlp-assembly-5.1.3.jar https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-5.1.3.jar && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY app.py .

# Set the entrypoint to run the Flask application
ENTRYPOINT ["python3", "app.py"]
