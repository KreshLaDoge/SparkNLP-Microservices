# Use Nvidia CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set the working directory
WORKDIR /usr/src/app

# Switch to the root user to install dependencies
USER root

# Add Nvidia GPG key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# Install necessary packages including PySpark, Python, and JDE headless
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && \
    apt-get install -y python3 python3-pip wget unzip openjdk-11-jre-headless && \
    pip3 install numpy pyspark==3.5 flask spark-nlp==5.1.4

# Fetch Spark-NLP and clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download the JAR directly and place it in the PySpark jars directory
RUN wget -P /usr/local/lib/python3.10/dist-packages/pyspark/jars/ https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-5.1.4.jar

# Set Python environment variable for PySpark
ENV PYSPARK_PYTHON=python3