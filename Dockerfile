# Use Nvidia CUDA base image
FROM nvidia/cuda:11.1.1-runtime-ubuntu20.04

# Set the working directory
WORKDIR /usr/src/app

# Switch to the root user to install dependencies
USER root

# Add Nvidia GPG key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# Install necessary packages including Apache Spark and Python
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && \
    apt-get install -y python3 python3-pip wget unzip openjdk-11-jdk && \
    pip3 install numpy pyspark==3.5 flask spark-nlp==5.1.4

# Download the Spark archive with verbose output
RUN wget --verbose https://downloads.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz && \
    # Extract the archive and check for any errors
    tar -xzf spark-3.5.0-bin-hadoop3.tgz -C /opt/ || (echo "tar command failed" && exit 1) && \
    # Remove the archive and check for any errors
    rm spark-3.5.0-bin-hadoop3.tgz || (echo "rm command failed" && exit 1) && \
    wget -O  /tmp/spark-nlp-assembly-gpu-5.1.4.jar https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-gpu-assembly-5.1.4.jar && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up the necessary environment variables for Spark
ENV SPARK_HOME=/opt/spark-3.5.0-bin-hadoop3
ENV PATH=$SPARK_HOME/bin:$PATH
ENV PYSPARK_PYTHON=python3