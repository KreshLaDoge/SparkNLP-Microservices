# Use Nvidia CUDA base image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set the working directory
WORKDIR /usr/src/app

# Switch to the root user to install dependencies
USER root

# Add Nvidia GPG key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# Install necessary packages including Apache Spark and Python
RUN apt-get update

RUN apt-get install -y python3 python3-pip wget unzip openjdk-11-jdk

RUN pip3 install numpy pyspark==3.5 flask spark-nlp==5.1.3

# Download the Spark archive with verbose output
RUN wget --verbose https://downloads.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
# Now extract the archive and check for any errors
RUN tar -xzf spark-3.5.0-bin-hadoop3.tgz -C /opt/ || (echo "tar command failed" && exit 1)

# Now remove the archive and check for any errors
RUN rm spark-3.5.0-bin-hadoop3.tgz || (echo "rm command failed" && exit 1)


RUN wget https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip \
    && unzip sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip -d model \
    && rm sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip

RUN wget -O  /tmp/spark-nlp-assembly-5.1.3.jar https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-5.1.3.jar

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Set up the necessary environment variables for Spark
ENV SPARK_HOME=/opt/spark-3.5.0-bin-hadoop3
ENV PATH=$SPARK_HOME/bin:$PATH
ENV PYSPARK_PYTHON=python3

# Copy the application file into the container
COPY app.py .

# Set the entrypoint to run the Flask application
ENTRYPOINT ["python3", "app.py"]
