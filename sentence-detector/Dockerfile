# Use custom spark-cuda base image
FROM spark-cuda:latest

# Download the required Spark NLP model
RUN wget https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip \
    && unzip sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip -d model \
    && rm sentence_detector_dl_xx_2.7.0_2.4_1609610616998.zip

    
# Copy the protobuf definition file into the container and generate the Python gRPC files
#COPY sentence_detection.proto .
#RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. sentence_detection.proto

# Copy the server files into the container
COPY fastapi_server.py ./fastapi_server.py
#COPY grpc_server.py ./grpc_server.py

# Copy supervisord configuration file into the container
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set the entrypoint to run supervisord
ENTRYPOINT ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
