# NLP Testing Base Image

This is a custom Docker base image designed for Natural Language Processing (NLP) projects, leveraging Nvidia CUDA for GPU acceleration. It's especially built for testing projects that require Apache Spark and Spark NLP.

## Features

- Based on Nvidia's CUDA 11.1.1 runtime for Ubuntu 20.04.
- Comes pre-installed with Python 3, Apache Spark 3.5.0, and Spark NLP 5.1.4.
- Set up for GPU acceleration with Spark NLP.

## Building the Image

To build the image, you'll need Docker installed. Navigate to the directory containing the Dockerfile and execute:

```bash
docker build -t [your-image-name]:[tag] .
```

Replace `[your-image-name]` with your desired image name and `[tag]` with the desired version or tag.

## Environment Variables

- `SPARK_HOME`: This is set to the directory where Apache Spark is installed.
- `PATH`: Updated to include the `bin` directory of Apache Spark.
- `PYSPARK_PYTHON`: Set to `python3` to ensure PySpark runs with Python 3.

## Installed Packages

The image comes with the following key packages:

- Apache Spark 3.5.0
- Python 3
- Spark NLP 5.1.4 (GPU version)
- Flask
- NumPy

## Additional Notes

- Always ensure that your host machine's Nvidia drivers are compatible with the CUDA version used in this image.
- For any issues with Spark or Spark NLP, refer to their respective official documentation or community forums.

## Contribution

Feel free to fork, modify, and make pull requests to this repository. All contributions are welcome!
