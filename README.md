# NLP Testing Base Image

This is a custom Docker base image designed for Natural Language Processing (NLP) projects, leveraging Nvidia CUDA for GPU acceleration. It's especially built for testing projects that require Apache Spark and Spark NLP.

## Features

- Based on Nvidia's CUDA 11.1.1 runtime for Ubuntu 20.04.
- Comes pre-installed with Python 3, PySpark 3.5.0, and Spark NLP 5.1.4.
- Set up for GPU acceleration with Spark NLP.

## Building the Image

To build the image, you'll need Docker installed. Navigate to the directory containing the Dockerfile and execute:

```bash
docker build -t [your-image-name]:[tag] .
```

Replace `[your-image-name]` with your desired image name and `[tag]` with the desired version or tag.

## Environment Variables

- `PYSPARK_PYTHON`: Set to `python3` to ensure PySpark runs with Python 3.

## Installed Packages

The image comes with the following key packages:

- PySpark 3.5.0
- Python 3
- Spark NLP 5.1.4 (GPU version)
- Flask
- NumPy

## Additional Notes

- Always ensure that your host machine's Nvidia drivers are compatible with the CUDA version used in this image.
- For any issues with Spark or Spark NLP, refer to their respective official documentation or community forums.

## Common Errors with GPU Containers

### Error: 'Could not load dynamic library 'libcuda.so.1'

**Error Message**:
Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory


**Cause**:

This error occurs when a GPU-enabled Docker container attempts to access the NVIDIA GPU driver's shared library (`libcuda.so.1`) but cannot find it. The reason is that this library, part of the NVIDIA drivers on the host system, is dynamically made available to the container at runtime via the NVIDIA Container Toolkit.

If you don't use the `--gpus` flag when running your Docker container, the NVIDIA runtime won't be activated, and the container won't have access to the GPU or the associated driver libraries.

**Solution**:

Always use the `--gpus` flag when running your GPU-enabled Docker container to ensure the necessary GPU driver libraries are accessible and the container can communicate with the GPU. For example:

```bash
docker run --gpus all -it your_image_name
```

By using this flag, the NVIDIA Container Toolkit will facilitate the container's access to the required NVIDIA driver libraries from the host, allowing GPU-accelerated applications to run without issues.

## Contribution

Feel free to fork, modify, and make pull requests to this repository. All contributions are welcome!
