FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN sed -i -e "s/ main[[:space:]]*\$/ main contrib non-free/" /etc/apt/sources.list

# Install system dependencies for building tick (build-essential, g++, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pip \
    g++ \
    cmake \
    git \
    swig \
    intel-mkl-full \
    && rm -rf /var/lib/apt/lists/*

# Install required Python dependencies first (to cache them in the image)
RUN pip install --upgrade pip
RUN pip install numpy cython==0.29.36 scipy==1.7.3 \
                scikit-learn==0.22.1 dill pandas \
                matplotlib numpydoc packaging pandas jupyter

# Install tick from PyPI (or source)
RUN pip install -v tick==0.6.0

# Set the working directory
WORKDIR /workspace

# Set the command to run on container start
CMD ["bash"]
