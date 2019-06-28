#FROM pytorch/pytorch
FROM nvcr.io/nvidia/pytorch:19.03-py3

RUN apt-get update && apt-get install -y \
build-essential \
git \
vim \
wget \
unzip \
python3-pyside \
python3.5 \
python3-dev \
python3-numpy \
python3-pip \
python3-tk


WORKDIR /workspace
COPY . /workspace
RUN chmod -R a+w /workspace

RUN pip3 install --upgrade --no-cache-dir -r requirements.txt

RUN ["python3.5", "test_docker.py"]
