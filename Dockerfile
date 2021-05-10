# Based on a PyTorch docker image that matches the minimum requirements: PyTorch 1.1.0
FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

RUN python -m pip install --upgrade pip

COPY . /enet
WORKDIR /enet
RUN pip install -r requirements.txt
