FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN apt-get update &&\
apt-get -y install build-essential&&\
apt-get -y install python3.8&&\
apt-get -y install python3-pip&&\
pip3 install --no-cache-dir datasets==2.1.0 transformers==4.18.0 pandas==1.4.2 tqdm==4.64.3 gradio==3.0.17 more_itertools==8.13.0 numpy==1.19.2
