FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ENV HF_HOME="/data/csproject/hf_cache_home"

RUN echo "INSTALLS" \
	&& apt-get update --allow-releaseinfo-change \
    && apt-get -qq -y install git curl unzip sqlite3 wget

RUN echo "PYTHON INSTALLS" \
    && pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html \
    && pip install tqdm networkx torch-geometric

ENV PATH_TO_GAT="/atml-group-11/kally/src/"
