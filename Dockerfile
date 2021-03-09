FROM pytorch/pytorch:latest

RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common gcc g++ sox

WORKDIR /app

COPY docker_requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
RUN python3 setup.py install
