#/bin/bash

docker build -t interactive-tokenization:${1:-0.0.1} .
docker tag interactive-tokenization:${1:-0.0.1} docker.texttechnologylab.org/interactive-tokenization:latest
docker push docker.texttechnologylab.org/interactive-tokenization:latest