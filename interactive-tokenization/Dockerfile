FROM python:3.10
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y --fix-missing && apt-get upgrade -y
RUN apt-get install -y hunspell libhunspell-dev

RUN mkdir /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install streamlit hunspell tokenizers transformers

COPY app.py .
COPY data/ /app/data/

ENV PORT 80
EXPOSE 80

CMD ["streamlit", "run", "app.py", "--server.port", "80"]
