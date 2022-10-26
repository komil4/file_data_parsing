# первый этап
FROM python:3.10.8 AS builder

WORKDIR /server

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx poppler-utils tesseract-ocr libtesseract-dev

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

COPY ./src ./src

RUN mkdir -p images files csvs results

CMD python3 ./src/main.py -host 0.0.0.0

EXPOSE 3000