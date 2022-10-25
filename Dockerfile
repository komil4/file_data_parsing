# первый этап
FROM python:3.10.8 AS builder

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y

RUN pip install --upgrade pip

RUN adduser server_user
USER server_user

WORKDIR /home/server_user

ENV PATH="/home/server_user/.local/bin:${PATH}"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=server_user:server_user ./src ./src

EXPOSE 3000

CMD python3 ./src/main.py -host 0.0.0.0