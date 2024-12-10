FROM python:3.12-slim

WORKDIR /objdetection

COPY ./requirements.txt /objdetection/requirements.txt

ENV DEBIAN_FRONTEMD noninteractive

RUN apt-get clean && apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size


RUN pip install --upgrade pip

RUN pip3 install --default-timeout=100 -r requirements.txt --no-cache-dir
RUN pip3 install --no-cache-dir fastapi[all]

#Don't forget to add the .pt models to /app dir
COPY ./app /objdetection/app
EXPOSE 5000

# set --workers param to the uvicorn
CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]