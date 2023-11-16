FROM python:3.11.3
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal


COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

CMD ["python", "app.py"]
