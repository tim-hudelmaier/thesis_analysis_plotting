# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dependencies
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl git build-essential python3-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir --upgrade pip

# Set work directory
WORKDIR /app
COPY requirements.txt /app/

# Install project dependencies
RUN pip install -r requirements.txt

# Copy all the rest of the project files
COPY . /app/

CMD ["python", "incremental_training_experiment.py"]
