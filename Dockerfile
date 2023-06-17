# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dependencies
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir --upgrade pip \
  && curl -sSL https://install.python-poetry.org | python3 -

# Set work directory
WORKDIR /app

# Copy project file
COPY ./pyproject.toml /app/

# Install project dependencies
RUN poetry config virtualenvs.create false \
  && poetry install --no-dev --no-interaction --no-ansi

# Copy all the rest of the project files
COPY . /app/

CMD ["python", "incremental_training_experiment.py"]
