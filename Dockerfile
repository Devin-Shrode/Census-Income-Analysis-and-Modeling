# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files to container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn joblib numpy catboost pydantic

# Run FastAPI server
CMD ["uvicorn", "census_income_api:app", "--host", "0.0.0.0", "--port", "8000"]