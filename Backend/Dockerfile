FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim

WORKDIR /app

# Installing python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying the app code

COPY . .

# Expose the port and run Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]