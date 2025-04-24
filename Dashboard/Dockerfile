FROM python:3.9-slim

WORKDIR /app

# Install Streamlit and co
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dashboard code
COPY . .

# Expose and launch Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]