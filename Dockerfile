# Dockerfile

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860

# Streamlit runtime configuration
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.headless=true", "--browser.serverAddress=0.0.0.0"]



