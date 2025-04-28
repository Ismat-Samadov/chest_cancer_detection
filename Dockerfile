FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a models directory if it doesn't exist
RUN mkdir -p models

# Make the setup script executable
RUN chmod +x setup.sh

# Run the setup script and start the Streamlit app
CMD ["sh", "-c", "./setup.sh && streamlit run app.py"]