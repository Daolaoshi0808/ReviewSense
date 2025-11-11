# Use a Python base image
FROM python:3.11-slim


SHELL ["/bin/bash", "-c"] 


# Install system dependencies
RUN apt-get update -qq && apt-get upgrade -qq && \
    apt-get install -qq man wget sudo vim tmux

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file and install dependencies
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python scripts and data into the container
COPY ReviewSense.py /app/
COPY ReviewSense_frontend.py /app/
COPY final_review_chunked_df.csv /app/
# Copy the model directory
COPY my_model/ /app/my_model/
COPY .env /app/  


# Create a command that allows running either script
CMD ["python", "ReviewSense.py"]
