# Use the official Python image from Docker Hub
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the contents of your repository to the working directory
COPY . /app

# Install the required packages
RUN pip install -r requirements.txt

# Expose the port that the app will run on
EXPOSE 8080

# Set the command to run your app with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
