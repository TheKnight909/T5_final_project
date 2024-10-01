# Step 1: Use an official Python runtime as a parent image
FROM python:3.8

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . .

# Step 4: Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Step 5: Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Define environment variables (optional)
# ENV API_KEY=<your_api_key>
# ENV OTHER_CONFIG=<other_configuration>

# Step 7: Expose the port your app runs on
EXPOSE 8501

# Step 8: Command to run your application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
