# Use an official Python runtime as the base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /docker

# Copy the requirements.txt file from your host to your current location in the container
COPY docker/requirements.txt .

# Copy the H2O python script from your host to your current location in the container
COPY docker/h2o_script.py .

# Copy the data into the container
COPY preprocessing/data/ ./data/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenJDK
RUN apt-get update \
    && apt-get install -y default-jdk \
    && apt-get clean


# Run h2o.py when the container launches
CMD ["python3", "./h2o_script.py"]
