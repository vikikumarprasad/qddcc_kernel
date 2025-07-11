# Use a Miniconda base image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app

# Copy your Conda environment file
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Activate the environment and install any additional packages if needed
# This part is optional if all dependencies are in environment.yml
# RUN conda activate myenv && pip install some_other_package

# Set the default shell to bash and activate the conda environment
# This is crucial for subsequent commands to run within the conda environment
SHELL ["/bin/bash", "--login", "-c"]
RUN conda activate qml_research

# You can define a default command to run, but we'll override it in Slurm
# CMD ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "your_default_script.py"]