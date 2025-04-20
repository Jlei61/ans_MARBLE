FROM continuumio/miniconda3

WORKDIR /app

# Copy environment file
COPY environment.yaml .

# Create conda environment
RUN conda env create -f environment.yaml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "marble", "/bin/bash", "-c"]

# Activate conda environment
RUN echo "conda activate marble" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Set PYTHONPATH to recognize marble package
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command
CMD ["conda", "run", "--no-capture-output", "-n", "marble", "python", "-c", "import marble; print('MARBLE environment loaded successfully')"] 