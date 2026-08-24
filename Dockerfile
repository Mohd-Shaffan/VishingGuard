FROM python:3.12-slim

# Create a new user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /code

# Install dependencies first for Docker cache efficiency
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user . /code

# Fix pickling/version mismatch by retraining the model natively inside the container
RUN python train_vishing_model.py

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]