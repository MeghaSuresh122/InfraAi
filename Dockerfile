FROM python:3.14-slim

# Install uv
RUN pip install uv

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (excluding dev dependencies)
RUN uv sync --no-dev --frozen

# Copy the rest of the application code
COPY . .

# Expose the application port (as configured in main.py)
EXPOSE 8081

# Activate the virtual environment created by uv sync
ENV PATH="/app/.venv/bin:$PATH"

# Command to run the application
CMD ["python", "main.py"]
