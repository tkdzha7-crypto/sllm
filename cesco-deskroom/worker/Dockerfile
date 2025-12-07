# STEP 1: "Builder" Stage - Install uv
# We use the official Python image. This fixes the "not found" error.
FROM python:3.10-slim AS builder

# Install uv using pip
RUN pip install --no-cache-dir uv

# STEP 2: "Final" Stage - Create the lean worker image
FROM python:3.10-slim

# Install build dependencies for psycopg2 and ODBC drivers for pyodbc
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    unixodbc \
    unixodbc-dev \
    curl \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/11/prod bullseye main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /opt/deskroom_worker

# Copy the uv binary from the "builder" stage
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# Copy the Python library list
COPY worker/requirements.txt .

# Copy the application code
COPY flows/ ./flows/
COPY src ./src/

# Install the libraries using uv (this is fast)
# --system: Installs to the main Python environment, not a venv
RUN uv pip install --no-cache --system -r requirements.txt
