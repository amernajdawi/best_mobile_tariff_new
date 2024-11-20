FROM python:3.10.14-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Configure Poetry to not create virtual environments inside project
RUN poetry config virtualenvs.in-project true

WORKDIR /workspace

# Create and activate virtual environment
RUN python -m venv .venv
ENV PATH="/workspace/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/workspace/.venv"

# Install IPython kernel in the virtual environment
RUN pip install ipykernel && \
    python -m ipykernel install --user --name=.venv --display-name="Python (.venv)"

# Pre-install common data science packages
RUN pip install pandas matplotlib seaborn numpy scikit-learn jupyter notebook

# Make sure the virtual environment is activated by default
RUN echo "source /workspace/.venv/bin/activate" >> ~/.bashrc






