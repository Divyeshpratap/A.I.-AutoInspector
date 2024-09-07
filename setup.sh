#!/bin/bash

# Step 1: Create and activate virtual environment
python3 -m venv carDDEnv
source carDDEnv/bin/activate

# Step 2: Install dependencies for Image Processing
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install openmim
mim install mmdet==2.25.1
mim install mmcv-full==1.7.0
pip install yapf==0.40.1

# Step 3: Install general dependencies
pip install -r requirements.txt

# Step 4: Install and configure Ollama for Ubuntu/Linux
# Download the Ollama binary
curl -LO https://ollama.com/download/ollama-linux.tar.gz

# Extract the binary
tar -xzf ollama-linux.tar.gz

# Move the binary to a directory in your PATH
sudo mv ollama /usr/local/bin/

# Make sure the binary is executable
sudo chmod +x /usr/local/bin/ollama

# Confirm installation
ollama --version

echo "Setup complete. You can now run your application with 'flask run'."
