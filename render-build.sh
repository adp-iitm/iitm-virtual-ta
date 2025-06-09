#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Install system dependencies for pytesseract
apt-get update
apt-get install -y tesseract-ocr libtesseract-dev

# Install Python dependencies
pip install -r requirements.txt
