#!/bin/bash

# Check if the 'venv' directory exists
if [ -d "venv" ]; then
    echo "Virtual environment 'venv' already exists. Activating..."
    # Activate the virtual environment
    source venv/bin/activate
else
    echo "Virtual environment 'venv' does not exist. Creating..."
    # Create a new virtual environment
    python3 -m venv venv
    # Activate the newly created virtual environment
    source venv/bin/activate
fi

# load Python module
module load Python


# install requirements into virtual environment
pip install -r requirements.txt

# run the Python script
python --project=. single_mcsweep.py 
