# Install requirements

    python3 -m pip install -r requirements.txt
or 
    
    python -m pip install -r requirements.txt


# Create a settings.py

Copy `settings_example.py` in the `/settings` folder and name it `settings.py`. Fill in the keys.


# Run Virtual Machine

## Setup
### Install Google Client (Mac)
    brew install --cask google-cloud-sdk

### Install Google Client (Windows)
    winget install Google.CloudSDK

### Login
    gcloud auth login
    gcloud config set project auto-design-434413

## Run machine
    gcloud compute ssh furniture-detection-machine --zone europe-north2-b