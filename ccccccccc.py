import os

# Get HF_HOME environment variable
hf_home = os.getenv("HF_HOME")
print("HF cache is at:", hf_home)
