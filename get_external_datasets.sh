#!/bin/bash

# Set your file ID and desired file name
fileId="19C4MvZ6JAMHAnnBZUyS0xSRo8Q9NEU9w"
fileName="data.zip"

# Create a data directory if it doesn't exist
mkdir -p data

# Navigate to the data directory
cd data

# Begin download process
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' ./cookie)&id=${fileId}" -o ${fileName}

# Unzip the contents of the downloaded file
unzip -o ${fileName}

# Clean up cookie file
rm -f ./cookie

# Navigate back to the original directory
cd ..

# Echo completion
echo "Download and extraction complete."