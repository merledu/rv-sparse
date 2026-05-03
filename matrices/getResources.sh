#!/bin/bash

# Traverse the "mat_resources" file
while IFS= read -r link; do
    # Download the link using wget
    wget "$link"

    # Extract the tar file
    tar -xf "$(basename "$link")"

    # Delete the tar file
    rm "$(basename "$link")"
done < mat_resources.txt
