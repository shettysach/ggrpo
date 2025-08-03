#!/bin/bash

# Define the URL (IP address of the server)
url="203.57.40.162"

# Define the SSH key and port
ssh_key="~/.ssh/id_ed25519"
port="10214"

# Download the files using scp (remote to local)
scp -i $ssh_key -p -O -P $port root@$url:/workspace/results.csv ./results.csv
