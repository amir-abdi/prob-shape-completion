#!/bin/bash

FILENAME=mandibles117
wget -O ./$FILENAME.zip "www.ece.ubc.ca/~amirabdi/public/mandibles117.zip"
echo "Data downloaded"

mkdir data
unzip ./$FILENAME.zip -d ./data
echo "Data unzipped into directory: $PWD/data"

export DATASETS=$PWD/data
echo 'Environment variable $DATASETS set to' $DATASETS

