#!/bin/bash


if [ ! -d "./data" ]; then
    mkdir ./data
fi

if [ ! -d "./data/UCIHAR" ]; then
    wget https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip -O ./data/UCIHAR.zip

    unzip ./data/UCIHAR.zip -d ./data/UCIHAR
    rm ./data/UCIHAR.zip

    unzip './data/UCIHAR/UCI HAR Dataset.zip' -d ./data/UCIHAR
    rm './data/UCIHAR/UCI HAR Dataset.zip'
    rm './data/UCIHAR/UCI HAR Dataset.names'
    rm -rf './data/UCIHAR/__MACOSX'
    mv './data/UCIHAR/UCI HAR Dataset/train' './data/UCIHAR/train' 
    mv './data/UCIHAR/UCI HAR Dataset/test' './data/UCIHAR/test'
    rm -rf './data/UCIHAR/UCI HAR Dataset'
fi

