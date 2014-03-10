#!/bin/bash
cat ../data/train.txt | ./mapper.py | sort | ./reducer.py > output.txt
./check.py output.txt ../data/duplicates.txt

