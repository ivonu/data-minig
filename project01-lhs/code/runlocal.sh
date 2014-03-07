#!/bin/bash
cat ../data/train.txt | ./mapper.py | sort | ./reducer.py
