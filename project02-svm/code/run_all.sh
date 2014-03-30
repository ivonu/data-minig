#!/bin/bash
cat ../data/training.txt | ./mapper.py | sort | ./reducer.py > reducer_out.txt
./predict.py reducer_out.txt ../data/training.txt > ../visual_test/prediction.txt
cd ../visual_test
./visual_test.py
