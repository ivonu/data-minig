#!/bin/bash
cat ../data/training | ./mapper.py | sort | ./reducer.py > reducer_out.txt
./predict.py reducer_out.txt ../data/training > ../visual_test/prediction.txt
cd ../visual_test
./visual_test.py