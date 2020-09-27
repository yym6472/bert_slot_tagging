#!/bin/bash

python3 test.py --output_dir ./output/$1
cd ./data/tianchi
cp ./chusai_xuanshou/*.ann ./test_output/
zip -q -r test_output.zip ./test_output
python3 check_zip.py