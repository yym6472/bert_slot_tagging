#!/bin/bash

# 该文件从 ./data/tianchi/chusai_xuanshou 中拷贝ann，生成提交的zip文件，并验证文件格式合法性
# 根目录下运行: sh generate_upload_zip.sh

cd ./data/tianchi
cp ./chusai_xuanshou/*.ann ./test_output/
zip -q -r test_output.zip ./test_output
python3 check_zip.py