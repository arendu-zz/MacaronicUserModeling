#!/bin/sh
set -e
python setup.py build_ext --inplace
ln -s array_utils/c_array_utils_64.so c_array_utils_64.so
ln -s array_utils/c_array_utils.so c_array_utils.so

