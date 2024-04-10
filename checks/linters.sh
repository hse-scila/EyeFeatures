#!/bin/bash

echo '=========== RUN LINTERS FOR LIBRARY ==========='
echo '--BLACK--' && python3 -m black eyetracking;
echo '--ISORT--' && python3 -m isort eyetracking;
echo '--FLAKE8--' && python3 -m flake8 eyetracking;
