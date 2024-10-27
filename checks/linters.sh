#!/bin/bash

echo '=========== RUN LINTERS FOR LIBRARY ==========='
echo '--BLACK--' && python3 -m black eyefeatures;
echo '--ISORT--' && python3 -m isort eyefeatures;
echo '--FLAKE8--' && python3 -m flake8 eyefeatures;
