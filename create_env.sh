#!/usr/bin/env bash

echo "Here I want the magic of Data Science to be happening!!!"

echo "But first let's check if we have right setup here:"

echo "Now show me the version of tensorflow:"

python -c 'import tensorflow as tf; print(tf.__VERSION__)'

echo "Ok, show me the version of keras:"

python -c 'import keras; print(keras.__VERSION__)'

echo "Good, looks like everything is more or less fine."