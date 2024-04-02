#!/bin/sh

mzn-bench collect-objectives $1 ../campaign/$(basename $1)-objectives.csv
mzn-bench collect-statistics $1 ../campaign/$(basename $1).csv

