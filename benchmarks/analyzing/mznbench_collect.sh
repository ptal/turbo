#!/bin/sh

mzn-bench collect-objectives $1 objectives.csv
mzn-bench collect-statistics $1 statistics.csv

