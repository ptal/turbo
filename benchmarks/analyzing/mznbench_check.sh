#!/bin/sh

mzn-bench check-solutions -c 0 $1 &&
mzn-bench check-statuses $1
