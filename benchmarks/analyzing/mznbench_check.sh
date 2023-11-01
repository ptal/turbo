#!/bin/sh

mzn-bench check-solutions $1 &&
mzn-bench check-statuses $1
