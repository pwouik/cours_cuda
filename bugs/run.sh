#!/bin/bash
set -e
set -x

cd "$( dirname "${BASH_SOURCE[0]}" )"

mkdir -p build
cd build
cmake .. $@
make
./bugs
