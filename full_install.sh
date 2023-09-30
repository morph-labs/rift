#!/bin/bash

CWD=$(pwd)

cd ./editors/rift-vscode

bash reinstall.sh

cd $CWD

pip install -e ./rift-engine

