#!/bin/sh
# build the master TeX and compile it (run in 資料集/00_全体の記録)
set -e
python 作成スクリプト/build.py
C:/Users/sumat/tools/tectonic/tectonic.exe -X compile LLM-BLM_全体の記録.tex
