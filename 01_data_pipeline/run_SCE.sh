#!/bin/bash
# SCE/SCS 파이프라인(최신): 계산 -> AVE&TREND -> 지역 지도
set -e
D=$(cd "$(dirname "$0")" && pwd)
echo "[1/3] SCE/SCS 계산";  python3 "$D/snowcover_sce_scs_calc.py"
echo "[2/3] AVE & TREND";   python3 "$D/snowcover_trend_calc.py"
echo "[3/3] 지역 지도";     python3 "$D/../22_snow_interaction/snowcover_region_maps.py"
echo "DONE."
