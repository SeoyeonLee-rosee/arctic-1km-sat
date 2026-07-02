#!/bin/bash
# GPP MAX 파이프라인: 월별정렬 -> MAX/TREND -> 지도/지역
set -e
D=$(cd "$(dirname "$0")" && pwd)
echo "[1/3] gpp_max_1_monthly_align.py"; python3 "$D/gpp_max_1_monthly_align.py"
echo "[2/3] gpp_max_2_max_trend.py";     python3 "$D/gpp_max_2_max_trend.py"
echo "[3/3] gpp_max_3_maps_regions.py";  python3 "$D/gpp_max_3_maps_regions.py"
echo "DONE. 결과: ../FIGs/GPP_MAX/"
