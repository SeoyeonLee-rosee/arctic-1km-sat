#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
MODIS Water Reservoir monthly HDF 파서 (안정화 버전)
- VDATA 레코드 고정 언패킹 제거
- 필드 이름 기반 파싱(유연한 키 후보)
- 에러 개별 파일 단위 로깅 및 스킵
- start_ym ~ end_ym 기간 내 연-월별 파일 자동 매칭(파일명의 줄리안 데이→연-월 변환)
- 월별 결과 CSV 저장

예시:
python3 modis_wr_debug.py \
  --start_ym 2000-03 --end_ym 2022-04 \
  --value_field lake_evap_rate \
  --hdf_dir /data1/DATA_ARCHIVE/Satellite/MODIS/water_reservoir_monthly/ \
  --out_dir ../data/MOD/WR/
"""

import argparse
import datetime as dt
import os
import re
import sys
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# pyhdf는 시스템에 설치되어 있어야 합니다.
from pyhdf.HDF import HDF, HC, HDF4Error
from pyhdf.VS import VS
from pyhdf.SD import SD, SDC


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_year_month(s: str) -> dt.date:
    # "YYYY-MM" -> datetime.date(YYYY, MM, 1)
    y, m = s.strip().split("-")
    return dt.date(int(y), int(m), 1)


def iter_months(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    # start, end: month-start dates. inclusive range
    y, m = start.year, start.month
    while True:
        d = dt.date(y, m, 1)
        if d > end:
            break
        yield d
        # next month
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1


MODIS_FILE_RE = re.compile(r"MOD28C3\.A(?P<year>\d{4})(?P<doy>\d{3})\.\d{3}\.\d{13}\.hdf$")


def ymd_from_year_doy(year: int, doy: int) -> dt.date:
    return dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)


def scan_hdf_files_by_month(hdf_dir: str) -> Dict[str, List[str]]:
    """
    디렉토리(재귀)에서 MOD28C3 HDF 경로를 스캔해
    key="YYYY-MM" -> [파일 경로 리스트] 로 묶어 반환
    """
    bucket: Dict[str, List[str]] = {}
    root = Path(hdf_dir)
    if not root.exists():
        raise FileNotFoundError(f"hdf_dir not found: {hdf_dir}")

    for p in root.rglob("*.hdf"):
        m = MODIS_FILE_RE.search(p.name)
        if not m:
            continue
        year = int(m.group("year"))
        doy = int(m.group("doy"))
        d = ymd_from_year_doy(year, doy)
        key = f"{d.year:04d}-{d.month:02d}"
        bucket.setdefault(key, []).append(str(p))

    return bucket


# ---------------------------
# HDF 읽기 유틸
# ---------------------------

FIELD_CANDIDATES = {
    "lat": ["lat", "latitude", "Latitude", "LAT", "Lat"],
    "lon": ["lon", "longitude", "Longitude", "LON", "Lon"],
    "id": ["id", "lake_id", "wd_id", "WR_ID", "wr_id", "Lake_ID", "RES_ID"],
    "name": ["name", "lake_name", "NAME", "Lake_Name", "RES_NAME"],
}

VALUE_FIELD_FALLBACKS = [
    "lake_evaporation",
    "lake_evap_rate",
    "evap_rate",
    "evaporation",
    "evap",
    "value",
]


def _decode_bytes(v):
    if isinstance(v, (bytes, bytearray)):
        return v.decode("ascii", "ignore").strip("\x00 ").strip()
    return v


def _pick_key(candidates: List[str], field_idx: Dict[str, int]) -> Optional[str]:
    for k in candidates:
        if k in field_idx:
            return k
    return None


def read_vdata_rows(hdf_path: str, value_field: str) -> List[Dict]:
    """
    VDATA에서 이름 기반으로 필드 선택 후 전체 레코드 파싱.
    실패 시 예외를 던집니다.
    """
    rows: List[Dict] = []
    h = HDF(hdf_path, HC.READ)
    vs = h.vstart()
    vd = None
    try:
        # value_field 이름으로 바로 붙어보고, 없으면 find
        try:
            vd = vs.attach(value_field, "r")
        except HDF4Error:
            ref = vs.find(value_field)
            if ref is None:
                # 이름이 조금씩 다른 경우가 많으므로 VDATA들을 스캔하여 후보 찾기
                # vs.iterate()는 없으므로, common한 후보들로 한 번 더 시도
                for alt in VALUE_FIELD_FALLBACKS:
                    ref = vs.find(alt)
                    if ref is not None:
                        vd = vs.attach(ref)
                        break
                if vd is None:
                    raise RuntimeError(f"VDATA '{value_field}' not found")

            else:
                vd = vs.attach(ref)

        nrecs, interlace, field_names_str, size, name = vd.inquire()
        field_names = [s.strip() for s in field_names_str.split(",") if s.strip()]
        # pyhdf는 레코드 접근 시 튜플 인덱스 방식이므로, 이름→인덱스 맵 필요
        field_idx = {fn: i for i, fn in enumerate(field_names)}

        # value 키 확정
        val_key = value_field if value_field in field_idx else _pick_key(VALUE_FIELD_FALLBACKS, field_idx)
        if val_key is None:
            raise RuntimeError(f"값 필드를 찾을 수 없습니다. 후보={VALUE_FIELD_FALLBACKS}, 실제={field_names}")

        lat_key = _pick_key(FIELD_CANDIDATES["lat"], field_idx)
        lon_key = _pick_key(FIELD_CANDIDATES["lon"], field_idx)
        if lat_key is None or lon_key is None:
            raise RuntimeError(f"좌표 필드(lat/lon)를 찾을 수 없습니다. 실제={field_names}")

        id_key = _pick_key(FIELD_CANDIDATES["id"], field_idx)
        nm_key = _pick_key(FIELD_CANDIDATES["name"], field_idx)

        # 전체 읽기
        if nrecs <= 0:
            return rows
        recs = vd.read(nrecs)

        for rec in recs:
            def get(k):
                return _decode_bytes(rec[field_idx[k]])

            try:
                item = {
                    "id": _decode_bytes(rec[field_idx[id_key]]) if id_key else None,
                    "name": _decode_bytes(rec[field_idx[nm_key]]) if nm_key else None,
                    "lat": float(get(lat_key)),
                    "lon": float(get(lon_key)),
                    "value": float(get(val_key)),
                }
                rows.append(item)
            except Exception as e:
                # 개별 레코드 불량은 스킵
                continue

        return rows

    finally:
        try:
            if vd is not None:
                vd.detach()
        except Exception:
            pass
        try:
            vs.end()
        except Exception:
            pass
        try:
            h.close()
        except Exception:
            pass


def read_sds_rows(hdf_path: str, value_field: str) -> List[Dict]:
    """
    SDS로 값이 저장되어 있는 경우를 아주 보수적으로 시도.
    (대부분의 WR 제품은 VDATA에 메타+값이 같이 들어있어 SDS가 필요 없을 수 있습니다)
    여기서는 lat/lon/id/name을 SDS로 구할 방법이 없으면 빈 리스트 반환.
    """
    # 구현은 안전하게 no-op. 필요 시 확장하세요.
    try:
        sd = SD(hdf_path, SDC.READ)
        # 최소한 값 dataset 존재 여부만 확인
        if value_field in sd.datasets():
            # 좌표/ID를 함께 제공하는 SDS 명세가 없으므로 여기선 읽지 않음.
            pass
        sd.end()
    except Exception:
        pass
    return []


def read_lake_records(hdf_path: str, value_field: str) -> List[Dict]:
    # 1순위: VDATA
    try:
        rows = read_vdata_rows(hdf_path, value_field)
        if rows:
            return rows
    except Exception as e:
        raise

    # 2순위: SDS (좌표가 없으면 실익이 적어 현 단계에선 생략)
    return read_sds_rows(hdf_path, value_field)


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def save_csv(rows: List[Dict], out_csv: str) -> None:
    if not rows:
        # 비어있으면 이전 파일 삭제(있다면)
        if os.path.exists(out_csv):
            os.remove(out_csv)
        return
    ensure_dir(os.path.dirname(out_csv))
    fieldnames = ["id", "name", "lat", "lon", "value", "yyyymm"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_ym", required=True, help="YYYY-MM")
    ap.add_argument("--end_ym", required=True, help="YYYY-MM")
    ap.add_argument("--value_field", required=False, default="lake_evaporation",
                    help="VDATA/SDS에서 읽을 값 필드명(기본: lake_evaporation)")
    ap.add_argument("--hdf_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    start = parse_year_month(args.start_ym)
    end = parse_year_month(args.end_ym)

    # 파일 인덱싱
    files_by_month = scan_hdf_files_by_month(args.hdf_dir)

    # 처리할 달 정보 출력
    months = list(range(1, 13))
    log(f"[WR] 대상 월: {months}")

    for month_date in iter_months(start, end):
        key = f"{month_date.year:04d}-{month_date.month:02d}"

        # 파일 매칭
        files = sorted(files_by_month.get(key, []))
        if files:
            # 예시 파일 하나 보여주기
            sample = os.path.basename(files[-1])
            log(f"[WR] month={key[-2:]} 파일 매치: {len(files)} (예:{sample})")
        else:
            log(f"[WR] month={key[-2:]} 파일 매치: 0")
            log(f"[WR] month={key[-2:]} 데이터 없음")
            continue

        # 각 파일에서 레코드 수집
        all_rows: List[Dict] = []
        # 후보 VDATA 이름 힌트 출력
        log(f"[WR] VDATA 후보: {args.value_field or 'lake_evaporation'} (필드수:알 수 없음)")

        for fp in files:
            try:
                rows = read_lake_records(fp, args.value_field or "lake_evaporation")
                if not rows:
                    # SDS 경로를 타도 빈 리스트면 "데이터 없음" 취급(파일 자체는 정상일 수 있음)
                    raise RuntimeError("no parsable rows")
                # yyyymm 라벨 추가
                for r in rows:
                    r["yyyymm"] = key.replace("-", "")
                all_rows.extend(rows)
            except Exception as e:
                log(f"[WR] 파일 실패: {os.path.basename(fp)} → {e}")

        if all_rows:
            out_csv = os.path.join(args.out_dir, f"{key}_{args.value_field}.csv")
            save_csv(all_rows, out_csv)
        else:
            log(f"[WR] month={key[-2:]} 데이터 없음")

    log("[WR] 완료")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
