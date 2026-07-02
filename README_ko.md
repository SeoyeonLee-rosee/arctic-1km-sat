# ver1/code — 코드 정리본 (한글 설명)

원본 폴더(`script`, `scripts`, `scripts_new`, `scripts_newest`, `scripts_fin` 및 `script/script`의 노트북)에
흩어져 여러 번 반복(버전)되던 코드를, **같은 계산은 가장 최신 버전 1개만** 남겨 주제별로 모은 정리본입니다.
저널(Nature Communications) 투고용 저장소로 바로 올릴 수 있도록 정리했습니다.

- **원본은 그대로 보존**됩니다(이 폴더는 복사본). 옛 버전이 필요하면 원본 폴더를 보세요.
- 영문 설명·저장소 안내는 [`README.md`](README.md).

## 이번 코드 정리에서 한 일

1. **ML·자료처리 노트북 포함** — `script/script/`의 핵심 노트북 6개를 `00_ml_pipeline/`에 넣고,
   용량을 줄이고 깔끔하게 하려고 **셀 출력(그림)을 모두 비웠습니다**(예: 4.3MB → 32KB).
2. **경로 보정** — 파일들이 `../data` 같은 경로를 쓰는데, `code/<폴더>/`로 옮겨져 어긋난 것을,
   실행 시 작업폴더를 자동으로 `code/`(=`ver1` 바로 아래)로 고정하도록 각 파일 상단에 3줄을 넣어
   원래대로 데이터를 찾게 했습니다. (`SCRIPT_DIR`을 쓰던 파일은 한 단계 상향)
3. **SCE 파이프라인 교정** — 최신 `SCE_1/2/3`(12/5)로 교체(초기에 옛 `SC_cal`을 넣었던 것 수정).
4. **문법 오류 정리** — f-string 백슬래시 오류 수정, 미완성 파일(`check_WR.py`) 제외.
5. **`.sh` 실행스크립트**를 새 파일명에 맞게 재작성.
6. 저장소 파일 추가: `README.md`, `LICENSE`(MIT), `requirements.txt`, `.gitignore`.

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `00_ml_pipeline/` | **ML 핵심 파이프라인(노트북)** — SAT 산출물 생성 |
| `01_data_pipeline/` | 원자료(MODIS/재분석) → 배열 (SCE/SCS, SOS, GPP_max, 재격자화 등) |
| `02_figure_prep/` | 본문 Figure 3용 중간 데이터셋 빌더 |
| `10_main_figures/` | 논문 본문 그림 Figure 1~4 |
| `11_supp_figures/` | 보충 그림 Figure S3~S9 |
| `20_temperature/` | 추가 온도 분석 |
| `21_lst/` | 주야 지표온도 분석 |
| `22_snow_interaction/` | 적설↔온도/GPP 상호작용 + SCE/SCS 지도 |
| `23_gpp_vegetation/` | GPP/FPAR 분석 |
| `24_station_validation/` | 관측소 추출·검증 |

## 00_ml_pipeline 노트북 (출처)

| code/ 파일 | 원본 | 설명 |
|---|---|---|
| 01_station_data_prep.ipynb | script/script/1.station.ipynb | MODIS+관측소 학습자료 추출 |
| 02_station_map.ipynb | 2.station_map.ipynb | 관측소·연구지역 지도 |
| 03_ML_training.ipynb | 3.ML.ipynb | **RF/LGBM/XGB 월별 12모델 학습(핵심)** |
| 04_CRU_reference.ipynb | 4.CRU.ipynb | CRU 기준 기후 처리 |
| 06_ML_apply_to_MODIS.ipynb | 6.ML_MODIS.ipynb | **전 격자 적용 → SAT 산출물** |
| 09_constraint_validation.ipynb | 9.constraint_new.ipynb | CRU 대비 검증 |

## 주요 .py 출처 (요약)

- `01_data_pipeline/`: modis_1km_monthly←scripts_newest, snowcover_sce_scs_calc←scripts/SCE_1,
  snowcover_trend_calc←scripts/SCE_2, start_of_season_calc←scripts_newest/cal_sos,
  reanalysis_mean_trend←scripts_newest/data_make, gpp_max_1~3←scripts/gpp_max_1~3,
  regrid_*←script/data_0.x, sat_landmask←scripts_fin/sat_mask
- `10_main_figures/`: fig1~4 ← scripts_fin/figure1~4 (+figure3_map)
- `11_supp_figures/`: figS3~S9 ← scripts_fin/fs3~fs9
- `22_snow_interaction/snowcover_region_maps.py` ← scripts/SCE_3
- 그 외 20/21/23/24 폴더: 원본 `script`·`scripts` 최신본 (자세한 표는 이전 정리 기록 참조)

> 전체 파일별 출처·날짜 표가 더 필요하면 말씀해 주세요. 원본은 모두 보존돼 있어 언제든 대조 가능합니다.

## 검토 권장 (사람 판단 필요)

1. **논문 범위 확인** — CMIP6/시나리오 노트북(8,10,11,12)은 미래전망을 논문에 넣을 때만 추가하면 됩니다(현재 제외).
2. **fig3 변형** — figure3_3panel / figure3_error 등 레이아웃 변형 중 본문 채택본 확정.
3. **외부 데이터 경로** — 스크립트 상단의 ETOPO2022·MODIS 아카이브 절대경로는 본인 환경에 맞게 확인.
4. **공유모듈** — 재현성 보장을 위해 코드 로직은 바꾸지 않았습니다. 원하시면 공통요소(지역정의·ETOPO
   로딩 등)를 `common.py`로 추출하는 리팩터링을 그림 재현 확인과 함께 진행할 수 있습니다.
