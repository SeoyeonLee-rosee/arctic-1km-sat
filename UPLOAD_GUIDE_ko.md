# 업로드 가이드 (GitHub + Zenodo) & 논문 문구 초안

## A. 논문에 넣을 문구 초안 (영문 — DOI만 채우면 됨)

**Code availability**
> The code used to train the machine-learning air-temperature downscaling model
> and to generate all analyses and figures in this study is openly available at
> Zenodo (https://doi.org/10.5281/zenodo.XXXXXXX) and was developed at
> https://github.com/<사용자명>/<저장소명>.

**Data availability**
> All primary datasets are publicly available from their original providers:
> MODIS products (MOD10A2, MOD11A2, MOD13A3, MOD17, MOD15) from NASA LP DAAC;
> CRU TS4.06 from the Climatic Research Unit (University of East Anglia);
> ERA5 from the Copernicus Climate Change Service; MERRA-2 from NASA GMAO;
> CPC and GHCN-CAMS from NOAA PSL; and ETOPO 2022 from NOAA NCEI. The
> machine-learning-derived surface-air-temperature product generated in this
> study is archived at Zenodo (https://doi.org/10.5281/zenodo.XXXXXXX).
> In-situ station observations are available from [제공기관].

---

## B. GitHub 올리기 (계정 만들기 → 저장소 생성 → 코드 push)

### 1) 계정 만들기
1. https://github.com/signup 접속 → 이메일(회사메일 가능)·비밀번호·아이디 입력
2. 이메일 인증 클릭

### 2) 새 저장소(repository) 만들기
1. 오른쪽 위 **+** → **New repository**
2. Repository name 예: `arctic-sat-ml`
3. **Private** 선택 (심사 중엔 비공개 권장)
4. README/…는 체크하지 말고 (이미 있음) → **Create repository**

### 3) 코드 올리기
GitHub이 만든 저장소 페이지에 명령이 뜹니다. 아래를 **이 code/ 폴더에서** 터미널에 붙여넣으세요
(로컬 git 커밋은 제가 이미 만들어 뒀습니다):

```bash
cd /data1/sylee/ML/ver1/code
git remote add origin https://github.com/<사용자명>/arctic-sat-ml.git
git branch -M main
git push -u origin main
```
- push할 때 GitHub **아이디**와 **Personal Access Token**(비밀번호 대신)을 물어봅니다.
  Token은 GitHub → Settings → Developer settings → Personal access tokens → *Generate*
  (repo 권한 체크)에서 발급.

### (심사용) 리뷰어에게 비공개 저장소 보여주기
- 저장소 **Settings → Collaborators**에 리뷰어 초대, 또는
- 아래 Zenodo의 **reviewer link**를 논문에 첨부(더 간단, 추천).

---

## C. Zenodo에서 DOI 받기 (인용용 — Nature가 원하는 것)

### 1) 계정 (GitHub로 로그인 가능)
1. https://zenodo.org 접속 → **Sign up** → **Log in with GitHub** 선택하면 계정 연동 끝

### 2) GitHub 저장소 연동
1. Zenodo 오른쪽 위 이름 → **GitHub**
2. 목록에서 방금 만든 저장소 토글을 **ON**

### 3) 릴리스(release) 만들면 DOI 자동 발급
1. GitHub 저장소 → 오른쪽 **Releases** → **Create a new release**
2. Tag 예: `v1.0.0`, 제목 아무거나 → **Publish release**
3. 잠시 뒤 Zenodo에 자동으로 박제되고 **DOI**(`10.5281/zenodo.…`)가 생깁니다.
4. 그 DOI를 위 A의 문구 `XXXXXXX` 자리에 넣으면 끝.

> ⚠️ Nature Communications는 **GitHub 주소만이 아니라 이 Zenodo DOI**를 요구합니다.

---

## D. 올리기 전 마지막 체크
- [ ] `LICENSE`의 `[YOUR NAME]`을 실제 이름으로 수정
- [ ] `README.md`의 제목·`[fill in]` 항목(논문 제목, 관측소 제공기관) 채우기
- [ ] 스크립트 상단 외부경로(ETOPO2022, MODIS 아카이브)가 본인 환경 기준인지 확인
- [ ] double-blind 심사면 GitHub 대신 익명 링크 사용 (기본 single-blind면 실명 GitHub OK)
