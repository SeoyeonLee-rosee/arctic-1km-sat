# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np

RAW_DIR = "../RAW"
mask = np.load(os.path.join(RAW_DIR, "mod44w_land_mask.npy")).astype(bool)

for mm in range(1, 13):
    fp = os.path.join(RAW_DIR, f"sat_{mm:02d}.npy")
    arr = np.load(fp)   # (34,time,1200,1200)

    arr = np.where(mask[:, None, :, :], arr, np.nan)

    out = os.path.join(RAW_DIR, f"sat_{mm:02d}_land.npy")
    np.save(out, arr.astype(np.float32))
    print("saved", out)