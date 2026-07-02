#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def compute_rmse(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)) if mask.any() else np.nan

def main(
    csv_path="../RAW/sat_station_data.csv",
    output_dir="rmse_results"
):
    # 1) 데이터 로드
    df = pd.read_csv(csv_path, dtype={"STN": str})
    
    # 2) DATE 및 ALT 그룹 생성
    df["MONTH"] = df["MONTH"].astype(int).astype(str).str.zfill(2)
    df["DATE"]  = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"] + "-01")
    df["ALT_GROUP"] = pd.cut(
        df["ALT"].fillna(-9999),
        bins=[-np.inf, 500, 1500, np.inf],
        labels=["Low (<500m)", "Mid (500–1500m)", "High (≥1500m)"]
    )

    # 3) 전체 RMSE 계산
    overall_ml  = compute_rmse(df["SAT"].values, df["ML"].values)
    overall_cru = compute_rmse(df["SAT"].values, df["CRU"].values)
    print(f"Overall RMSE — ML: {overall_ml:.3f}, CRU: {overall_cru:.3f}")

    # 4) 관측소별 RMSE
    station_rmse = (
        df
        .groupby("STN")
        .apply(lambda g: pd.Series({
            "RMSE_ML":  compute_rmse(g["SAT"].values, g["ML"].values),
            "RMSE_CRU": compute_rmse(g["SAT"].values, g["CRU"].values)
        }))
        .reset_index()
    )

    # 5) 고도 그룹별 평균 RMSE
    rmse_with_alt = station_rmse.merge(
        df.drop_duplicates("STN")[["STN","ALT_GROUP"]],
        on="STN"
    )
    grp_rmse = rmse_with_alt.groupby("ALT_GROUP")[["RMSE_ML","RMSE_CRU"]].mean()

    # 6) 결과 저장 폴더 준비
    os.makedirs(output_dir, exist_ok=True)
    station_rmse.to_csv(f"{output_dir}/station_rmse.csv", index=False)
    grp_rmse.to_csv(f"{output_dir}/rmse_by_altitude_group.csv")
    print(f"Saved RMSE tables in `{output_dir}`")

    # 7) 시각화
    # 7.1) RMSE 분포 (Violin)
    melt = station_rmse.melt(
        id_vars="STN",
        value_vars=["RMSE_ML","RMSE_CRU"],
        var_name="Source",
        value_name="RMSE"
    )
    plt.figure(figsize=(8,5))
    sns.violinplot(data=melt, x="Source", y="RMSE", inner="quartile")
    plt.title("RMSE Distribution: ML vs CRU")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/violin_rmse.png")
    plt.close()

    # 7.2) 고도 그룹별 평균 RMSE 막대그래프
    plt.figure(figsize=(8,5))
    grp_rmse.plot(kind="bar")
    plt.title("Mean RMSE by Altitude Group")
    plt.ylabel("RMSE")
    plt.xlabel("Altitude Group")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bar_rmse_by_altitude.png")
    plt.close()

    # 7.3) 고도 그룹별 시계열
    ts = (
        df
        .groupby(["ALT_GROUP","DATE"])[["SAT","ML","CRU"]]
        .mean()
        .reset_index()
    )
    for grp, sub in ts.groupby("ALT_GROUP"):
        plt.figure(figsize=(12,6))
        plt.plot(sub["DATE"], sub["SAT"], label="Observed SAT")
        plt.plot(sub["DATE"], sub["ML"],  label="ML Estimate")
        plt.plot(sub["DATE"], sub["CRU"],label="CRU Data")
        plt.title(f"{grp} — SAT / ML / CRU Time Series")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        safe = grp.replace(" ", "_").replace("(", "").replace(")","").replace("≥","ge").replace("–","-")
        plt.savefig(f"{output_dir}/ts_{safe}.png")
        plt.close()

    print(f"All plots saved in `{output_dir}`")

if __name__ == "__main__":
    main()
