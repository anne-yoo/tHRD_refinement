#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import sys
import joblib
import pandas as pd
import sklearn.ensemble.forest  # Python 2에서는 원래 이 이름
sys.modules['sklearn.ensemble._forest'] = sklearn.ensemble.forest  # 호환 alias

# -----------------------------
# 사용자 입력
# -----------------------------
lane = 5
seed = 120
MODEL_PATH = "/home/hyeongu/DATA5/hyeongu/TCGA-OV/data/TU/RF_result/platinum_model_00-2/RCV/model/lane_5/RF_based_std_120_transcripts.h5"
TU_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/POLO/tHRDmodel/83_POLO_TU_old.txt"

# 출력 디렉토리 설정
OUT_DIR = os.path.join(os.path.dirname(TU_PATH), "RF_result", "lane" + str(lane))
if not os.path.exists(OUT_DIR):
    parent_dir = os.path.dirname(OUT_DIR)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    os.makedirs(OUT_DIR)

# -----------------------------
# 모델 불러오기
# -----------------------------
print("Loading model from: {}".format(MODEL_PATH))
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")

# -----------------------------
# 입력 데이터 로드
# -----------------------------
print("Loading transcript TU data from: {}".format(TU_PATH))
tu_df = pd.read_csv(TU_PATH, sep="\t")

# gene_ENST 혹은 transcript id 컬럼명 탐색
id_col = None
for c in tu_df.columns:
    if "gene_ENST" in c or "transcript" in c or "ENST" in c:
        id_col = c
        break
if id_col is None:
    raise ValueError("Cannot find gene_ENST or transcript id column in TU file!")

tu_df = tu_df.set_index(id_col)

# 모델 입력 feature 순서 파일 (seed별)
feature_path = "/home/hyeongu/DATA5/hyeongu/TCGA-OV/data/TU/RF_input/gHRD_model/ver2_highly/features_train_x_{}.txt".format(seed)
print("Loading feature order from: {}".format(feature_path))
target = pd.read_csv(feature_path, sep="\t", index_col="Unnamed: 0")

# feature 이름 맞추기
feature_names = [c.split(".")[0] for c in target.columns]
feature_order = feature_names
feature_set = set(feature_names)

# tu_df index 정제 (버전 .1, .2 등 제거)
tu_df["temp"] = tu_df.index.str.rpartition(".")[0]
tu_df = tu_df.set_index("temp")

# feature 교집합 추출
tu_set = set(tu_df.index)
shared = list(tu_set & feature_set)

# feature 순서대로 재정렬
tu_df = tu_df.loc[shared]
tu_df = tu_df.reindex(feature_order, fill_value=0)

# transpose → sample × feature
X = tu_df.T
print("Input matrix shape: {}".format(X.shape))

# -----------------------------
# 예측 수행
# -----------------------------
print("Running predictions...")
prob = model.predict_proba(X)[:, 1]
pred = model.predict(X)

# -----------------------------
# 결과 저장
# -----------------------------
out_prob_path = os.path.join(OUT_DIR, "predicted_proba_based_on_" + str(seed) + ".txt")
out_bin_path  = os.path.join(OUT_DIR, "predicted_binary_based_on_" + str(seed) + ".txt")

pd.DataFrame({
    "id": X.index,
    "pred_HRD_prob": prob
}).to_csv(out_prob_path, sep="\t", index=False)

pd.DataFrame({
    "id": X.index,
    "pred_HRD_binary": pred
}).to_csv(out_bin_path, sep="\t", index=False)

print("\n✅ Prediction completed!\nSaved probability scores to:\n  {}\n  {}".format(out_prob_path, out_bin_path))
