# ============================================================
# Tier 1 Forecasting Models Benchmark (Open Source Only - 2026)
# Models:
# - Chronos (Amazon)
# - TimesFM (Google)
# - Lag-Llama (ServiceNow)
# - Moirai (Salesforce)
# ============================================================

# Install Dependencies (run once in terminal before executing this script)
# python -m pip install pandas numpy matplotlib datasets torch transformers accelerate gluonts timesfm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Load Example Dataset
# ============================================================

url = "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv"

df = pd.read_csv(url)

df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])
df['unique_id'] = 'series1'

train = df[:-12]
test = df[-12:]

horizon = len(test)


# ============================================================
# 1. Chronos (Amazon) — Open Source
# ============================================================

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

chronos_model = "amazon/chronos-t5-small"

tokenizer = AutoTokenizer.from_pretrained(chronos_model)
model = AutoModelForSeq2SeqLM.from_pretrained(chronos_model)

series = train['y'].values.tolist()
input_text = ",".join(map(str, series))

inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_length=50)

chronos_forecast = tokenizer.decode(outputs[0])

print("Chronos Forecast:")
print(chronos_forecast)
