import numpy as np, io
from PIL import Image
import pytesseract

FEATURE_ORDER = ['rent_on_time_ratio','utility_on_time_ratio','avg_monthly_upi_txn','std_upi_txn','mobile_recharge_freq','income_stability_score']

def preprocess_input(d):
    return np.array([float(d.get(f,0.0)) for f in FEATURE_ORDER])

def run_fraud_checks(file_storage):
    try:
        img = Image.open(io.BytesIO(file_storage.read()))
        text = pytesseract.image_to_string(img)
    except Exception as e:
        return {'ok':False,'error':str(e)}
    suspicious = len(text)<20 or not any(c.isdigit() for c in text)
    return {'ok':not suspicious,'snippet':text[:200]}
