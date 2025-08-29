import numpy as np
import json

def np_encoder(obj):
    if isinstance(obj, (np.integer, np.int_, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)  # safe fallback for other odd types

def jprint(label, obj):
    print(f"\n{label} =")
    print(json.dumps(obj, indent=2, default=np_encoder, ensure_ascii=False))

