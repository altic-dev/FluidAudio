import json
import inspect as _inspect
import torch
from transformers import AutoModelForCTC, AutoProcessor
mid = "ibm-granite/granite-speech-5.0-470m-turboctc"
proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
model = AutoModelForCTC.from_pretrained(mid, dtype=torch.float32, trust_remote_code=True).eval()
print("MODEL CLASS:", type(model).__name__)
print("FORWARD SIG:", _inspect.signature(model.forward))
print("TOP MODULES:", [n for n,_ in model.named_children()])
print("PARAMS(M):", sum(p.numel() for p in model.parameters())/1e6)
import numpy as np
audio = np.zeros(16000*5, dtype=np.float32)
inputs = proc([audio], sampling_rate=16000)
for k,v in inputs.items():
    try: print("INPUT", k, tuple(v.shape), v.dtype)
    except Exception: print("INPUT", k, type(v))
with torch.no_grad():
    out = model(**{k: (v if torch.is_tensor(v) else torch.tensor(v)) for k,v in inputs.items()})
print("OUT KEYS:", list(out.keys()) if hasattr(out,'keys') else type(out))
print("LOGITS:", tuple(out.logits.shape))
