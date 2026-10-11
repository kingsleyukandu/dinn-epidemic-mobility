import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from two_region_sird import *
t=np.arange(0.,160.,3.)
model,training,_=train(t,simulate(t,SYNTHETIC_RATES),seed=0,output=ROOT/'validation/step3/pilot')
metrics,_,_,_=diagnose(model,np.linspace(0,160,641))
(ROOT/'validation/step3/pilot/metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps({'training':training,'metrics':metrics},indent=2))
