import sys
from pathlib import Path
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sird_validation import *
root=Path(__file__).resolve().parents[1]
times=np.arange(0,160,3,dtype=float);observations=simulate(times,TRUE)
config=Configuration(adam_steps=8000,lbfgs_steps=600)
model,training,history=train(times,observations,[.8,.1,.15],0,config,output=root/'validation/pilot')
metrics,*_=diagnose(model,np.linspace(0,160,641))
(root/'validation/pilot/metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps({'metrics':metrics,'training':training},indent=2),flush=True)
