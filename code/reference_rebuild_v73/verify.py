"""Frozen-asset checks and official fresh-process CPU evaluation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
os.environ['OPENBLAS_CORETYPE']='Haswell'
os.environ['NPY_DISABLE_CPU_FEATURES']='X86_V4,AVX512_ICL,AVX512_SPR'

import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--split',choices=('validation','test'),required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    code=Path(__file__).resolve().parents[1];freeze=json.loads((run/'freeze.json').read_text());plan=json.loads((run/'plan.json').read_text())
    for name,digest in freeze['source_sha256'].items():
        if sha(code/name)!=digest:raise RuntimeError('Frozen source mismatch: '+name)
    for name,digest in freeze['asset_sha256'].items():
        if sha(run/'checkpoint'/name)!=digest:raise RuntimeError('Frozen asset mismatch: '+name)
    for name,digest in plan['official_sha256'].items():
        if sha(code/name)!=digest:raise RuntimeError('Official tool changed: '+name)
    if sha(code/'data/manifest.json')!=plan['manifest_sha256'] or sha(run/'experiments.json')!=freeze['experiments_sha256']:raise RuntimeError('Selection/data manifest changed')
    prediction=output/(args.split+'-predictions.json');score=output/(args.split+'-score.json')
    commands=[[sys.executable,str(code/'predict.py'),'--method',str(code/'submission_v73.py'),'--checkpoint',str(run/'checkpoint'),'--split',args.split,'--output',str(prediction)],
              [sys.executable,str(code/'score_predictions.py'),'--split',args.split,'--predictions',str(prediction),'--output',str(score)]]
    elapsed=[]
    for command in commands:
        start=time.perf_counter();result=subprocess.run(command,cwd=code,capture_output=True,text=True);elapsed.append(time.perf_counter()-start)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')
    metadata=dict(at_utc=datetime.now(timezone.utc).isoformat(),official_score=json.loads(score.read_text()),
                  prediction_wall_seconds=elapsed[0],scoring_wall_seconds=elapsed[1],total_wall_seconds=sum(elapsed),
                  cpu=cpu,threads=4,device='cpu',asset_bytes=freeze['inference_asset_bytes'],reference_xeon_verified=False,
                  numpy=importlib.metadata.version('numpy'),python=sys.version,freeze_sha256=sha(run/'freeze.json'),
                  numeric_backend={k:os.environ[k] for k in ('OPENBLAS_CORETYPE','NPY_DISABLE_CPU_FEATURES')},commands=commands)
    (output/(args.split+'-benchmark.json')).write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
