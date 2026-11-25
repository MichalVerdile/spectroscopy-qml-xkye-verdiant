import json,os,signal,sys,time
from pathlib import Path
sys.path.insert(0,'/root/spectroscopy-qml/runpod')
from protocol import publish,sha256,write_json,stamp
pid=6113
root=Path('/workspace/shared_splits')
seeds=range(41,46)
while not all((root/f'cnmr_seed_{s}.provenance.json').exists() for s in seeds):
    os.kill(pid,0)
    time.sleep(10)
cmdline=Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'/root/spectroscopy-qml/runpod/prepare.py' in cmdline
copies=[]
os.kill(pid,signal.SIGSTOP)
try:
    time.sleep(0.2)
    assert any(line.startswith('State:') and 'T' in line for line in Path(f'/proc/{pid}/status').read_text().splitlines())
    for seed in seeds:
        source=root/f'cnmr_seed_{seed}.npz'
        proven=json.loads(source.with_suffix('.provenance.json').read_text())
        assert proven['kind']=='deterministic_reconstruction'
        assert sha256(source)==proven['sha256']
        for modality in ('hnmr','ir','msms_pos','msms_neg'):
            target=root/f'{modality}_seed_{seed}.npz'
            if target.exists():continue
            digest=publish(source,target)
            record={k:proven[k] for k in ('kind','historical_identity_verified','accepted_by_user','algorithm')}
            record['reused_deterministic_labels_from']=str(source)
            record['reuse_validation']='The unchanged load_split_artifact must validate every ordered target label and the seed before training.'
            write_json(target.with_suffix('.provenance.json'),record)
            copies.append({'target':str(target),'source':str(source),'sha256':digest})
    write_json(Path('/workspace/experiments/xgboost_raw_native/configurations/split_reuse.json'),{'created':stamp(),'copies':copies,'validation':'All reused files remain subject to the unchanged strict preflight label-digest, count, seed and disjointness checks.'})
    print('REUSED_SPLITS',len(copies),flush=True)
finally:
    os.kill(pid,signal.SIGCONT)
