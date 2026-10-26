"""Unchanged pure functions extracted from EXP12; no GPU engine or data bundled."""
import math
import numpy as np

def require(condition, message):
    if not condition:
        raise ValueError(message)

def make_cases(ids,split='test'):
    require(len(ids)>0 and len(ids)==len(set(ids)) and all(isinstance(u,str) and u for u in ids),'Invalid study IDs')
    return [dict(index=i,uid=u,split=split) for i,u in enumerate(ids)]

def split_cases(cases):
    require(len(cases)>=2,'Two-GPU execution requires at least two studies')
    return [cases[::2],cases[1::2]]

def global_rank(raw):
    raw=np.asarray(raw)
    require(raw.ndim==2 and raw.shape[1]==12 and len(raw)>0 and np.isfinite(raw).all(),'Invalid raw matrix')
    require(((raw>=0)&(raw<=1)).all(),'Raw probability range differs')
    # Exact original EXP10 argsort tie policy; preserve original test row order.
    order=raw.argsort(0).argsort(0).astype(np.float64)
    return (order/max(1,len(raw)-1)).astype(np.float32)

def merge_shards(cases,shards):
    expected={c['index']:(c['split'],c['uid']) for c in cases}
    require(len(expected)==len(cases) and sorted(expected)==list(range(len(cases))),'Case indices differ')
    raw=np.empty((len(cases),12),np.float32);seen=set();rows=[]
    for part in shards:
        require(len(part['cases'])==len(part['raw'])==len(part['rows']),'Shard lengths differ')
        for case,p,row in zip(part['cases'],part['raw'],part['rows']):
            i=case['index'];require(i in expected and i not in seen,'Duplicate or unexpected study index')
            require(expected[i]==(case['split'],case['uid']),'Shard UID/split mismatch')
            p=np.asarray(p);require(p.shape==(12,) and p.dtype==np.float32 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid shard probability')
            require(row['index']==i and row['uid']==case['uid'],'Timing row identity differs')
            raw[i]=p;seen.add(i);rows.append(row)
    require(seen==set(expected),'Incomplete shard coverage')
    return raw,sorted(rows,key=lambda r:r['index'])

def conservative_projection(setup,rows,count=1300,workers=1,historical_floor_seconds=0.):
    require(rows and count>0 and workers in (1,2),'Invalid projection')
    observed=max(r['total_seconds'] for r in rows);worst=max(observed,float(historical_floor_seconds))
    seconds=setup+1.5*math.ceil(count/workers)*worst+60
    return dict(assumed_studies=count,workers=workers,worst_seconds=worst,observed_current_worst_seconds=observed,
        historical_floor_seconds=float(historical_floor_seconds),setup_seconds=setup,
        safety_factor=1.5,export_allowance_seconds=60,projected_seconds=seconds,
        projected_hours=seconds/3600,within_6_75h=seconds<=24300,
        hidden_completion_established=False,filesystem_cache_guaranteed_cold=False)

def check_parity(reference,candidate,reference_rows,candidate_rows):
    raw_equal=bool(np.array_equal(reference,candidate))
    rank_equal=bool(np.array_equal(global_rank(reference),global_rank(candidate)))
    fields=('stack_sha256','mask_sha256','windows_sha256')
    require(len(reference_rows)==len(candidate_rows),'Parity row counts differ')
    inputs=all(all(a[k]==b[k] for k in fields) for a,b in zip(reference_rows,candidate_rows))
    return dict(raw_bitwise_equal=raw_equal,global_rank_bitwise_equal=rank_equal,input_hashes_equal=inputs,
        max_absolute_probability_delta=float(np.max(np.abs(reference.astype(float)-candidate.astype(float)))),
        passed=raw_equal and rank_equal and inputs)
