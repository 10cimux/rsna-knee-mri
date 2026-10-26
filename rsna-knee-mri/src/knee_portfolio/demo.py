"""Run only synthetic examples; writes no submission or patient data."""
import json
import numpy as np
from .evaluation import pair_decomposition
from .inference_contracts import make_cases, split_cases, merge_shards, global_rank

def main():
    audit = pair_decomposition([0, 1, 0, 1], [.1, .2, .8, .9], [1, 1, 2, 2])
    cases = make_cases([f"synthetic-{i}" for i in range(4)])
    raw = np.tile(np.array([.1, .2, .8, .9], dtype=np.float32)[:, None], (1, 12))
    shards = [{"cases": part, "raw": [raw[c["index"]] for c in part],
               "rows": [{"index": c["index"], "uid": c["uid"]} for c in part]}
              for part in split_cases(cases)]
    merged, _ = merge_shards(cases, list(reversed(shards)))
    assert np.array_equal(raw, merged)
    assert np.array_equal(global_rank(raw), global_rank(merged))
    print(json.dumps({"data": "synthetic only", "auc_decomposition": audit,
                      "shard_merge_and_global_rank_equal": True}, indent=2))

if __name__ == "__main__":
    main()
