"""Pairwise AUC decomposition; unknown targets are excluded, never zero-filled."""
import numpy as np

def pair_decomposition(target, prediction, folds):
    y = np.asarray(target, dtype=float)
    p = np.asarray(prediction, dtype=float)
    f = np.asarray(folds)
    if y.ndim != 1 or y.shape != p.shape or y.shape != f.shape or not len(y):
        raise ValueError("Expected aligned nonempty one-dimensional arrays")
    if not np.isfinite(p).all() or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("Invalid probabilities")
    if any(x is None or str(x).strip() == '' for x in f.tolist()):
        raise ValueError("Missing fold")
    if f.dtype.kind in 'fc' and not np.isfinite(f).all():
        raise ValueError("Non-finite fold")
    known = ~np.isnan(y)
    if not np.isin(y[known], [0, 1]).all():
        raise ValueError("Targets must be binary or NaN")
    y, p, f = y[known], p[known], f[known]
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    wins = (p[pos, None] > p[neg][None, :]).astype(float)
    wins += 0.5 * (p[pos, None] == p[neg][None, :])
    same = f[pos, None] == f[neg][None, :]
    def component(mask):
        n = int(mask.sum())
        score = float(wins[mask].sum())
        return {"pairs": n, "concordance_sum": score, "auc": score / n if n else None}
    rows = []
    for fold in np.unique(f):
        mask = (f[pos, None] == fold) & (f[neg][None, :] == fold)
        row = component(mask)
        row.update(fold=str(fold), positives=int(((f == fold) & (y == 1)).sum()),
                   negatives=int(((f == fold) & (y == 0)).sum()))
        rows.append(row)
    defined = [r["auc"] for r in rows if r["auc"] is not None]
    return {"known": len(y), "unknown": int((~known).sum()),
            "global": component(np.ones_like(same)), "within": component(same),
            "between": component(~same), "per_fold": rows,
            "mean_defined_fold_auc": float(np.mean(defined)) if defined else None}
