import unittest
import numpy as np
from knee_portfolio.evaluation import pair_decomposition
from knee_portfolio.inference_contracts import (
    make_cases, split_cases, merge_shards, global_rank, conservative_projection)

class EvaluationTests(unittest.TestCase):
    def test_fold_perfect_global_imperfect(self):
        r = pair_decomposition([0,1,0,1], [.1,.2,.8,.9], [1,1,2,2])
        self.assertEqual(r['mean_defined_fold_auc'], 1)
        self.assertEqual(r['global']['auc'], .75)
        self.assertEqual(r['between']['auc'], .5)
        self.assertEqual(r['global']['concordance_sum'],
                         r['within']['concordance_sum'] + r['between']['concordance_sum'])

    def test_ties_count_half(self):
        self.assertEqual(pair_decomposition([0,1],[.5,.5],[1,1])['global']['auc'], .5)

    def test_unknown_excluded(self):
        r = pair_decomposition([0,1,np.nan],[.1,.9,.8],[1,1,2])
        self.assertEqual(r['known'], 2)
        self.assertEqual(r['unknown'], 1)
        self.assertEqual(r['global']['auc'], 1)

    def test_single_class_undefined(self):
        self.assertIsNone(pair_decomposition([1,1],[.1,.9],[1,2])['global']['auc'])

    def test_invalid_inputs(self):
        for y,p,f in [([0,2],[.1,.9],[1,1]), ([0,1],[.1,np.nan],[1,1]),
                      ([0,1],[.1,.9],[1]), ([0,1],[.1,.9],[1,np.nan])]:
            with self.assertRaises(ValueError): pair_decomposition(y,p,f)

class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.cases = make_cases(['synthetic-a','synthetic-b','synthetic-c'])
        self.raw = np.tile(np.array([.7,.1,.5],np.float32)[:,None],(1,12))
        self.parts = [{'cases': cs,'raw': [self.raw[c['index']] for c in cs],
                       'rows': [{'index':c['index'],'uid':c['uid']} for c in cs]}
                      for cs in split_cases(self.cases)]

    def test_reverse_completion_preserves_original_order(self):
        raw, rows = merge_shards(self.cases, self.parts[::-1])
        np.testing.assert_array_equal(raw,self.raw)
        self.assertEqual([r['index'] for r in rows],[0,1,2])

    def test_missing_duplicate_and_identity_rejected(self):
        with self.assertRaises(ValueError): merge_shards(self.cases,self.parts[:1])
        with self.assertRaises(ValueError): merge_shards(self.cases,self.parts+self.parts[:1])
        self.parts[0]['rows'][0]['uid']='wrong'
        with self.assertRaises(ValueError): merge_shards(self.cases,self.parts)

    def test_ranking_must_be_global(self):
        whole = global_rank(self.raw)
        partial = global_rank(self.raw[::2])
        self.assertFalse(np.array_equal(whole[::2],partial))

    def test_original_tie_policy_preserved(self):
        x=np.full((3,12),.5,np.float32)
        np.testing.assert_array_equal(global_rank(x),
            (x.argsort(0).argsort(0)/2).astype(np.float32))

    def test_runtime_uses_historical_worst(self):
        p=conservative_projection(10,[{'total_seconds':1}],count=1300,workers=2,
                                  historical_floor_seconds=15)
        self.assertEqual(p['projected_seconds'],10+1.5*650*15+60)
        self.assertFalse(p['hidden_completion_established'])

    def test_nonfinite_and_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError): global_rank(np.full((3,12),np.nan))
        with self.assertRaises(ValueError): make_cases(['same','same'])

if __name__ == '__main__': unittest.main()
