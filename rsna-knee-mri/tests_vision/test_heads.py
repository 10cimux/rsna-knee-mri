"""Real PyTorch tests; synthetic inputs only, no fitting on Gold."""
import io
import unittest
import torch
from knee_portfolio.heads import (LABELS,HEAD_CONFIG,MeanPooling,FindingSpecificAttention,make_pair,
                        masked_bce_with_logits,parameter_count,state_sha256,ensemble_probabilities)

def batch(b=2,n=18):
    rng=torch.Generator().manual_seed(741)
    x=torch.randn(b,n,768,generator=rng,dtype=torch.float32)
    mask=torch.ones(b,n,dtype=torch.bool)
    if n>2:mask[:,-2:]=False
    pos=torch.linspace(-1,1,n,dtype=torch.float32)[None,:].repeat(b,1)
    plane=(torch.arange(n,dtype=torch.int64)%3)[None,:].repeat(b,1)
    return x,mask,pos,plane

class HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def pair(self):return {name:model.eval() for name,model in make_pair(42).items()}

    def test_declared_shapes_and_parameter_counts(self):
        pair=self.pair();args=batch()
        self.assertEqual(parameter_count(pair['mean_pooling']),50636)
        self.assertEqual(parameter_count(pair['finding_specific_attention']),55192)
        for model in pair.values():
            y=model(*args)
            self.assertEqual(y.logits.shape,(2,12));self.assertEqual(y.weights.shape,(2,12,18))
            self.assertEqual(y.logits.dtype,torch.float32)
            self.assertTrue(torch.isfinite(y.logits).all())
            torch.testing.assert_close(y.weights.sum(-1),torch.ones(2,12),atol=1e-6,rtol=0)
            self.assertTrue((y.weights[~args[1][:,None,:].expand_as(y.weights)]==0).all())

    def test_label_order_and_configuration(self):
        self.assertEqual(LABELS,('ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
                         'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture'))
        self.assertEqual(HEAD_CONFIG['seeds'],[42,1729,2026])
        self.assertEqual(HEAD_CONFIG['position_basis'],['r','r_squared','sin_pi_r','cos_pi_r'])

    def test_common_initialization_and_cpu_rng_restore(self):
        before=torch.get_rng_state().clone();a=make_pair(42)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for name,value in a['mean_pooling'].state_dict().items():
            self.assertTrue(torch.equal(value,a['finding_specific_attention'].state_dict()[name]))
        b=make_pair(42);c=make_pair(1729)
        for key in a:
            self.assertEqual(state_sha256(a[key]),state_sha256(b[key]))
            self.assertNotEqual(state_sha256(a[key]),state_sha256(c[key]))

    def test_mean_equals_direct_valid_token_average(self):
        model=self.pair()['mean_pooling'];x,mask,pos,plane=batch()
        h=model.tokens(x,mask,pos,plane)
        pooled=h.sum(1)/mask.sum(1,keepdim=True)
        expected=pooled@model.classifier_weight.T+model.classifier_bias
        torch.testing.assert_close(model(x,mask,pos,plane).logits,expected,atol=1e-6,rtol=1e-6)

    def test_attention_uniform_reduces_to_mean(self):
        pair=self.pair();attention=pair['finding_specific_attention']
        with torch.no_grad():attention.attention_scores.weight.zero_();attention.attention_scores.bias.zero_()
        a=pair['mean_pooling'](*batch());b=attention(*batch())
        torch.testing.assert_close(a.logits,b.logits,atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(a.weights,b.weights,atol=1e-7,rtol=0)

    def test_invalid_token_nan_inf_and_ids_cannot_change_output(self):
        args=batch();x,mask,pos,plane=[a.clone() for a in args]
        x[~mask]=float('nan');pos[~mask]=float('inf');plane[~mask]=-999
        for model in self.pair().values():
            a=model(*args);b=model(x,mask,pos,plane)
            self.assertTrue(torch.equal(a.logits,b.logits))
            self.assertTrue(torch.equal(a.weights,b.weights))

    def test_all_missing_rejected_including_one_empty_study(self):
        x,mask,pos,plane=batch();mask[1]=False
        for model in self.pair().values():
            with self.assertRaisesRegex(ValueError,'at least one valid'):model(x,mask,pos,plane)

    def test_single_valid_token_same_for_both_heads(self):
        x,mask,pos,plane=batch();mask[:]=False;mask[:,4]=True
        pair=self.pair();a=pair['mean_pooling'](x,mask,pos,plane);b=pair['finding_specific_attention'](x,mask,pos,plane)
        torch.testing.assert_close(a.logits,b.logits,atol=1e-7,rtol=1e-6)
        self.assertTrue((a.weights[:,:,4]==1).all());self.assertTrue((b.weights[:,:,4]==1).all())

    def test_token_permutation_invariance(self):
        args=batch();order=torch.randperm(18,generator=torch.Generator().manual_seed(33))
        for model in self.pair().values():
            a=model(*args);b=model(*(t[:,order] for t in args))
            torch.testing.assert_close(a.logits,b.logits,atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(a.weights[:,:,order],b.weights,atol=1e-6,rtol=1e-6)

    def test_padding_and_batch_invariance(self):
        args=batch(n=12)
        x,mask,pos,plane=args
        padded=(torch.cat([x,torch.full((2,6,768),float('nan'))],1),
                torch.cat([mask,torch.zeros(2,6,dtype=torch.bool)],1),
                torch.cat([pos,torch.full((2,6),float('nan'))],1),
                torch.cat([plane,torch.full((2,6),-1,dtype=torch.int64)],1))
        for model in self.pair().values():
            a=model(*args);b=model(*padded)
            torch.testing.assert_close(a.logits,b.logits,atol=1e-6,rtol=1e-6)
            one=model(*(t[:1] for t in args))
            torch.testing.assert_close(one.logits,a.logits[:1],atol=1e-6,rtol=1e-6)

    def test_attention_is_finding_specific(self):
        model=self.pair()['finding_specific_attention'];args=batch();before=model(*args)
        with torch.no_grad():model.attention_scores.weight[0]*=10
        after=model(*args)
        self.assertGreater(float((before.weights[:,0]-after.weights[:,0]).detach().abs().max()),1e-5)
        torch.testing.assert_close(before.weights[:,1:],after.weights[:,1:],atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(before.logits[:,1:],after.logits[:,1:],atol=1e-7,rtol=1e-6)

    def test_evaluation_repeat_and_state_are_deterministic(self):
        for model in self.pair().values():
            digest=state_sha256(model);a=model(*batch());b=model(*batch())
            self.assertTrue(torch.equal(a.logits,b.logits));self.assertTrue(torch.equal(a.weights,b.weights))
            self.assertEqual(digest,state_sha256(model))

    def test_masked_features_have_zero_gradient(self):
        for model in self.pair().values():
            x,mask,pos,plane=batch();x[~mask]=float('nan');x.requires_grad_()
            model(x,mask,pos,plane).logits.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue((x.grad[~mask]==0).all())
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad);self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_unknown_labels_indexed_before_bce_and_zero_gradient(self):
        logits=torch.zeros(2,12,dtype=torch.float32,requires_grad=True)
        targets=torch.full((2,12),float('nan'));targets[0,2]=1;targets[1,8]=0
        loss=masked_bce_with_logits(logits,targets);self.assertTrue(torch.isfinite(loss))
        loss.backward();unknown=torch.isnan(targets)
        self.assertTrue((logits.grad[unknown]==0).all())
        known=~unknown;other=targets.clone();other[unknown]=999
        torch.testing.assert_close(loss,masked_bce_with_logits(logits,other,known))

    def test_invalid_targets_and_empty_known_batch_rejected(self):
        for targets in (torch.full((2,12),float('nan')),torch.full((2,12),2.),torch.full((2,12),float('inf'))):
            with self.assertRaises(ValueError):masked_bce_with_logits(torch.zeros(2,12),targets)

    def test_bad_valid_input_shape_dtype_and_values_rejected(self):
        for kind in ('feature_nan','position_nan','position_range','plane_id','mask_dtype','shape','feature_dtype'):
            for model in self.pair().values():
                x,mask,pos,plane=batch()
                if kind=='feature_nan':x[0,0,0]=float('nan')
                if kind=='position_nan':pos[0,0]=float('nan')
                if kind=='position_range':pos[0,0]=1.01
                if kind=='plane_id':plane[0,0]=3
                if kind=='mask_dtype':mask=mask.float()
                if kind=='shape':x=x[:,:,:767]
                if kind=='feature_dtype':x=x.double()
                with self.assertRaises(ValueError,msg=kind):model(x,mask,pos,plane)

    def test_state_dict_roundtrip(self):
        for name,model in self.pair().items():
            buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
            clone=MeanPooling() if name=='mean_pooling' else FindingSpecificAttention()
            clone.load_state_dict(torch.load(buf,weights_only=True),strict=True);clone.eval()
            self.assertEqual(state_sha256(model),state_sha256(clone))
            self.assertTrue(torch.equal(model(*batch()).logits,clone(*batch()).logits))

    def test_ensemble_averages_probabilities_without_mutating_state(self):
        models=[make_pair(s)['mean_pooling'].eval() for s in (42,1729)]
        digests=[state_sha256(m) for m in models];args=batch()
        expected=torch.stack([m(*args).logits.sigmoid() for m in models]).mean(0)
        actual=ensemble_probabilities(models,*args)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        self.assertEqual(digests,[state_sha256(m) for m in models])
        with self.assertRaises(ValueError):ensemble_probabilities([],*args)
        models[0].train()
        with self.assertRaises(ValueError):ensemble_probabilities(models,*args)

if __name__=='__main__':unittest.main()
