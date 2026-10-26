"""EXP08 fixed MIL heads. No encoder, optimizer, fitting loop or fold selection."""
import hashlib
import json
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F

LABELS=('ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
        'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture')
HEAD_CONFIG={'feature_dim':768,'projection_dim':64,'gating_dim':32,'labels':list(LABELS),
             'planes':['Sagittal','Coronal','Axial'],'max_instances':18,'dropout':.3,
             'position_basis':['r','r_squared','sin_pi_r','cos_pi_r'],
             'layer_norm_eps':1e-5,'classifier_init_std':.02,
             'token_transform_order':'projection_layernorm_plus_plane_position_then_dropout',
             'initialization':'same_common_weights_for_mean_and_attention_given_seed',
             'seeds':[42,1729,2026],'folds':[1,2,3,4,5]}

def require(condition,message):
    if not condition:raise ValueError(message)

class HeadOutput(NamedTuple):
    logits:torch.Tensor
    weights:torch.Tensor

class CommonMILHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(768,64,dtype=torch.float32)
        self.norm=nn.LayerNorm(64,eps=1e-5,dtype=torch.float32)
        self.plane_embedding=nn.Embedding(3,64,dtype=torch.float32)
        self.position_projection=nn.Linear(4,64,dtype=torch.float32)
        self.dropout=nn.Dropout(.3)
        self.classifier_weight=nn.Parameter(torch.empty(12,64,dtype=torch.float32))
        self.classifier_bias=nn.Parameter(torch.zeros(12,dtype=torch.float32))
        nn.init.normal_(self.classifier_weight,mean=0.,std=.02)

    def tokens(self,features,mask,position,plane):
        require(features.ndim==3 and features.shape[0]>0 and 1<=features.shape[1]<=18 and features.shape[2]==768,
                'features must have shape [B,1..18,768]')
        shape=features.shape[:2]
        require(mask.shape==shape and position.shape==shape and plane.shape==shape,'Mask/position/plane shapes differ')
        require(features.dtype==torch.float32 and position.dtype==torch.float32,'FP32 features and positions required')
        require(mask.dtype==torch.bool and plane.dtype==torch.int64,'Boolean mask and int64 plane required')
        require(all(t.device==features.device for t in (mask,position,plane)), 'Input devices differ')
        require(features.device==self.classifier_weight.device,'Input/model devices differ')
        require(self.classifier_weight.dtype==torch.float32,'FP32 model required')
        require(bool(mask.any(dim=1).all()),'Every study must have at least one valid token')
        require(bool(torch.isfinite(features[mask]).all()),'Non-finite valid features')
        require(bool(torch.isfinite(position[mask]).all()),'Non-finite valid positions')
        require(bool((position[mask].abs()<=1).all()),'Valid positions must be in [-1,1]')
        require(bool(((plane[mask]>=0)&(plane[mask]<3)).all()),'Valid plane IDs must be 0,1,2')
        # Sanitize BEFORE linear/trigonometric operations so unknown padded values
        # cannot poison logits or gradients. These slots still have zero pooling weight.
        x=torch.where(mask[...,None],features,torch.zeros_like(features))
        r=torch.where(mask,position,torch.zeros_like(position))
        p=torch.where(mask,plane,torch.zeros_like(plane))
        basis=torch.stack((r,r*r,torch.sin(torch.pi*r),torch.cos(torch.pi*r)),dim=-1)
        h=self.norm(self.projection(x))+self.plane_embedding(p)+self.position_projection(basis)
        h=self.dropout(h)
        return torch.where(mask[...,None],h,torch.zeros_like(h))

    def classify(self,h,weights):
        require(bool(torch.isfinite(weights).all()),'Non-finite pooling weights')
        pooled=torch.einsum('bln,bnd->bld',weights,h)
        logits=torch.einsum('bld,ld->bl',pooled,self.classifier_weight)+self.classifier_bias
        require(bool(torch.isfinite(logits).all()),'Non-finite logits')
        return HeadOutput(logits,weights)

class MeanPooling(CommonMILHead):
    """Uniform weights over valid tokens; finding-specific linear classifiers."""
    def forward(self,features,mask,position,plane):
        h=self.tokens(features,mask,position,plane)
        weights=mask.to(h.dtype)/mask.sum(dim=1,keepdim=True)
        weights=weights[:,None,:].expand(-1,12,-1)
        return self.classify(h,weights)

class FindingSpecificAttention(CommonMILHead):
    """12 independent masked softmax distributions, shared gated token transform."""
    def __init__(self):
        super().__init__()
        self.gate_tanh=nn.Linear(64,32,dtype=torch.float32)
        self.gate_sigmoid=nn.Linear(64,32,dtype=torch.float32)
        self.attention_scores=nn.Linear(32,12,dtype=torch.float32)

    def forward(self,features,mask,position,plane):
        h=self.tokens(features,mask,position,plane)
        gated=torch.tanh(self.gate_tanh(h))*torch.sigmoid(self.gate_sigmoid(h))
        scores=self.attention_scores(gated).transpose(1,2)
        scores=scores.masked_fill(~mask[:,None,:],-torch.inf)
        weights=torch.softmax(scores,dim=-1)
        return self.classify(h,weights)

def make_pair(seed):
    """Match common initial states exactly; do not disturb caller CPU RNG state."""
    require(seed in HEAD_CONFIG['seeds'],'Use only the predefined seeds')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed); mean=MeanPooling()
        torch.manual_seed(seed); attention=FindingSpecificAttention()
    for name,value in mean.state_dict().items():
        require(torch.equal(value,attention.state_dict()[name]),'Common initialization differs')
    return {'mean_pooling':mean,'finding_specific_attention':attention}

def masked_bce_with_logits(logits,targets,known=None):
    """Index known binary targets BEFORE BCE; NaN is unknown, never a negative."""
    require(logits.ndim==2 and logits.shape[1]==12 and targets.shape==logits.shape,'Expected [B,12] logits/targets')
    require(logits.dtype==targets.dtype==torch.float32,'FP32 logits and targets required')
    require(logits.device==targets.device,'Loss tensor devices differ')
    require(bool(torch.isfinite(logits).all()),'Non-finite logits')
    if known is None:known=~torch.isnan(targets)
    require(known.shape==targets.shape and known.dtype==torch.bool and known.device==logits.device,'Invalid target mask')
    require(bool(known.any()),'Batch has no known targets')
    selected=targets[known]
    require(bool(torch.isfinite(selected).all()) and bool(((selected==0)|(selected==1)).all()),'Known targets must be finite binary values')
    return F.binary_cross_entropy_with_logits(logits[known],selected,reduction='mean')

def parameter_count(model):return sum(p.numel() for p in model.parameters())

def state_sha256(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        array=tensor.detach().cpu().contiguous().numpy().astype('<f4',copy=False)
        h.update(json.dumps({'name':name,'shape':list(array.shape),'dtype':'float32'},sort_keys=True).encode())
        h.update(array.tobytes(order='C'))
    return h.hexdigest()

def ensemble_probabilities(models,features,mask,position,plane):
    require(len(models)>0,'Empty head ensemble')
    require(all(not model.training for model in models),'Set every model to eval before inference')
    with torch.inference_mode():
        probs=torch.stack([torch.sigmoid(m(features,mask,position,plane).logits) for m in models])
    return probs.mean(dim=0)
