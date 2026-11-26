"""Independent CPU numerical checks of the cloned native extra-seed kernels.

Torch calculations run in a disposable subprocess so the planning test process
never imports torch. Set MEDVLM_CPU_EQUIVALENCE_PYTHON to a reviewed interpreter
with CPU-capable torch; absent torch skips only these numerical cases. No CUDA,
model cache, campaign state, dataset, adapter, or supervisor operation is used.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mediflow_medvlm import campaign_v6_completion as completion

REPO = Path(__file__).resolve().parents[1]


def runtime_config():
    return {"data":{"native_function_contract":completion.native_function_contract()},
            "source_files":{"src/mediflow_medvlm/campaign_main_train.py":completion.file_hash(completion.train.__file__)}}


def test_every_cloned_kernel_keeps_native_code_defaults_and_isolated_globals():
    native = completion.train; before = dict(vars(native)); contract = completion.native_function_contract()
    first = completion.training_runtime(runtime_config()); second = completion.training_runtime(runtime_config())
    namespaces = []
    for runtime in (first,second):
        namespace = runtime._train.__globals__; namespaces.append(namespace)
        assert namespace is not vars(native)
        for name in completion.CLONED:
            original,cloned = getattr(native,name),getattr(runtime,name)
            assert cloned is not original and cloned.__code__ is original.__code__
            assert cloned.__defaults__ == original.__defaults__ and cloned.__kwdefaults__ == original.__kwdefaults__
            assert cloned.__closure__ == original.__closure__ and cloned.__globals__ is namespace
            assert namespace[name] is cloned
        assert namespace["_validate"] is completion._training_validate
        assert namespace["_process"] is completion._training_process
        assert namespace["SCOPE"] == completion.TRAIN_SCOPE
    assert namespaces[0] is not namespaces[1]
    namespaces[0]["SCOPE"] = "private numerical fixture override"
    assert namespaces[1]["SCOPE"] == completion.TRAIN_SCOPE and native.SCOPE != namespaces[0]["SCOPE"]
    assert set(vars(native)) == set(before) and all(vars(native)[k] is v for k,v in before.items())
    assert completion.native_function_contract() == contract


@pytest.mark.parametrize("field",["contract","module"])
def test_cloning_rejects_changed_native_source_contract(field):
    config = runtime_config()
    if field == "contract": config["data"]["native_function_contract"]["_updates"] = "0"*64
    else: config["source_files"]["src/mediflow_medvlm/campaign_main_train.py"] = "0"*64
    with pytest.raises(ValueError,match="NATIVE_FUNCTION_CODE_CHANGED|NATIVE_MODULE_SOURCE_CHANGED"):
        completion.training_runtime(config)


@pytest.mark.parametrize("size,steps,tail",[(7997,500,13),(1979,124,11)])
def test_frozen_extra_seed_orders_keep_exact_once_only_exposure(size,steps,tail):
    ids = ["synthetic-"+str(i) for i in range(size)]
    orders = completion.train.common._orders(ids)
    assert set(orders) == {"20261023","20261114","20261115"}
    for seed in (20261114,20261115):
        order = orders[str(seed)]
        assert sorted(order["indices"]) == list(range(size))
        assert order["optimizer_steps"] == steps and order["update_example_counts"] == [16]*(steps-1)+[tail]
        assert order["micro_loss_divisor_per_update"] == order["update_example_counts"]
        assert order["requires_family_selection_before_execution"] is True
        assert order["indices"] != orders["20261023"]["indices"]
    assert orders["20261114"]["indices"] != orders["20261115"]["indices"]


CPU_RUNNER = r'''
import builtins,copy,hashlib,json,os,random,sys,types
from pathlib import Path
import torch
from mediflow_medvlm import campaign_v6_completion as c

torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
native=c.train
before=dict(vars(native)); source_before=c.native_function_contract()
config={'data':{'native_function_contract':c.native_function_contract()},'source_files':{
 'src/mediflow_medvlm/campaign_main_train.py':c.file_hash(native.__file__)}}

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.randn(4,4,dtype=torch.float64)*.12)
        self.base=torch.nn.Parameter(torch.arange(4,dtype=torch.float64),requires_grad=False)
        self.config=types.SimpleNamespace(model_type='synthetic_cpu_causal')
    @property
    def device(self): return self.weight.device
    def forward(self,input_ids,attention_mask=None):
        return types.SimpleNamespace(logits=torch.nn.functional.one_hot(input_ids,4).to(torch.float64)@self.weight*9.)
    def save_pretrained(self,path,**kw):
        (Path(path)/'adapter_model.safetensors').write_bytes(b'SYNTHETIC_CPU_FIXTURE_NOT_A_MODEL')

def records(size):
    return [{'sample_id':str(i),'native':{'supervised_tokens_after_shift':i%5+1,'sequence_tokens':i%2+1+i%5+1+1}} for i in range(size)]

def batch(r):
    i=int(r['sample_id']); n=r['native']['supervised_tokens_after_shift']; prompt=i%2+1; length=prompt+n+1
    labels=[-100]*prompt+[(i+j)%4 for j in range(n)]+[-100]
    return {'input_ids':torch.tensor([[(i+2*j)%4 for j in range(length)]]),
            'attention_mask':torch.tensor([[1]*(length-1)+[0]]),'labels':torch.tensor([labels])}

def target(m,r): return native._target_loss(m,batch(r))

def oracle(m,r):
    b=batch(r); mask=b['labels'][:,1:]!=-100; labels=b['labels'][:,1:][mask]
    logits=m(input_ids=b['input_ids'],attention_mask=b['attention_mask']).logits[:,:-1,:][mask].float()
    token_losses=torch.logsumexp(logits,dim=-1)-logits.gather(1,labels[:,None]).squeeze(1)
    return token_losses.mean(),token_losses

def adam(m,steps):
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.03,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:max(0.,1.-n/steps))
    return opt,sch

def state_equal(a,b):
    assert a.keys()==b.keys()
    for key in a:
        if isinstance(a[key],dict): state_equal(a[key],b[key])
        elif torch.is_tensor(a[key]): torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        elif isinstance(a[key],list):
            assert len(a[key])==len(b[key])
            for x,y in zip(a[key],b[key]):
                if isinstance(x,dict): state_equal(x,y)
                else: assert x==y
        else: assert a[key]==b[key]

def partial(size):
    torch.random.default_generator.manual_seed(71); a=Toy(); b=copy.deepcopy(a); ref=copy.deepcopy(a)
    rs=records(size); steps=2; ao,asch=adam(a,steps); bo,bsch=adam(b,steps); ro,rsch=adam(ref,steps)
    # A token-pooled objective must differ from the required equal-example mean.
    losses=[oracle(ref,r) for r in rs[:16]]
    correct=torch.stack([x[0] for x in losses]).mean()
    pooled=torch.cat([x[1] for x in losses]).mean()
    gc=torch.autograd.grad(correct,ref.weight,retain_graph=True)[0]
    gp=torch.autograd.grad(pooled,ref.weight)[0]
    assert float((gc-gp).abs().max())>1e-3
    def run(fn,m,opt,sch):
        gradients=[]
        def example(r):
            loss,n=target(m,r)
            return loss,n,r['native']['sequence_tokens']
        out=fn(m,opt,sch,rs,{},example,lambda:gradients.append(m.weight.grad.detach().clone()) or {})
        return out,gradients
    runtime=c.training_runtime(config)
    left,ga=run(native._updates,a,ao,asch); right,gb=run(runtime._updates,b,bo,bsch)
    assert left==right
    torch.testing.assert_close(a.weight,b.weight,rtol=0,atol=0); state_equal(ao.state_dict(),bo.state_dict())
    state_equal(asch.state_dict(),bsch.state_dict())
    for step,start in enumerate(range(0,size,16)):
        group=rs[start:start+16]; ro.zero_grad(set_to_none=True)
        torch.stack([oracle(ref,r)[0] for r in group]).mean().backward()
        torch.testing.assert_close(ga[step],ref.weight.grad,rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(ga[step],gb[step],rtol=0,atol=0)
        assert float(ga[step].norm())>1. # Exercise native clipping as well as pre-clip math.
        if len(group)<16:
            wrong=ref.weight.grad*len(group)/16.
            assert float((ga[step]-wrong).abs().max())>.01
        torch.nn.utils.clip_grad_norm_([ref.weight],1.)
        ro.step(); rsch.step()
    torch.testing.assert_close(a.weight,ref.weight,rtol=2e-6,atol=2e-6)
    assert left['micro_loss_divisor_per_update']==[16,size-16]
    assert left['example_supervised_tokens']==[r['native']['supervised_tokens_after_shift'] for r in rs]
    assert left['learning_rates']==[.03,.015] and left['final_learning_rate']==0.
    assert a.weight.grad is None and all(not p.requires_grad for p in [a.base])

def full_pass():
    for size,steps,tail in [(7997,500,13),(1979,124,11)]:
        torch.random.default_generator.manual_seed(88)
        a=torch.nn.Linear(1,1,bias=False,dtype=torch.float64); b=copy.deepcopy(a)
        order=native.common._orders([str(i) for i in range(size)])['20261114']['indices']
        rs=[{'sample_id':str(i)} for i in order]
        def run(fn,m):
            opt,sch=adam(m,steps)
            out=fn(m,opt,sch,rs,{},lambda r:((m.weight.reshape(())-(int(r['sample_id'])%7)/10.).square(),int(r['sample_id'])%5+1,9),lambda:{})
            return out,opt,sch
        l,lo,ls=run(native._updates,a); r,ro,rsch=run(c.training_runtime(config)._updates,b)
        assert l==r and l['optimizer_steps']==steps and l['samples_processed']==size
        assert l['sample_ids']==[x['sample_id'] for x in rs] and len(set(l['sample_ids']))==size
        assert l['micro_loss_divisor_per_update']==[16]*(steps-1)+[tail]
        assert l['final_learning_rate']==0.
        torch.testing.assert_close(a.weight,b.weight,rtol=0,atol=0); state_equal(lo.state_dict(),ro.state_dict())

def fresh_train(tmp):
    runtime=c.training_runtime(config); ns=runtime._train.__globals__
    cuda_calls=[]; instances=[]; optimizers=[]; schedulers=[]; seeds=[]
    fake_cuda=types.SimpleNamespace(is_available=lambda:True,manual_seed_all=lambda s:seeds.append(s),
        reset_peak_memory_stats=lambda:cuda_calls.append('reset'),synchronize=lambda:cuda_calls.append('synchronize'),
        get_device_properties=lambda i:types.SimpleNamespace(name='SYNTHETIC_CPU_ONLY',total_memory=0),
        get_device_capability=lambda i=None:(0,0),max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)
    def opt(params,**kwargs):
        value=torch.optim.AdamW(params,**kwargs); assert not value.state
        optimizers.append(value); return value
    def scheduler(*args,**kwargs):
        value=torch.optim.lr_scheduler.LambdaLR(*args,**kwargs); schedulers.append(value); return value
    proxy=types.SimpleNamespace(**{k:getattr(torch,k) for k in ('isfinite','nn','version','backends')},cuda=fake_cuda,
        manual_seed=lambda seed:torch.random.default_generator.manual_seed(seed),
        optim=types.SimpleNamespace(AdamW=opt,lr_scheduler=types.SimpleNamespace(LambdaLR=scheduler)))
    real_import=builtins.__import__
    ns['__builtins__']={**vars(builtins),'__import__':lambda name,*a,**kw:proxy if name=='torch' else real_import(name,*a,**kw)}
    # FunctionType caches its builtins at construction. Rewrap the identical
    # audited code only to install this CPU import seam; never patch the native
    # module, Python builtins, torch module, or actual CUDA methods.
    cpu_train=types.FunctionType(runtime._train.__code__,ns,'_train',runtime._train.__defaults__,runtime._train.__closure__)
    assert cpu_train.__code__ is runtime._train.__code__ is native._train.__code__
    ns['_inherited_supervisor_fds']=lambda config:None
    ns['_verify_native_contract']=lambda config,contract:None
    ns['_metrics_proof']=lambda config,metrics,rs:None
    ns['_progress']=lambda *a,**kw:None
    ns['runtime_versions']=lambda:{'synthetic_cpu':True}
    ns['_encode']=lambda row,r,bundle,config:(batch(r),{'sequence_tokens':r['native']['sequence_tokens']})
    def digest(m): return hashlib.sha256(m.weight.detach().numpy().tobytes()).hexdigest()
    def load(model,cell_id,**kw):
        assert model=={'original_pinned_base':True}
        m=Toy(); instances.append((m,m.weight.detach().clone(),random.random()))
        return types.SimpleNamespace(model=m,contract={'architecture':{}},processor=types.SimpleNamespace(save_pretrained=lambda p:(Path(p)/'fixture.json').write_text('{}')))
    def frozen(m):
        assert not m.base.requires_grad
        torch.testing.assert_close(m.base,torch.arange(4,dtype=torch.float64),rtol=0,atol=0)
    fake=types.ModuleType('mediflow_medvlm.training.v6_gpu')
    fake.load_v6_gpu_bundle=load; fake.adapter_digest=digest
    fake.projection_gradient_report=lambda m,a:{'finite':bool(torch.isfinite(m.weight.grad).all())}
    fake.projection_adapter_digests=lambda m,a:{'synthetic':{'A':digest(m),'B':digest(m)}}
    fake.assert_frozen_base=frozen
    sys.modules[fake.__name__]=fake
    outputs=[]
    for index,seed in enumerate([20261114,20261115,20261114]):
        # Deliberately dirty both RNGs before each fresh-start invocation.
        random.seed(91+index); random.random(); torch.random.default_generator.manual_seed(900+index); torch.rand(11)
        rs=records(29); path=Path(tmp)/str(index); path.mkdir()
        cfg={'recipe':{'seed':seed,'learning_rate':.03,'betas':[.9,.999],'eps':1e-8,'optimizer_steps':2},
             'model':{'original_pinned_base':True},'cell_id':'T2','fixed_template_date':'2026-11-14'}
        out=cpu_train(cfg,{r['sample_id']:r for r in rs},rs,path)
        outputs.append((out,instances[-1][0].weight.detach().clone(),optimizers[-1].state_dict(),schedulers[-1].state_dict()))
        assert out['retry_start']=='original_base' and out['optimizer_rng_resume_supported'] is False
    assert seeds==[20261114,20261115,20261114]
    assert len({id(x[0]) for x in instances})==3 and len({id(x) for x in optimizers})==3 and len({id(x) for x in schedulers})==3
    torch.testing.assert_close(instances[0][1],instances[2][1],rtol=0,atol=0)
    assert instances[0][2]==instances[2][2] and instances[0][2]!=instances[1][2]
    assert not torch.equal(instances[0][1],instances[1][1])
    torch.testing.assert_close(outputs[0][1],outputs[2][1],rtol=0,atol=0)
    assert not torch.equal(outputs[0][1],outputs[1][1])
    state_equal(outputs[0][2],outputs[2][2]); state_equal(outputs[0][3],outputs[2][3])
    assert outputs[0][0]['example_losses']==outputs[2][0]['example_losses']
    assert cuda_calls==['reset','synchronize']*3

case=sys.argv[1]
if case.startswith('partial:'): partial(int(case.split(':')[1]))
elif case=='full_pass': full_pass()
elif case=='fresh_train': fresh_train(sys.argv[2])
else: raise AssertionError(case)
assert set(vars(native))==set(before) and all(vars(native)[k] is v for k,v in before.items())
assert c.native_function_contract()==source_before and not torch.cuda.is_initialized()
print(json.dumps({'case':case,'status':'PASS','cuda_initialized':False,'native_globals_changed':False}))
'''


def run_cpu_case(case,tmp_path):
    python = os.environ.get("MEDVLM_CPU_EQUIVALENCE_PYTHON",sys.executable)
    env = {**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONPATH":str(REPO/"src"),"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}
    probe = subprocess.run([python,"-c","import importlib.util; raise SystemExit(0 if importlib.util.find_spec('torch') else 17)"],env=env,capture_output=True,text=True)
    if probe.returncode == 17: pytest.skip("Set MEDVLM_CPU_EQUIVALENCE_PYTHON to a reviewed CPU-capable torch interpreter")
    assert probe.returncode == 0,probe.stderr
    result = subprocess.run([python,"-c",CPU_RUNNER,case,str(tmp_path)],cwd=REPO,env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode == 0,result.stdout+result.stderr
    assert '"cuda_initialized": false' in result.stdout and '"native_globals_changed": false' in result.stdout


@pytest.mark.parametrize("tail",[13,11])
def test_cpu_full_target_causal_ce_mean_examples_gradient_clip_adamw_and_lr_match_native(tail,tmp_path):
    run_cpu_case("partial:"+str(16+tail),tmp_path)


def test_cpu_full_7997_1979_passes_match_native_500_124_updates(tmp_path):
    run_cpu_case("full_pass",tmp_path)


def test_cpu_train_extra_seeds_fresh_rng_model_optimizer_and_scheduler(tmp_path):
    run_cpu_case("fresh_train",tmp_path)
