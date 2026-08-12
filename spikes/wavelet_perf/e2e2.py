import sys, time; sys.path.insert(0,'/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf')
import jax, jax.numpy as jnp, numpy as np
from harness import Problem
from mg import MG
from stencil_mg import StencilMG
from e2e import make_pcg, _time

def run(dim,nl,nc,kind,ct,mass=1.0):
    p=Problem(nl,nc,dim,kind=kind,contrast=ct,mass=mass); N=p.N
    f=jnp.asarray(np.random.default_rng(0).normal(size=N)); b=p.wn_transpose(f)
    res={}
    D=p.D('hybrid'); Ahat=lambda v: p.A_wave(v/D)/D; bh=b/D
    cg=make_pcg(Ahat,None,N); x,k,rn=cg(bh); x.block_until_ready()
    res['hybrid']=(int(k),float(rn),_time(cg,bh),0.0)
    ref=p.wn_apply(x/D)
    for tag,build in [('mg-arith',lambda: MG(p.a,p.side,dim,p.h,p.mass,n_levels=nl,how='arith')),
                      ('mg-rap',  lambda: StencilMG(p.a,p.side,dim,p.h,p.mass,n_levels=nl))]:
        t0=time.perf_counter(); m=build()
        setup=jax.jit(lambda a: None)  # setup timing below via jitted build is not meaningful; time trace+exec
        Minv=lambda v: p.wn_inv(m.apply(p.wn_inv_T(v)))
        cg2=make_pcg(p.A_wave,Minv,N); x2,k2,rn2=cg2(b); x2.block_until_ready()
        t=_time(cg2,b)
        # setup cost = one jitted M^-1 build+apply is fused into the jit; report probe cost separately
        res[tag]=(int(k2),float(rn2),t,float(jnp.linalg.norm(p.wn_apply(x2)-ref)/jnp.linalg.norm(ref)))
        res[tag+'_mv']=_time(jax.jit(Minv),b)/_time(jax.jit(p.A_wave),b)
    return res
