"""Direct masked-path NET wall-clock (not estimated): masked-hybrid vs masked-MG,
fully jitted incl. MG build, at the production frozen-solve structure."""
import sys, time
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from harness import Problem
from mg import MG
from e2e import make_pcg
from maddening.nodes.adaptive.wavelets import cdd as CDD
from maddening.nodes.adaptive.wavelets import matrixfree as MF

def masked(fn, m):
    def op(v):
        vm = jnp.where(m, v, 0.0); return jnp.where(m, fn(vm), v)
    return op

def run(dim, nl, nc, kind, cts):
    print(f"\n=== masked NET wall-clock {nc*2**nl}^{dim} {kind} (jitted, MG build counted) ===")
    print(f"{'contrast':>9} | {'hybrid wall':>11} | {'MG wall':>9} | {'net x':>6}")
    for ct in cts:
        p = Problem(nl, nc, dim, kind=kind, contrast=ct, mass=1.0)
        N=p.N; D=p.D("hybrid")
        Ahat=lambda v: p.A_wave(v/D)/D
        f=jnp.asarray(np.random.default_rng(0).normal(size=N)); b=p.wn_transpose(f)/D
        lev=np.asarray(p.levels); coarse=jnp.asarray(lev==lev.min())
        sm=lambda m,r: MF.masked_cg_solve(Ahat,m,r,rtol=1e-8,atol=1e-10)
        mask,_,_=CDD.cdd_select(Ahat,sm,b,coarse,max(8,N//16)); mask=jax.lax.stop_gradient(mask)
        beff=jnp.where(mask,b,0.0)
        def hyb(bb):
            return make_pcg(masked(Ahat,mask),None,N,1e-8,5000)(bb)
        def mg(bb):
            m=MG(p.a,p.side,dim,p.h,p.mass,n_levels=nl,how="arith")
            Minv=masked(lambda v: p.wn_inv(m.apply(p.wn_inv_T(v))), mask)
            return make_pcg(masked(p.A_wave,mask),Minv,N,1e-8,5000)(jnp.where(mask,p.wn_transpose(f),0.0))
        jh,jm=jax.jit(hyb),jax.jit(mg)
        jax.block_until_ready(jh(beff)); jax.block_until_ready(jm(beff))
        def t(fn,arg):
            t0=time.perf_counter()
            for _ in range(3): r=fn(arg)
            jax.block_until_ready(r); return (time.perf_counter()-t0)/3
        th=t(jh,beff); tm=t(jm,beff)
        print(f"{ct:>9.0e} | {th*1e3:>9.1f}ms | {tm*1e3:>7.1f}ms | {th/tm:>5.2f}x")

run(3,4,2,"smooth",[1.0,1e2,1e3])
run(2,5,2,"jump",[1e2,1e3])
