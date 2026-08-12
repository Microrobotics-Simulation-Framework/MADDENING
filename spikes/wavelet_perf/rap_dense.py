"""Test of principle: op-dep P + TRUE Galerkin RAP coarse operators (dense).
Answers: is contrast-INDEPENDENCE reachable at all for this operator?"""
import numpy as np, jax, jax.numpy as jnp
from harness import dense_of
from opdep import make_opdep_prolong

class MGRAP:
    def __init__(self, a, side, dim, h, mass, n_levels, A0, nu=2, omega=0.8, coarse_direct=True):
        self.dim,self.nu,self.omega,self.coarse_direct=dim,nu,omega,coarse_direct
        self.As=[np.asarray(A0)]; self.Ps=[]; self.shapes=[(side,)*dim]
        ac=a.reshape((side,)*dim); s=side; hh=h
        for lev in range(n_levels):
            Pf=make_opdep_prolong(ac,dim,hh,mass)
            nc=s//2
            Pm=np.asarray(dense_of(lambda e: Pf(e.reshape((nc,)*dim)).reshape(-1), nc**dim))
            self.Ps.append(Pm)
            self.As.append(Pm.T@self.As[-1]@Pm)          # Galerkin RAP
            # coarse coefficient only needed to build the NEXT P: use full-weighting arith
            from mg import coarsen_coeff
            ac=coarsen_coeff(ac,dim,'arith'); s=nc; hh*=2.0
            self.shapes.append((s,)*dim)
        self.nlev=len(self.As)
        self.diags=[np.diag(A) for A in self.As]
    def _sm(self,l,u,f,n):
        for _ in range(n): u=u+self.omega*(f-self.As[l]@u)/self.diags[l]
        return u
    def _v(self,l,u,f):
        if l==self.nlev-1:
            return np.linalg.solve(self.As[l],f) if self.coarse_direct else self._sm(l,u,f,50)
        u=self._sm(l,u,f,self.nu)
        r=f-self.As[l]@u
        ec=self._v(l+1,np.zeros(self.Ps[l].shape[1]),self.Ps[l].T@r)
        u=u+self.Ps[l]@ec
        return self._sm(l,u,f,self.nu)
    def apply_np(self,r):
        return self._v(0,np.zeros_like(r),np.asarray(r))
