"""
current_witness_check.py

Legacy standalone check of the encoded current from Section 6.

For local hard-core modes,
    Im Gamma_jk = (1/4)<X_j Y_k - Y_j X_k>.
A real density operator has zero current in the declared computational
basis. A nonzero value is basis-dependent and is not, by itself, an
entanglement or noncommutativity certificate.
"""
import numpy as np, pennylane as qml
n=3
def current_op(rho,j,k):
    P={"X":np.array([[0,1],[1,0]],dtype=complex),"Y":np.array([[0,-1j],[1j,0]],dtype=complex),"I":np.eye(2,dtype=complex)}
    def op(w):
        o=np.array([[1.+0j]])
        for q in range(n): o=np.kron(o,P.get(w.get(q,"I")))
        return o
    return 0.25*(np.trace(rho@op({j:"X",k:"Y"}))-np.trace(rho@op({j:"Y",k:"X"}))).real

def witness():
    rng=np.random.default_rng(0)
    S=np.array([[1,.7,.2],[.7,1,.5],[.2,.5,1.]],float)
    X=rng.normal(size=(300,n))@np.linalg.cholesky(S).T
    lo,hi=X.min(0),X.max(0); sc=lambda D:(D-lo)/(hi-lo)*np.pi
    Jp=np.linalg.inv(S).copy(); np.fill_diagonal(Jp,0); Jp*=0.5
    dev=qml.device("default.qubit",wires=n)
    @qml.qnode(dev)
    def real_enc(x):
        for j in range(n): qml.RY(x[j],wires=j)
        return qml.state()
    @qml.qnode(dev)
    def nc_enc(x):
        for j in range(n): qml.RY(x[j]/2,wires=j)
        for j in range(n):
            for k in range(j+1,n): qml.IsingZZ(2*Jp[j,k],[j,k])
        for j in range(n): qml.RY(x[j]/2,wires=j)
        return qml.state()
    def maxJ(enc):
        r=np.zeros((2**n,2**n),dtype=complex)
        for x in sc(X): p=np.asarray(enc(x)); r+=np.outer(p,p.conj())
        r/=len(X)
        return max(abs(current_op(r,j,k)) for j in range(n) for k in range(j+1,n))
    print("ENCODED CURRENT:")
    print(f"  real Ry control:          max|J| = {maxJ(real_enc):.5f}")
    print(f"  Ry-ZZ sandwich          :  max|J| = {maxJ(nc_enc):.5f}")

if __name__=="__main__":
    witness()
