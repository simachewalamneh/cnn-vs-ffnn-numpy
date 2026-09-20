import numpy as np
import cnn_vs_ffnn as M
rng=np.random.default_rng(1)
def check(model,name):
    # float64 for the check
    for l in model.layers:
        for p,g in l.params(): pass
    x=rng.random((4,1,28,28)).astype(np.float32); y=np.array([1,3,5,7])
    for l in model.layers:
        for attr in ("W","b","dW","db"):
            if hasattr(l,attr): setattr(l,attr,getattr(l,attr).astype(np.float64))
    x=x.astype(np.float64)
    loss,g=M.cross_entropy(model.forward(x),y); model.backward(g)
    worst=0
    for p,gr in model.params():
        for _ in range(6):
            i=tuple(rng.integers(0,s) for s in p.shape); old=p[i]; e=1e-5
            p[i]=old+e; lp,_=M.cross_entropy(model.forward(x),y)
            p[i]=old-e; lm,_=M.cross_entropy(model.forward(x),y); p[i]=old
            num=(lp-lm)/(2*e); worst=max(worst,abs(num-gr[i])/(abs(num)+abs(gr[i])+1e-12))
    print(name,"max relative grad error:",worst)
check(M.build_ffnn(np.random.default_rng(0)),"FFNN")
check(M.build_cnn(np.random.default_rng(0)),"CNN")
