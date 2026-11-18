import sys, json, numpy as np, pandas as pd
from multiprocessing import Pool
import trexboost._selector as S
from trexboost import TRexBoostSelector
from trexboost._dummies import DummyGenerator

def data(seed, n=600):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({f"num{j}":rng.standard_normal(n) for j in range(12)})
    for j in range(12):  # 50-level categoricals, skewed frequencies
        pr=rng.dirichlet(np.ones(50)); df[f"cat{j}"]=pd.Categorical(rng.choice(50,n,p=pr))
    e0,e1=rng.standard_normal(50),rng.standard_normal(50)
    y=(0.8*e0[df.cat0.cat.codes]+0.8*e1[df.cat1.cat.codes]+0.6*df.num0+0.6*np.sin(2*df.num1)+0.4*df.num2*df.num3
       +rng.standard_normal(n))
    truth={"num0","num1","num2","num3","cat0","cat1"}
    return df,y.to_numpy(),truth

def run(args):
    mode,seed=args
    df,y,truth=data(seed)
    if mode=="naive":  # categorical splits on real columns, numeric Gaussian dummies
        S.DummyGenerator=lambda kind,X,cat=None: DummyGenerator("gaussian",X,None)
    else:
        S.DummyGenerator=DummyGenerator
    X=df
    spec="from_dtype"
    if mode=="ordinal":
        X=df.copy()
        for c in df.columns:
            if c.startswith("cat"): X[c]=df[c].cat.codes
        spec=None
    s=TRexBoostSelector(0.1,booster="xgboost",categorical_features=spec,max_rounds=1500,random_state=seed).fit(X,y)
    sel=set(s.get_feature_names_out())
    fd=len(sel-truth); 
    return dict(mode=mode,seed=seed,n_sel=len(sel),fdp=fd/max(1,len(sel)),tpr=len(sel&truth)/len(truth),
                cat_tp=len(sel&{"cat0","cat1"}), null_cat=len([c for c in sel-truth if c.startswith("cat")]), sel=sorted(sel))

if __name__=="__main__":
    jobs=[(m,s) for s in range(int(sys.argv[1])) for m in ["matched","naive","ordinal"]]
    with Pool(int(sys.argv[2])) as p:
        for r in p.imap_unordered(run,jobs):
            print(json.dumps(r),flush=True)
