import numpy as np, pandas as pd, joblib, os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

N=500; rng=np.random.RandomState(42)
rent=rng.beta(8,2,N); util=rng.beta(7,3,N)
upi=rng.poisson(30,N); std=rng.gamma(2,2,N)
recharge=rng.poisson(2,N); inc=rng.beta(5,3,N)
scores=(1-rent)*.4+(1-util)*.2+(std/std.max())*.2+(1-inc)*.2
labels=pd.cut(scores,[-1,.2,.5,2],labels=['Low','Medium','High'])
X=pd.DataFrame({'rent_on_time_ratio':rent,'utility_on_time_ratio':util,'avg_monthly_upi_txn':upi,'std_upi_txn':std,'mobile_recharge_freq':recharge,'income_stability_score':inc})
y=labels
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2)
m=GradientBoostingClassifier().fit(Xtr,ytr)
print('trained:',m.score(Xte,yte))
out=os.path.join(os.path.dirname(__file__),'saved_models','model.joblib')
os.makedirs(os.path.dirname(out),exist_ok=True)
joblib.dump(m,out)
print('saved',out)
