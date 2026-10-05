import pandas as pd, numpy as np, xgboost as xgb, time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, average_precision_score
p=pd.read_parquet('data/processed/passes.parquet')
p=p[(p.pass_len>=0.5)&p.nearest_opp_dist.notna()]
cat=['play_pattern','pass_type','body_part','pass_height']
num=['under_pressure','score_diff','period','minute','ball_x','ball_y','ball_dist_goal','ball_angle_goal','opp_within_5','nearest_opp_dist','opp_in_cone','def_line_x','tm_ahead','n_visible_teammates','n_visible_opponents','visible_area','end_x','end_y','pass_len','pass_angle','progress_x','end_dist_goal','end_angle_goal','opp_near_target_3','lane_opp_2','target_in_box']
X=pd.get_dummies(p[num+cat],columns=cat,dtype=float); loc=['ball_x','ball_y','end_x','end_y']
tr,te=p.split=='train',p.split=='test'
res=[]
def ev(name,task,y,pr):
    res.append(dict(task=task,model=name,logloss=log_loss(y,pr),brier=brier_score_loss(y,pr),auc=roc_auc_score(y,pr),prauc=average_precision_score(y,pr)))
for task,ycol,mask in [('pass_success','y_success',np.ones(len(p),bool)),('shot10|completed','y_shot10',(p.y_success==1).values)]:
    m_tr,m_te=tr.values&mask,te.values&mask; y=p[ycol].values
    ev('B0 prior',task,y[m_te],np.full(m_te.sum(),y[m_tr].mean()))
    lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0)).fit(X[m_tr],y[m_tr])
    ev('B1 logistic',task,y[m_te],lr.predict_proba(X[m_te])[:,1])
    for nm,cols in [('B3 XGB location-only',loc),('B2 XGB all features',list(X.columns))]:
        t=time.time()
        m=xgb.XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,n_jobs=1,eval_metric='logloss').fit(X.loc[m_tr,cols],y[m_tr])
        ev(nm,task,y[m_te],m.predict_proba(X.loc[m_te,cols])[:,1]); print(nm,task,round(time.time()-t,1),'s',flush=True)
r=pd.DataFrame(res).round(4); print(r.to_string(index=False)); r.to_csv('quick_baselines.csv',index=False)
