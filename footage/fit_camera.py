import cv2, numpy as np, time, pickle, sys
from calib import *
from detect import detect
OV=[(55,30,505,72),(1090,20,1270,65),(0,650,420,720),(1215,645,1280,720)]
def make_obs(f):
    p,b=detect(f); m,g=line_mask(f,p,OV)
    ign=np.zeros(m.shape,np.uint8)
    for x1,y1,x2,y2 in p[:,:4].astype(int): ign[max(y1-3,0):y2+3,max(x1-3,0):x2+3]=1
    for x1,y1,x2,y2 in OV: ign[y1:y2,x1:x2]=1
    gd=cv2.dilate(g.astype(np.uint8),np.ones((13,13),np.uint8))
    return Obs(m,gd,ign)
if __name__=='__main__':
    ts=[int(a) for a in sys.argv[1:]]
    obs={t:make_obs(cv2.imread(f'frames/f_{t}.png')) for t in ts}
    cands=[]
    for D in [30,45,60,80]:
        for H in [12,20,28,38]:
            C=np.array([0.0,-HW-D,H]); tot=0; per={}
            for t,o in obs.items():
                s,p=grid_search(C,o)
                p2,s2=refine(p,C,o,iters=120)
                per[t]=(p2,s2); tot+=s2
            cands.append((tot,C,per)); print(D,H,round(tot,2),{t:round(v[1],2) for t,v in per.items()},flush=True)
    cands.sort(key=lambda c:c[0])
    pickle.dump(cands[0],open('cam_init.pkl','wb'))
