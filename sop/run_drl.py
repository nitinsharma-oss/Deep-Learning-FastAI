import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
import random, copy, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=64

class Env:
    def __init__(s,A_B=1.5,A_E=0.5,Pn=0.1,Px=5.,Pb=2.,lam=.3,rB=.85,rE=.85,hn=.2,hx=3.):
        s.A_B,s.A_E,s.Pn,s.Px,s.Pb,s.lam=A_B,A_E,Pn,Px,Pb,lam
        s.rB,s.rE,s.hn,s.hx=rB,rE,hn,hx
    def reset(s):
        s.hB=np.random.uniform(s.hn,s.hx); s.hE=np.random.uniform(s.hn,s.hx); return s._s()
    def _s(s): return np.array([(s.hB-s.hn)/(s.hx-s.hn),(s.hE-s.hn)/(s.hx-s.hn)],dtype=np.float32)
    def step(s,a):
        P=np.clip(a,s.Pn,s.Px); r=max(.5*np.log(2+s.A_B*P*s.hB)-.5*np.log(2+s.A_E*P*s.hE),0)-s.lam*(P-s.Pb)
        s.hB=np.clip(s.rB*s.hB+(1-s.rB)*np.random.uniform(s.hn,s.hx),s.hn,s.hx)
        s.hE=np.clip(s.rE*s.hE+(1-s.rE)*np.random.uniform(s.hn,s.hx),s.hn,s.hx)
        return s._s(),r,False
    def ag(s,P,hB,hE): return .5*(s.A_B*hB/(2+s.A_B*P*hB)-s.A_E*hE/(2+s.A_E*P*hE))-s.lam

class RB:
    def __init__(s,c=80000): s.b=deque(maxlen=c)
    def push(s,*a): s.b.append(a)
    def sample(s,n):
        b=random.sample(s.b,n); st,a,r,s2,d=zip(*b)
        return(torch.tensor(np.array(st),dtype=torch.float32),torch.tensor(a,dtype=torch.float32).unsqueeze(1),torch.tensor(r,dtype=torch.float32).unsqueeze(1),torch.tensor(np.array(s2),dtype=torch.float32),torch.tensor(d,dtype=torch.float32).unsqueeze(1))
    def __len__(s): return len(s.b)

def mk(d):
    l=[]
    for i in range(len(d)-1): l+=[nn.Linear(d[i],d[i+1])]; (l.append(nn.ReLU()) if i<len(d)-2 else None)
    return nn.Sequential(*l)

def su(t,s,tau=.005):
    for tp,sp in zip(t.parameters(),s.parameters()): tp.data.copy_(tau*sp.data+(1-tau)*tp.data)

def tp(e,a): return e.Pn+a*(e.Px-e.Pn)
def ev(ag,e,n=5):
    t=0
    for _ in range(n):
        s=e.reset(); er=0
        for _ in range(200):
            with torch.no_grad(): v=ag(torch.tensor(s).unsqueeze(0)).item()
            s,r,_=e.step(tp(e,v)); er+=r
        t+=er/200
    return t/n

# ========== DDPG ==========
def train_ddpg(steps=15000):
    e=Env(); a=nn.Sequential(mk([2,H,H,1]),nn.Sigmoid()); at=copy.deepcopy(a)
    c=mk([3,H,H,1]); ct=copy.deepcopy(c); buf=RB()
    oa=torch.optim.Adam(a.parameters(),3e-4); oc=torch.optim.Adam(c.parameters(),3e-4)
    s=e.reset(); log=[]; bs=128
    for i in range(1,steps+1):
        with torch.no_grad(): v=a(torch.tensor(s).unsqueeze(0)).item()
        v=np.clip(v+np.random.normal(0,.2),0,1); P=tp(e,v)
        s2,r,_=e.step(P); buf.push(s,P,r,s2,0.); s=s2
        if i%200==0: s=e.reset()
        if len(buf)>=bs:
            S,A,R,S2,D=buf.sample(bs)
            with torch.no_grad(): a2=at(S2); qt=R+.99*(1-D)*ct(torch.cat([S2,tp(e,a2)],1))
            oc.zero_grad(); (c(torch.cat([S,A],1))-qt).pow(2).mean().backward(); oc.step()
            ap=a(S); oa.zero_grad(); (-c(torch.cat([S,tp(e,ap)],1))).mean().backward(); oa.step()
            su(at,a); su(ct,c)
        if i%1000==0: log.append((i,ev(a,e))); print(f'  DDPG {i:6d} {log[-1][1]:.4f}')
    return a,'DDPG',log

# ========== SAC ==========
class SACa(nn.Module):
    def __init__(s): super().__init__(); s.sh=mk([2,H,H]); s.mu=nn.Linear(H,1); s.ls=nn.Linear(H,1)
    def forward(s,x): h=F.relu(s.sh(x)); return s.mu(h),s.ls(h).clamp(-5,2)
    def sample(s,x):
        mu,ls=s(x); std=ls.exp(); z=mu+std*torch.randn_like(std); a=torch.sigmoid(z)
        lp=-.5*((z-mu)/std)**2-ls-.5*math.log(2*math.pi)-torch.log(a*(1-a)+1e-8)
        return a,lp

def train_sac(steps=15000):
    e=Env(); a=SACa(); c1=mk([3,H,H,1]); c2=mk([3,H,H,1])
    c1t=copy.deepcopy(c1); c2t=copy.deepcopy(c2); buf=RB()
    la=torch.tensor(0.,requires_grad=True); te=-1.; bs=128
    oa=torch.optim.Adam(a.parameters(),3e-4); oc=torch.optim.Adam(list(c1.parameters())+list(c2.parameters()),3e-4)
    oal=torch.optim.Adam([la],3e-4); s=e.reset(); log=[]
    for i in range(1,steps+1):
        with torch.no_grad(): ac,_=a.sample(torch.tensor(s).unsqueeze(0))
        P=tp(e,ac.item()); s2,r,_=e.step(P); buf.push(s,P,r,s2,0.); s=s2
        if i%200==0: s=e.reset()
        if len(buf)>=bs:
            S,A,R,S2,D=buf.sample(bs); al=la.exp().detach()
            with torch.no_grad():
                a2,lp2=a.sample(S2); ap2=tp(e,a2); sa2=torch.cat([S2,ap2],1)
                qt=R+.99*(1-D)*(torch.min(c1t(sa2),c2t(sa2))-al*lp2)
            sa=torch.cat([S,A],1); lc=(c1(sa)-qt).pow(2).mean()+(c2(sa)-qt).pow(2).mean()
            oc.zero_grad(); lc.backward(); oc.step()
            an,lpn=a.sample(S); apn=tp(e,an); san=torch.cat([S,apn],1)
            lac=(al*lpn-torch.min(c1(san),c2(san))).mean()
            oa.zero_grad(); lac.backward(); oa.step()
            la2=-(la.exp()*(lpn.detach()+te)).mean(); oal.zero_grad(); la2.backward(); oal.step()
            su(c1t,c1); su(c2t,c2)
        if i%1000==0:
            def evs(st,ex=False):
                with torch.no_grad(): return torch.sigmoid(a(torch.tensor(st).unsqueeze(0))[0]).item()
            aa=nn.Module(); aa.forward=lambda x: torch.sigmoid(a(x)[0])
            # inline eval
            t=0
            for _ in range(5):
                ss=e.reset(); er=0
                for _ in range(200):
                    with torch.no_grad(): v=torch.sigmoid(a(torch.tensor(ss).unsqueeze(0))[0]).item()
                    ss,r,_=e.step(tp(e,v)); er+=r
                t+=er/200
            log.append((i,t/5)); print(f'  SAC  {i:6d} {log[-1][1]:.4f}')
    # wrap for eval
    class W(nn.Module):
        def __init__(s2,aa): super().__init__(); s2.aa=aa
        def forward(s2,x): return torch.sigmoid(s2.aa(x)[0])
    return W(a),'SAC',log

# ========== PPO ==========
class PPOa(nn.Module):
    def __init__(s): super().__init__(); s.sh=mk([2,H,H]); s.mu=nn.Linear(H,1); s.ls=nn.Parameter(torch.zeros(1))
    def forward(s,x): h=F.relu(s.sh(x)); return torch.sigmoid(s.mu(h))
    def lp(s,x,a): mu=s(x); std=s.ls.exp(); return -.5*((a-mu)/std)**2-torch.log(std)-.5*math.log(2*math.pi)
    def sample(s,x): mu=s(x); std=s.ls.exp(); return torch.clamp(mu+std*torch.randn_like(mu),0,1)

def train_ppo(steps=15000,hz=256):
    e=Env(); a=PPOa(); v=mk([2,H,H,1]); bs=64; K=8; ec=.2
    oa=torch.optim.Adam(a.parameters(),3e-4); ov=torch.optim.Adam(v.parameters(),3e-4)
    log=[]; st=0
    while st<steps:
        SS,AA,RR,OL=[],[],[],[]; ss=e.reset()
        for _ in range(hz):
            t=torch.tensor(ss).unsqueeze(0); ac=a.sample(t); lpp=a.lp(t,ac)
            P=tp(e,ac.item()); s2,r,_=e.step(P)
            SS.append(ss); AA.append(ac.item()); RR.append(r); OL.append(lpp.item()); ss=s2
        S=torch.tensor(np.array(SS),dtype=torch.float32); A=torch.tensor(AA,dtype=torch.float32).unsqueeze(1)
        R=np.array(RR); V=v(S).squeeze().detach().numpy()
        adv=np.zeros_like(R); g2=0; Ve=np.append(V,0)
        for t in reversed(range(len(R))): d=R[t]+.99*Ve[t+1]-Ve[t]; g2=d+.99*.95*g2; adv[t]=g2
        ret=torch.tensor(adv+V,dtype=torch.float32).unsqueeze(1); adv_t=torch.tensor(np.array(adv),dtype=torch.float32).unsqueeze(1)
        adv_t=(adv_t-adv_t.mean())/(adv_t.std()+1e-8); olp=torch.tensor(OL,dtype=torch.float32).unsqueeze(1)
        for _ in range(K):
            ix=np.random.permutation(len(R))
            for j in range(0,len(R),bs):
                k=ix[j:j+bs]; sb,ab,adb,rb,ob=S[k],A[k],adv_t[k],ret[k],olp[k]
                lp=a.lp(sb,ab); rat=(lp-ob).exp()
                la=-torch.min(rat*adb,torch.clamp(rat,1-ec,1+ec)*adb).mean()
                oa.zero_grad(); la.backward(); oa.step()
                lv=(v(sb)-rb).pow(2).mean(); ov.zero_grad(); lv.backward(); ov.step()
        st+=hz
        if st%1024<hz: rv=ev(a,e); log.append((st,rv)); print(f'  PPO  {st:6d} {rv:.4f}')
    return a,'PPO',log

# ========== ECDPO (Hybrid) ==========
def train_ecdpo(steps=15000):
    e=Env(); a=nn.Sequential(mk([2,H,H,1]),nn.Sigmoid()); ao=copy.deepcopy(a)
    c1=mk([3,H,H,1]); c2=mk([3,H,H,1]); c1t=copy.deepcopy(c1); c2t=copy.deepcopy(c2); buf=RB()
    oa=torch.optim.Adam(a.parameters(),3e-4); oc=torch.optim.Adam(list(c1.parameters())+list(c2.parameters()),3e-4)
    s=e.reset(); log=[]; bs=128; uc=0
    for i in range(1,steps+1):
        with torch.no_grad(): v=a(torch.tensor(s).unsqueeze(0)).item()
        v=np.clip(v+np.random.normal(0,.15),0,1); P=tp(e,v)
        s2,r,_=e.step(P); buf.push(s,P,r,s2,0.); s=s2
        if i%200==0: s=e.reset()
        if len(buf)>=bs:
            S,A,R,S2,D=buf.sample(bs)
            with torch.no_grad():
                a2=a(S2); sa2=torch.cat([S2,tp(e,a2)],1)
                qt=R+.99*(1-D)*torch.min(c1t(sa2),c2t(sa2))
            sa=torch.cat([S,A],1); lc=(c1(sa)-qt).pow(2).mean()+(c2(sa)-qt).pow(2).mean()
            oc.zero_grad(); lc.backward(); oc.step()
            an=a(S); apn=tp(e,an); san=torch.cat([S,apn],1)
            qv=torch.min(c1(san),c2(san))
            ent=-(an*torch.log(an+1e-8)+(1-an)*torch.log(1-an+1e-8)).mean()
            la=-qv.mean()-0.05*ent
            oa.zero_grad(); la.backward(); oa.step()
            su(c1t,c1); su(c2t,c2)
        if i%1000==0: log.append((i,ev(a,e))); print(f'  ECDPO {i:6d} {log[-1][1]:.4f}')
    return a,'ECDPO',log

# ========== CAPG (Novel) ==========
def train_capg(steps=15000):
    e=Env(); a=nn.Sequential(mk([2,H,H,1]),nn.Sigmoid()); at=copy.deepcopy(a)
    c=mk([3,H,H,1]); ct=copy.deepcopy(c); buf=RB()
    oa=torch.optim.Adam(a.parameters(),3e-4); oc=torch.optim.Adam(c.parameters(),3e-4)
    lb=torch.tensor(0.,requires_grad=True); ob=torch.optim.Adam([lb],1e-3)
    s=e.reset(); log=[]; bs=128
    for i in range(1,steps+1):
        with torch.no_grad(): v=a(torch.tensor(s).unsqueeze(0)).item()
        v=np.clip(v+np.random.normal(0,.15),0,1); P=tp(e,v)
        s2,r,_=e.step(P); buf.push(s,P,r,s2,0.); s=s2
        if i%200==0: s=e.reset()
        if len(buf)>=bs:
            S,A,R,S2,D=buf.sample(bs)
            with torch.no_grad(): a2=at(S2); qt=R+.99*(1-D)*ct(torch.cat([S2,tp(e,a2)],1))
            oc.zero_grad(); (c(torch.cat([S,A],1))-qt).pow(2).mean().backward(); oc.step()
            ap=a(S); app=tp(e,ap)
            lcr=-c(torch.cat([S,app],1)).mean()
            hB=S[:,0]*(e.hx-e.hn)+e.hn; hE=S[:,1]*(e.hx-e.hn)+e.hn
            Pd=app.squeeze().detach()
            gr=.5*(e.A_B*hB/(2+e.A_B*Pd*hB)-e.A_E*hE/(2+e.A_E*Pd*hE))-e.lam
            lan=-(gr.detach()*ap.squeeze()).mean()
            beta=torch.sigmoid(lb)
            la=(1-beta)*lcr+beta*lan; oa.zero_grad(); la.backward(); oa.step()
            su(at,a); su(ct,c)
        if i%1000==0: log.append((i,ev(a,e))); print(f'  CAPG {i:6d} {log[-1][1]:.4f}')
    return a,'CAPG',log

# ========== MAIN ==========
print('='*55); print(' Training 5 DRL agents for wiretap power control'); print('='*55)
results={}
for fn,nm in [(train_ddpg,'DDPG'),(train_sac,'SAC'),(train_ppo,'PPO'),(train_ecdpo,'ECDPO'),(train_capg,'CAPG')]:
    print(f'\n--- {nm} ---'); _,_,log=fn(8000); results[nm]=log

# Water-filling baseline
e=Env(); wr=[]; s=e.reset()
for _ in range(2000):
    hB=s[0]*(e.hx-e.hn)+e.hn; hE=s[1]*(e.hx-e.hn)+e.hn
    lo,hi=e.Pn,e.Px
    for _ in range(30): mid=(lo+hi)/2; g=e.ag(mid,hB,hE); lo,hi=(mid,hi) if g>0 else (lo,mid)
    P=(lo+hi)/2
    if e.A_B*hB<=e.A_E*hE: P=e.Pn
    s,r,_=e.step(P); wr.append(r)
wf=np.mean(wr); print(f'\nWater-filling baseline: {wf:.4f}')

fig,ax=plt.subplots(figsize=(10,6))
cols={'DDPG':'#2a78d6','SAC':'#1baf7a','PPO':'#eda100','ECDPO':'#e34948','CAPG':'#4a3aa7'}
for nm,log in results.items():
    xs,ys=zip(*log); ax.plot(xs,ys,label=nm,color=cols[nm],lw=2,alpha=.85)
ax.axhline(wf,color='gray',ls='--',lw=1.5,label='Water-filling')
ax.set_xlabel('Steps',fontsize=13); ax.set_ylabel('Avg secrecy reward/step',fontsize=13)
ax.set_title('DRL Power Control: ASC Maximisation',fontsize=14,fontweight='bold')
ax.legend(fontsize=11); ax.grid(True,alpha=.3); plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/drl_comparison.png',dpi=150)
print('Plot saved.\n')
print(f'{"Method":<12} {"Final reward":>14}')
print('-'*28)
for nm,log in results.items(): print(f'{nm:<12} {log[-1][1]:>14.4f}')
print(f'{"WaterFill":<12} {wf:>14.4f}')
