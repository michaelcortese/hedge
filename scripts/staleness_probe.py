"""Staleness-decay probe (Strategy B build/no-build gate).

Polls KXBTCPERP mid at ~2Hz. When it moves >= MOVE_BPS within <= 3s, snapshots
the nearest-ATM binary books at +0.3/0.7/1.5/3/6s and logs whether the quotes
repriced. Answers: how long do stale binary quotes persist after a spot move,
and how much edge do they carry?"""
import json, math, time, urllib.request
from datetime import datetime, timezone

PERP="https://external-api.kalshi.com/trade-api/v2"
EVT="https://api.elections.kalshi.com/trade-api/v2"
UA={"User-Agent":"probe/0.1"}
MOVE_BPS=4.0   # trigger threshold

def get(base,path):
    with urllib.request.urlopen(urllib.request.Request(base+path,headers=UA),timeout=8) as r:
        return json.load(r)

def perp_mid():
    ob=get(PERP,"/margin/markets/KXBTCPERP/orderbook?depth=1")["orderbook"]
    # asks/bids lists; best = min ask, max bid
    best_ask=min(float(p) for p,_ in ob["asks"]); best_bid=max(float(p) for p,_ in ob["bids"])
    return (best_ask+best_bid)/2/0.0001

def atm_targets():
    """nearest-ATM strikes of front hourly + current 15M."""
    out=[]
    for series in ("KXBTC15M","KXBTCD"):
        try: d=get(EVT,f"/markets?series_ticker={series}&status=open&limit=100")
        except Exception: continue
        now=datetime.now(timezone.utc)
        for m in d.get("markets",[]):
            ct=datetime.fromisoformat(m["close_time"].replace("Z","+00:00"))
            mins=(ct-now).total_seconds()/60
            if not (2<mins<45): continue
            yb=float(m.get("yes_bid_dollars",0) or 0); ya=float(m.get("yes_ask_dollars",1) or 1)
            if 0.15<= (yb+ya)/2 <=0.85 and m.get("floor_strike"):
                out.append((m["ticker"], m["floor_strike"]))
    return out[:4]

def book(t):
    try:
        ob=get(EVT,f"/markets/{t}/orderbook?depth=1").get("orderbook_fp",{})
        yes,no=ob.get("yes_dollars") or [],ob.get("no_dollars") or []
        if not yes or not no: return None
        return float(yes[-1][0]), 1-float(no[-1][0])
    except Exception:
        return None

def main(minutes=25):
    log=open("staleness_log.jsonl","a")
    end=time.time()+minutes*60
    hist=[]  # (ts, mid)
    targets=[]; last_targets=0
    n_events=0
    while time.time()<end:
        t0=time.time()
        try: mid=perp_mid()
        except Exception: time.sleep(0.6); continue
        hist.append((t0,mid)); hist=[(t,m) for t,m in hist if t0-t<=4]
        if t0-last_targets>60:
            targets=atm_targets(); last_targets=t0
        base=hist[0][1]
        move=(mid/base-1)*1e4
        if abs(move)>=MOVE_BPS and targets:
            n_events+=1
            pre={}
            for tk,K in targets: pre[tk]=book(tk)
            snaps={"0.0":pre}
            for delay in (0.7,1.5,3.0,6.0):
                while time.time()-t0<delay: time.sleep(0.05)
                snaps[str(delay)]={tk:book(tk) for tk,_ in targets}
            rec=dict(ts=t0,move_bps=round(move,2),mid=mid,base=base,
                     targets={tk:K for tk,K in targets},snaps=snaps)
            log.write(json.dumps(rec)+"\n"); log.flush()
            print(f"event {n_events}: move {move:+.1f}bps @ {datetime.fromtimestamp(t0,timezone.utc):%H:%M:%S}",flush=True)
            hist=[]
        time.sleep(max(0.05,0.5-(time.time()-t0)))
    print(f"done, {n_events} events",flush=True)

if __name__=="__main__":
    main(25)
