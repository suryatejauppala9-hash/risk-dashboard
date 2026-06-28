import numpy as np
import pandas as pd


def compute_drawdown_series(returns: pd.Series)->pd.Series:
    cum=(1+returns).cumprod()
    rolling_peak=cum.cummax()
    drawdown=(cum-rolling_peak)/rolling_peak
    drawdown.name="Drawdown"
    return drawdown

def max_drawdown(returns: pd.Series)->float:
    dd=compute_drawdown_series(returns)
    return float(abs(dd.min()))

def drawdown_details(returns: pd.Series)->pd.DataFrame:
    cum=(1+returns).cumprod()
    dd_series=compute_drawdown_series(returns)
    periods=[]
    in_drawdown=False
    peak_date=None
    peak_val=None
    trough_date=None
    trough_val=None
    for date,val in cum.items():
        if not in_drawdown:
            if dd_series[date]<0:
                in_drawdown=True
                peak_date=cum[:date].idxmax()
                peak_val=cum[peak_date]
                trough_date=date
                trough_val=val
        else:
            if val<trough_val:
                trough_date=date
                trough_val=val
            if val>=peak_val:
                recovery_date=date
                dd_pct=(trough_val-peak_val)/peak_val
                periods.append({
                    "Peak": peak_date.date(),
                    "Trough": trough_date.date(),
                    "Recovery": recovery_date.date(),
                    "Drawdown": f"{dd_pct*100:.2f}%",
                    "Drawdown (raw)": dd_pct,
                    "Peak-to-Trough (d)": (trough_date-peak_date).days,
                    "Recovery (d)": (recovery_date-trough_date).days,
                })
                in_drawdown=False
    if in_drawdown and peak_date and trough_date:
        dd_pct=(trough_val-peak_val)/peak_val
        periods.append({
            "Peak":              peak_date.date(),
            "Trough":            trough_date.date(),
            "Recovery":          None,
            "Drawdown":          f"{dd_pct*100:.2f}%",
            "Drawdown (raw)":    dd_pct,
            "Peak-to-Trough (d)": (trough_date - peak_date).days,
            "Recovery (d)":      None,
        })

    if not periods:
        return pd.DataFrame()

    df = pd.DataFrame(periods)
    df = df.sort_values("Drawdown (raw)").head(5).reset_index(drop=True)
    df.index += 1
    return df

def avg_drawdown_duration(returns: pd.Series)->float:
    dd=compute_drawdown_series(returns)
    in_dd=dd<0
    durations=[]
    count=0
    for val in in_dd:
        if val:
            count+=1
        elif count>0:
            durations.append(count)
            count=0
    if count>0:
        durations.append(count)
    return float(np.mean(durations)) if durations else 0.0