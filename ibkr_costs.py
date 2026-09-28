"""
IBKR cost model for stock fills on account DU7922803, used to show every bot's P&L NET of what IBKR actually charges.

Calibrated 2026-09-28 against all 54 stock fills in the saved activity statements (live_scripts/IBKR Audit Trail/,
24 Jul - 3 Aug 2026): model == IBKR "Comm/Fee" on 54/54 fills to the cent ($69.88 vs $69.88).
  commission  : USD 0.005/share, min USD 1.00, max 1% of trade value   (IBKR Pro Fixed)
  on sells    : + SEC fee 0.0000206 x value + FINRA TAF 0.000195/share (max USD 9.79)
  then        : + 9% GST (Singapore sales tax) on the whole charge

The bots record closed trades three ways (checked row by row against IBKR's statement "Realized P/L"):
  1. raw fill prices, dollar_pnl = (exit - entry) x shares               -> deduct entry + exit costs
  2. est_entry and/or exit_price taken from IBKR's commission-inclusive average price (off the cent grid by exactly
     the modelled charge)                                                -> deduct only the side(s) not already inside
  3. dollar_pnl copied from IBKR's realized P&L (it differs from (exit - entry) x shares) -> already net, deduct nothing
"""
import numpy as np
import pandas as pd

GST = 0.09
SEC_FEE_PER_USD = 20.6e-6
TAF_PER_SHARE, TAF_MAX = 0.000195, 9.79


def fill_cost(shares, price, is_sell: bool) -> float:
    sh = abs(float(shares)); val = sh * float(price)
    c = min(max(1.0, 0.005 * sh), 0.01 * val)
    if is_sell:
        c += SEC_FEE_PER_USD * val + min(TAF_PER_SHARE * sh, TAF_MAX)
    return round(c * (1 + GST), 2)


def _cost_in_price(px, shares, cost, is_sell) -> bool:
    """True if px is IBKR's commission-inclusive average price: buys (fill x sh + cost) / sh, sells (fill x sh - cost) / sh."""
    if abs(px - round(px, 2)) < 1e-4:
        return False                                   # on the cent grid -> raw fill price
    raw = (px * shares + (cost if is_sell else -cost)) / shares
    tol = 1e-4 + 0.006 / shares                        # px is stored to 4 dp and the charge to the cent
    return abs(raw - round(raw, 2)) < tol


def apply_ibkr_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Closed rows: gross_pnl = bot's dollar_pnl, ibkr_costs = modelled entry + exit charges, dollar_pnl = net."""
    if df.empty or 'dollar_pnl' not in df.columns or 'status' not in df.columns:
        return df
    df = df.copy()
    df['gross_pnl'] = pd.to_numeric(df['dollar_pnl'], errors='coerce')
    df['ibkr_costs'] = np.nan
    for i, r in df.iterrows():
        try:
            if r['status'] != 'closed' or pd.isna(df.at[i, 'gross_pnl']):
                continue
            sh = abs(float(r['shares'])); entry = float(r['est_entry']); exit_ = float(r['exit_price'])
        except (KeyError, TypeError, ValueError):
            continue
        if not (sh > 0 and entry > 0 and exit_ > 0):
            continue
        short = str(r.get('direction', 'LONG')).upper() == 'SHORT'
        price_pnl = (entry - exit_) * sh if short else (exit_ - entry) * sh
        if abs(df.at[i, 'gross_pnl'] - price_pnl) > 0.02:
            df.at[i, 'ibkr_costs'] = 0.0               # case 3: already IBKR's net realized P&L
            continue
        entry_cost = fill_cost(sh, entry, is_sell=short)
        exit_cost = fill_cost(sh, exit_, is_sell=not short)
        if _cost_in_price(exit_, sh, entry_cost + exit_cost, is_sell=not short):
            df.at[i, 'ibkr_costs'] = 0.0               # exit price already carries the whole round-trip charge
            continue
        cost = (0.0 if _cost_in_price(entry, sh, entry_cost, is_sell=short) else entry_cost) \
             + (0.0 if _cost_in_price(exit_, sh, exit_cost, is_sell=not short) else exit_cost)
        df.at[i, 'ibkr_costs'] = round(cost, 2)
        df.at[i, 'dollar_pnl'] = round(df.at[i, 'gross_pnl'] - cost, 2)
    return df
