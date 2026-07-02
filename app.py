"""
AlphaX Paper Trade Dashboard
Displays live P&L, equity curves, drawdown and trade logs for:
  AEARN-MOMO-001, CNP-DKPL-V2, ODFL-NETPREM-001,
  NOPE-EOD-001, EARN-B03-001, EPS-BCDC-001,
  FSCORE-001, HURST-001, INST-ACT-AVGDOWN-001
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
LIVE_MARKS_FILE = os.path.join(DATA_DIR, 'live_marks.csv')

STRATEGIES = {
    'AEARN-MOMO-001'  : {'file': 'aearn_trades.csv',   'color': '#2ecc71', 'capital': 10000},
    'CNP-DKPL-V2'     : {'file': 'cnp_trades.csv',     'color': '#3498db', 'capital': 10000},
    'ODFL-NETPREM-001': {'file': 'odfl_trades.csv',    'color': '#e67e22', 'capital': 10000},
    'NOPE-EOD-001'    : {'file': 'nope_trades.csv',    'color': '#a855f7', 'capital': 10000},
    'EARN-B03-001'    : {'file': 'earn_trades.csv',    'color': '#f59e0b', 'capital': 10000},
    'EPS-BCDC-001'    : {'file': 'eps_trades.csv',     'color': '#06b6d4', 'capital': 10000},
    'FSCORE-001'      : {'file': 'fscore_trades.csv',  'color': '#ec4899', 'capital': 10000},
    'HURST-001'            : {'file': 'hurst_trades.csv',        'color': '#84cc16', 'capital': 10000},
    'INST-ACT-AVGDOWN-001' : {'file': 'inst_avgdown_trades.csv', 'color': '#f97316', 'capital':  5000},
}

# Columns to show in trade table per strategy (extra signal columns)
EXTRA_COLS = {
    'NOPE-EOD-001' : ['z_score', 'eod_nope'],
    'EARN-B03-001' : ['reaction_pct', 'expected_pct', 'source'],
    'EPS-BCDC-001' : ['score'],
    'FSCORE-001'   : ['score'],
    'HURST-001'            : ['hurst', 'regime'],
    'INST-ACT-AVGDOWN-001' : ['discount_pct', 'institution'],
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_trades(filepath):
    """
    Load trade CSV safely.
    Normalises column names across all 5 strategies:
      - entry_date / date  → date
      - ticker / symbol    → symbol
      - NOPE trades have no direction → filled as LONG
    """
    if not os.path.exists(filepath):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return pd.DataFrame()

        # Normalise date column
        if 'entry_date' in df.columns and 'date' not in df.columns:
            df = df.rename(columns={'entry_date': 'date'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

        # Normalise ticker / symbol
        if 'ticker' in df.columns and 'symbol' not in df.columns:
            df = df.rename(columns={'ticker': 'symbol'})

        # NOPE is always LONG — add direction if missing
        if 'direction' not in df.columns:
            df['direction'] = 'LONG'

        return df
    except Exception:
        return pd.DataFrame()


def compute_stats(df, capital=10000):
    """Compute key performance stats from trade log."""
    empty = {
        'total_pnl': 0, 'total_return': 0, 'sharpe': 0,
        'max_dd': 0, 'win_rate': 0, 'n_trades': 0, 'profit_factor': 0,
    }
    if df.empty or 'dollar_pnl' not in df.columns:
        return {**empty, 'n_trades': len(df)}   # show trade count even before close

    pnl = pd.to_numeric(df['dollar_pnl'], errors='coerce').dropna().values
    if len(pnl) == 0:
        return {**empty, 'n_trades': len(df)}

    cum   = np.cumsum(pnl)
    peak  = np.maximum.accumulate(cum)
    dd    = cum - peak
    wins  = pnl[pnl > 0]
    loss  = pnl[pnl < 0]
    ret   = pnl / capital * 100
    sharpe = (ret.mean() / ret.std(ddof=1) * np.sqrt(252)) if ret.std(ddof=1) > 0 else 0

    return {
        'total_pnl'    : round(cum[-1], 2),
        'total_return' : round(cum[-1] / capital * 100, 2),
        'sharpe'       : round(sharpe, 2),
        'max_dd'       : round(dd.min(), 2),
        'win_rate'     : round((pnl > 0).mean() * 100, 1),
        'n_trades'     : len(pnl),
        'profit_factor': round(wins.sum() / abs(loss.sum()), 2) if len(loss) > 0 else 0,
    }


def equity_curve(df, capital=10000):
    if df.empty or 'dollar_pnl' not in df.columns:
        return [], []
    pnl = pd.to_numeric(df['dollar_pnl'], errors='coerce').fillna(0).values
    return df['date'].tolist(), (capital + np.cumsum(pnl)).tolist()


def load_live_marks():
    """
    Load the unrealized-P&L snapshot written by sync_live_marks.py.

    This is a periodic snapshot (every ~15 min during market hours), not a
    live feed -- the dashboard runs on Render and can't reach the local IBKR
    Gateway directly. Returns (DataFrame, latest 'as_of' timestamp or None).
    """
    if not os.path.exists(LIVE_MARKS_FILE):
        return pd.DataFrame(), None
    try:
        df = pd.read_csv(LIVE_MARKS_FILE)
        if df.empty:
            return df, None
        df['as_of'] = pd.to_datetime(df['as_of'], errors='coerce')
        latest = df['as_of'].max()
        return df, latest
    except Exception:
        return pd.DataFrame(), None


# ── APP ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title='AlphaX Dashboard')
server = app.server  # for Render/gunicorn

CARD_STYLE_BASE = {
    'backgroundColor': '#161b22', 'borderRadius': '10px',
    'padding': '20px', 'flex': '1', 'minWidth': '200px',
}
SECTION_STYLE = {
    'backgroundColor': '#161b22', 'borderRadius': '10px',
    'padding': '20px', 'marginBottom': '25px',
}
H3_STYLE = {'color': '#58a6ff', 'marginBottom': '10px'}

# ── LAYOUT ────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={'backgroundColor': '#0d1117', 'minHeight': '100vh',
           'fontFamily': 'Arial', 'color': '#e6edf3', 'padding': '20px'},
    children=[

        # ── Header ───────────────────────────────────────────────────────────
        html.Div(style={'textAlign': 'center', 'marginBottom': '30px'}, children=[
            html.H1('🤖 AlphaX Paper Trade Dashboard',
                    style={'color': '#58a6ff', 'marginBottom': '5px'}),
            html.P('9 Strategies  ·  IBKR Paper Account DU7922803',
                   style={'color': '#8b949e', 'fontSize': '13px', 'marginBottom': '3px'}),
            html.P(id='last-updated', style={'color': '#8b949e', 'fontSize': '12px'}),
            html.P(id='marks-updated', style={'color': '#8b949e', 'fontSize': '12px'}),
        ]),

        # ── Strategy Cards (row 1: original 3, row 2: new 2) ─────────────────
        html.Div([
            html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px',
                            'flexWrap': 'wrap'},
                     id='cards-row1'),
            html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px',
                            'flexWrap': 'wrap'},
                     id='cards-row2'),
            html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px',
                            'flexWrap': 'wrap'},
                     id='cards-row3'),
        ], style={'marginBottom': '10px'}),

        # ── Combined Card ─────────────────────────────────────────────────────
        html.Div(id='combined-card', style={'marginBottom': '25px'}),

        # ── Equity Curves ─────────────────────────────────────────────────────
        html.Div(style=SECTION_STYLE, children=[
            html.H3('📈 Equity Curves', style=H3_STYLE),
            dcc.Graph(id='equity-chart', style={'height': '400px'}),
        ]),

        # ── Drawdown ──────────────────────────────────────────────────────────
        html.Div(style=SECTION_STYLE, children=[
            html.H3('📉 Drawdown', style=H3_STYLE),
            dcc.Graph(id='drawdown-chart', style={'height': '250px'}),
        ]),

        # ── Monthly P&L Heatmap ───────────────────────────────────────────────
        html.Div(style=SECTION_STYLE, children=[
            html.H3('📊 Monthly P&L ($) — All Strategies', style=H3_STYLE),
            dcc.Graph(id='monthly-chart', style={'height': '380px'}),
        ]),

        # ── Strategy Breakdown Bar ────────────────────────────────────────────
        html.Div(style=SECTION_STYLE, children=[
            html.H3('🏆 P&L by Strategy', style=H3_STYLE),
            dcc.Graph(id='strategy-bar', style={'height': '280px'}),
        ]),

        # ── Trade Log ─────────────────────────────────────────────────────────
        html.Div(style=SECTION_STYLE, children=[
            html.H3('📋 Trade Log', style=H3_STYLE),
            dcc.Dropdown(
                id='strategy-filter',
                options=[{'label': 'All Strategies', 'value': 'ALL'}] +
                        [{'label': s, 'value': s} for s in STRATEGIES],
                value='ALL',
                style={'backgroundColor': '#21262d', 'color': '#000',
                       'width': '320px', 'marginBottom': '12px'},
            ),
            html.Div(id='trade-table'),
        ]),

        # Auto-refresh every 5 min
        dcc.Interval(id='interval', interval=5 * 60 * 1000, n_intervals=0),
    ]
)

# ── CALLBACKS ─────────────────────────────────────────────────────────────────
@app.callback(
    Output('last-updated',   'children'),
    Output('marks-updated',  'children'),
    Output('cards-row1',     'children'),
    Output('cards-row2',     'children'),
    Output('cards-row3',     'children'),
    Output('combined-card',  'children'),
    Output('equity-chart',   'figure'),
    Output('drawdown-chart', 'figure'),
    Output('monthly-chart',  'figure'),
    Output('strategy-bar',   'figure'),
    Input('interval',        'n_intervals'),
)
def update_dashboard(n):
    now        = datetime.now().strftime('%Y-%m-%d %H:%M SGT')
    all_trades = []
    all_stats  = {}
    all_unrealized = {}
    cards      = {}
    eq_fig     = go.Figure()
    dd_fig     = go.Figure()

    strategy_names = list(STRATEGIES.keys())

    df_marks, marks_as_of = load_live_marks()

    # Unrealized P&L snapshot is periodic (see sync_live_marks.py docstring),
    # not real-time -- surface how stale it is so a missed sync is obvious
    # rather than silently showing outdated numbers as if they were current.
    if marks_as_of is None:
        marks_text = 'Unrealized P&L: no live marks snapshot found yet.'
    else:
        # sync_live_marks.py writes 'as_of' in US Eastern Time (now_et()) --
        # compare against ET "now" here too, not server-local time (SGT
        # locally, UTC on Render), which would silently give a bogus offset.
        now_et_naive = datetime.now(ZoneInfo('America/New_York')).replace(tzinfo=None)
        age_min = (now_et_naive - marks_as_of.to_pydatetime().replace(tzinfo=None)).total_seconds() / 60
        staleness = ' ⚠️ STALE' if age_min > 45 else ''
        marks_text = f'Unrealized P&L as of: {marks_as_of.strftime("%Y-%m-%d %H:%M")} ET{staleness}'

    for name, cfg in STRATEGIES.items():
        filepath = os.path.join(DATA_DIR, cfg['file'])
        df       = load_trades(filepath)
        stats    = compute_stats(df, cfg['capital'])
        color    = cfg['color']
        capital  = cfg['capital']
        all_stats[name] = stats

        unrealized = 0.0
        if not df_marks.empty:
            unrealized = float(df_marks.loc[df_marks['strategy'] == name, 'unrealized_pnl'].sum())
        all_unrealized[name] = unrealized

        pnl_color   = '#2ecc71' if stats['total_pnl'] >= 0 else '#e74c3c'
        unreal_color = '#2ecc71' if unrealized >= 0 else '#e74c3c'

        # ── Strategy stat card ──────────────────────────────────────────────
        card = html.Div(style={
            **CARD_STYLE_BASE,
            'borderLeft': f'4px solid {color}',
        }, children=[
            html.H4(name, style={'color': color, 'marginBottom': '10px',
                                 'fontSize': '13px', 'fontWeight': 'bold'}),
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr',
                            'gap': '8px'}, children=[
                html.Div([html.Span('Realized P&L', style={'color': '#8b949e', 'fontSize': '11px'}),
                          html.P(f'${stats["total_pnl"]:+,.2f}',
                                 style={'color': pnl_color, 'margin': '2px 0', 'fontWeight': 'bold'})]),
                html.Div([html.Span('Unrealized P&L', style={'color': '#8b949e', 'fontSize': '11px'}),
                          html.P(f'${unrealized:+,.2f}',
                                 style={'color': unreal_color, 'margin': '2px 0', 'fontWeight': 'bold'})]),
                html.Div([html.Span('Return', style={'color': '#8b949e', 'fontSize': '11px'}),
                          html.P(f'{stats["total_return"]:+.2f}%',
                                 style={'color': pnl_color, 'margin': '2px 0'})]),
                html.Div([html.Span('Sharpe', style={'color': '#8b949e', 'fontSize': '11px'}),
                          html.P(f'{stats["sharpe"]}', style={'margin': '2px 0'})]),
                html.Div([html.Span('Max DD', style={'color': '#8b949e', 'fontSize': '11px'}),
                          html.P(f'${stats["max_dd"]:,.2f}',
                                 style={'color': '#e74c3c', 'margin': '2px 0'})]),
                html.Div([html.Span('Win Rate', style={'color': '#8b949e', 'fontSize': '11px'}),
                          html.P(f'{stats["win_rate"]}%', style={'margin': '2px 0'})]),
                html.Div([html.Span('Trades', style={'color': '#8b949e', 'fontSize': '11px'}),
                          html.P(f'{stats["n_trades"]}', style={'margin': '2px 0'})]),
            ]),
        ])
        cards[name] = card

        # ── Equity curve ────────────────────────────────────────────────────
        dates, eq = equity_curve(df, capital)
        if dates:
            # Append the live mark as one extra point so open exposure shows
            # up on the curve instead of the line stopping at the last close.
            plot_dates, plot_eq = list(dates), list(eq)
            if marks_as_of is not None:
                plot_dates.append(marks_as_of)
                plot_eq.append(eq[-1] + unrealized)

            eq_fig.add_trace(go.Scatter(
                x=plot_dates, y=plot_eq, name=name,
                line=dict(color=color, width=2),
                hovertemplate='%{x}<br>$%{y:,.2f}<extra>' + name + '</extra>'
            ))
            pnl_arr = pd.to_numeric(df['dollar_pnl'], errors='coerce').fillna(0).values
            cum     = np.cumsum(pnl_arr)
            if marks_as_of is not None:
                cum = np.append(cum, cum[-1] + unrealized)
            peak    = np.maximum.accumulate(cum)
            dd_arr  = cum - peak
            dd_fig.add_trace(go.Scatter(
                x=plot_dates, y=dd_arr.tolist(), name=name,
                fill='tozeroy', line=dict(color=color, width=1),
                hovertemplate='%{x}<br>$%{y:,.2f}<extra>' + name + '</extra>'
            ))

        if not df.empty:
            df['strategy'] = name
            all_trades.append(df)

    # ── Card rows (3 / 3 / 3) ────────────────────────────────────────────────
    row1 = [cards[s] for s in strategy_names[:3] if s in cards]
    row2 = [cards[s] for s in strategy_names[3:6] if s in cards]
    row3 = [cards[s] for s in strategy_names[6:] if s in cards]

    # ── Equity chart layout ──────────────────────────────────────────────────
    _dark_layout(eq_fig, yprefix='$')
    eq_fig.add_hline(y=10000, line_dash='dash', line_color='#8b949e',
                     annotation_text='Capital $10k')

    # ── Drawdown chart layout ────────────────────────────────────────────────
    _dark_layout(dd_fig, yprefix='$')

    # ── Combined card ────────────────────────────────────────────────────────
    combined_card = html.Div()
    total_unrealized = sum(all_unrealized.values())
    if all_trades:
        df_all       = pd.concat(all_trades, ignore_index=True)
        total_pnl    = pd.to_numeric(df_all.get('dollar_pnl', pd.Series()), errors='coerce').fillna(0).sum()
        total_trades = len(df_all)
        pnl_arr      = pd.to_numeric(df_all.get('dollar_pnl', pd.Series()), errors='coerce').dropna()
        win_rate     = round((pnl_arr > 0).mean() * 100, 1) if len(pnl_arr) > 0 else 0
        pnl_color    = '#2ecc71' if total_pnl >= 0 else '#e74c3c'
        unreal_color = '#2ecc71' if total_unrealized >= 0 else '#e74c3c'

        combined_card = html.Div(style={
            'backgroundColor': '#161b22', 'borderRadius': '10px', 'padding': '20px',
            'borderLeft': '4px solid #58a6ff', 'display': 'flex',
            'gap': '40px', 'flexWrap': 'wrap', 'marginBottom': '25px',
        }, children=[
            html.H4('📊 PORTFOLIO COMBINED',
                    style={'color': '#58a6ff', 'width': '100%', 'margin': '0 0 10px 0'}),
            html.Div([html.Span('Realized P&L',    style={'color': '#8b949e', 'fontSize': '12px'}),
                      html.H3(f'${total_pnl:+,.2f}', style={'color': pnl_color, 'margin': '2px 0'})]),
            html.Div([html.Span('Unrealized P&L',  style={'color': '#8b949e', 'fontSize': '12px'}),
                      html.H3(f'${total_unrealized:+,.2f}', style={'color': unreal_color, 'margin': '2px 0'})]),
            html.Div([html.Span('Total Trades', style={'color': '#8b949e', 'fontSize': '12px'}),
                      html.H3(str(total_trades),      style={'margin': '2px 0'})]),
            html.Div([html.Span('Win Rate',     style={'color': '#8b949e', 'fontSize': '12px'}),
                      html.H3(f'{win_rate}%',         style={'margin': '2px 0'})]),
            html.Div([html.Span('Strategies',   style={'color': '#8b949e', 'fontSize': '12px'}),
                      html.H3(f'{len(STRATEGIES)}',   style={'margin': '2px 0'})]),
        ])

    # ── Monthly heatmap ──────────────────────────────────────────────────────
    monthly_fig = go.Figure()
    if all_trades:
        df_all       = pd.concat(all_trades, ignore_index=True)
        df_all['year']  = pd.to_datetime(df_all['date']).dt.year
        df_all['month'] = pd.to_datetime(df_all['date']).dt.strftime('%b')
        pnl_col = pd.to_numeric(df_all.get('dollar_pnl', pd.Series(dtype=float)), errors='coerce').fillna(0)
        df_all['dollar_pnl_num'] = pnl_col
        pivot = df_all.groupby(['year', 'month'])['dollar_pnl_num'].sum().unstack(fill_value=0)
        month_order = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        cols  = [m for m in month_order if m in pivot.columns]
        pivot = pivot[cols]
        monthly_fig = go.Figure(go.Heatmap(
            z=pivot.values.tolist(),
            x=pivot.columns.tolist(),
            y=[str(y) for y in pivot.index.tolist()],
            colorscale='RdYlGn', zmid=0,
            text=[[f'${v:,.0f}' for v in row] for row in pivot.values],
            texttemplate='%{text}',
            hovertemplate='%{y} %{x}: $%{z:,.2f}<extra></extra>',
        ))
        _dark_layout(monthly_fig)

    # ── Strategy P&L bar chart ───────────────────────────────────────────────
    bar_fig = go.Figure()
    bar_names  = list(all_stats.keys())
    bar_values = [all_stats[s]['total_pnl'] for s in bar_names]
    bar_colors = [STRATEGIES[s]['color'] for s in bar_names]
    bar_fig.add_trace(go.Bar(
        x=bar_names, y=bar_values,
        marker_color=bar_colors,
        text=[f'${v:+,.0f}' for v in bar_values],
        textposition='outside',
        hovertemplate='%{x}<br>P&L: $%{y:,.2f}<extra></extra>',
    ))
    bar_fig.add_hline(y=0, line_color='#8b949e', line_width=1)
    _dark_layout(bar_fig, yprefix='$')
    bar_fig.update_layout(showlegend=False, bargap=0.35)

    return (
        f'Last updated: {now}',
        marks_text,
        row1,
        row2,
        row3,
        combined_card,
        eq_fig,
        dd_fig,
        monthly_fig,
        bar_fig,
    )


@app.callback(
    Output('trade-table', 'children'),
    Input('strategy-filter', 'value'),
    Input('interval',        'n_intervals'),
)
def update_table(strategy_filter, n):
    all_trades = []
    for name, cfg in STRATEGIES.items():
        filepath = os.path.join(DATA_DIR, cfg['file'])
        df       = load_trades(filepath)
        if not df.empty:
            df['strategy'] = name
            all_trades.append(df)

    if not all_trades:
        return html.P('No trades yet — run the strategies at 9:15pm SGT!',
                      style={'color': '#8b949e', 'textAlign': 'center', 'padding': '20px'})

    df_all = pd.concat(all_trades, ignore_index=True)

    if strategy_filter != 'ALL':
        df_all = df_all[df_all['strategy'] == strategy_filter]

    if df_all.empty:
        return html.P('No trades for selected strategy.', style={'color': '#8b949e'})

    # Base columns always shown — include status so we can style by it
    base_cols = ['date', 'strategy', 'symbol', 'direction', 'shares',
                 'est_entry', 'stop', 'tp', 'dollar_pnl', 'status']

    # Add strategy-specific signal columns when filtering to one strategy
    extra = []
    if strategy_filter != 'ALL' and strategy_filter in EXTRA_COLS:
        extra = EXTRA_COLS[strategy_filter]

    show_cols = base_cols + extra
    show_cols = [c for c in show_cols if c in df_all.columns]
    df_show   = df_all[show_cols].copy().sort_values('date', ascending=False)

    # Format for display
    for col, fmt in [('dollar_pnl', lambda x: f'${x:+,.2f}' if pd.notna(x) else '—'),
                     ('est_entry',  lambda x: f'${x:.2f}'   if pd.notna(x) else '—'),
                     ('stop',       lambda x: f'${x:.2f}'   if pd.notna(x) else '—'),
                     ('tp',         lambda x: f'${x:.2f}'   if pd.notna(x) else '—'),
                     ('z_score',    lambda x: f'{x:.3f}'    if pd.notna(x) else '—'),
                     ('eod_nope',   lambda x: f'{x:.6f}'    if pd.notna(x) else '—'),
                     ('reaction_pct', lambda x: f'{x:+.1f}%' if pd.notna(x) else '—'),
                     ('expected_pct', lambda x: f'{x:.1f}%'  if pd.notna(x) else '—')]:
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(fmt)

    df_show.columns = [c.replace('_', ' ').title() for c in df_show.columns]

    return dash_table.DataTable(
        data=df_show.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in df_show.columns],
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': '#21262d', 'color': '#58a6ff',
            'fontWeight': 'bold', 'border': '1px solid #30363d',
        },
        style_cell={
            'backgroundColor': '#0d1117', 'color': '#e6edf3',
            'border': '1px solid #21262d', 'fontSize': '13px',
            'padding': '8px', 'textAlign': 'center',
        },
        style_data_conditional=[
            # Closed / cancelled rows — dim everything to grey first
            {'if': {'filter_query': '{Status} = "closed" || {Status} = "cancelled" || {Status} = "closed_pending"'},
             'color': '#555e6b', 'backgroundColor': '#0d1117'},
            # Open rows — colour by direction
            {'if': {'filter_query': '{Status} = "open" && {Direction} = "LONG"'},  'color': '#2ecc71'},
            {'if': {'filter_query': '{Status} = "open" && {Direction} = "SHORT"'}, 'color': '#e74c3c'},
            # PnL colouring (takes precedence for the PnL cell on closed rows)
            {'if': {'filter_query': '{Dollar Pnl} contains "+"', 'column_id': 'Dollar Pnl'}, 'color': '#2ecc71'},
            {'if': {'filter_query': '{Dollar Pnl} contains "-"', 'column_id': 'Dollar Pnl'}, 'color': '#e74c3c'},
            # Highlight status badge
            {'if': {'filter_query': '{Status} = "open"',   'column_id': 'Status'}, 'color': '#f59e0b', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{Status} = "closed"', 'column_id': 'Status'}, 'color': '#555e6b'},
        ],
        page_size=25,
        sort_action='native',
        filter_action='native',
    )


# ── SHARED LAYOUT HELPER ──────────────────────────────────────────────────────
def _dark_layout(fig, yprefix=''):
    fig.update_layout(
        paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
        font=dict(color='#e6edf3'),
        legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
        xaxis=dict(gridcolor='#21262d'),
        yaxis=dict(gridcolor='#21262d',
                   tickprefix=yprefix if yprefix else ''),
        hovermode='x unified',
        margin=dict(l=55, r=20, t=20, b=40),
    )


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
