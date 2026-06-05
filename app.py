"""
AlphaX Paper Trade Dashboard
Displays live P&L, equity curves, drawdown and trade logs
for AEARN-MOMO-001, CNP-DKPL-V2, ODFL-NETPREM-001
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

STRATEGIES = {
    'AEARN-MOMO-001' : {'file': 'aearn_trades.csv',  'color': '#2ecc71', 'capital': 10000},
    'CNP-DKPL-V2'    : {'file': 'cnp_trades.csv',    'color': '#3498db', 'capital': 10000},
    'ODFL-NETPREM-001': {'file': 'odfl_trades.csv',  'color': '#e67e22', 'capital': 10000},
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_trades(filepath):
    """Load trade CSV safely."""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()

def compute_stats(df, capital=10000):
    """Compute key performance stats from trade log."""
    if df.empty or 'dollar_pnl' not in df.columns:
        return {
            'total_pnl'   : 0, 'total_return': 0,
            'sharpe'      : 0, 'max_dd'      : 0,
            'win_rate'    : 0, 'n_trades'    : 0,
            'profit_factor': 0,
        }
    pnl   = df['dollar_pnl'].values
    cum   = np.cumsum(pnl)
    peak  = np.maximum.accumulate(cum)
    dd    = (cum - peak)
    wins  = pnl[pnl > 0]
    loss  = pnl[pnl < 0]
    std   = pnl.std(ddof=1) if len(pnl) > 1 else 1
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
    """Return dates and equity values."""
    if df.empty or 'dollar_pnl' not in df.columns:
        return [], []
    dates  = pd.to_datetime(df['date'])
    equity = capital + np.cumsum(df['dollar_pnl'].values)
    return dates.tolist(), equity.tolist()

def monthly_pnl(df):
    """Return monthly P&L pivot."""
    if df.empty or 'dollar_pnl' not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df['year']  = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%b')
    pivot = df.groupby(['year','month'])['dollar_pnl'].sum().unstack(fill_value=0)
    month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    cols = [m for m in month_order if m in pivot.columns]
    return pivot[cols]

# ── APP ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title='AlphaX Dashboard')
server = app.server  # for Render/gunicorn

# ── LAYOUT ────────────────────────────────────────────────────────────────────
app.layout = html.Div(style={'backgroundColor':'#0d1117','minHeight':'100vh','fontFamily':'Arial','color':'#e6edf3','padding':'20px'}, children=[

    # Header
    html.Div(style={'textAlign':'center','marginBottom':'30px'}, children=[
        html.H1('🤖 AlphaX Paper Trade Dashboard', style={'color':'#58a6ff','marginBottom':'5px'}),
        html.P(id='last-updated', style={'color':'#8b949e','fontSize':'13px'}),
    ]),

    # Strategy Cards
    html.Div(id='strategy-cards', style={'display':'flex','gap':'15px','marginBottom':'25px','flexWrap':'wrap'}),

    # Combined Card
    html.Div(id='combined-card', style={'marginBottom':'25px'}),

    # Equity Curve
    html.Div([
        html.H3('📈 Equity Curves', style={'color':'#58a6ff','marginBottom':'10px'}),
        dcc.Graph(id='equity-chart', style={'height':'400px'}),
    ], style={'backgroundColor':'#161b22','borderRadius':'10px','padding':'20px','marginBottom':'25px'}),

    # Drawdown Chart
    html.Div([
        html.H3('📉 Drawdown', style={'color':'#58a6ff','marginBottom':'10px'}),
        dcc.Graph(id='drawdown-chart', style={'height':'250px'}),
    ], style={'backgroundColor':'#161b22','borderRadius':'10px','padding':'20px','marginBottom':'25px'}),

    # Monthly Heatmaps
    html.Div([
        html.H3('📊 Monthly P&L ($)', style={'color':'#58a6ff','marginBottom':'10px'}),
        dcc.Graph(id='monthly-chart', style={'height':'350px'}),
    ], style={'backgroundColor':'#161b22','borderRadius':'10px','padding':'20px','marginBottom':'25px'}),

    # Trade Log Table
    html.Div([
        html.H3('📋 Trade Log', style={'color':'#58a6ff','marginBottom':'10px'}),
        html.Div([
            dcc.Dropdown(
                id='strategy-filter',
                options=[{'label':'All Strategies','value':'ALL'}] +
                        [{'label':s,'value':s} for s in STRATEGIES],
                value='ALL',
                style={'backgroundColor':'#21262d','color':'#000','width':'300px','marginBottom':'10px'}
            ),
        ]),
        html.Div(id='trade-table'),
    ], style={'backgroundColor':'#161b22','borderRadius':'10px','padding':'20px','marginBottom':'25px'}),

    # Auto refresh every 5 min
    dcc.Interval(id='interval', interval=5*60*1000, n_intervals=0),
])

# ── CALLBACKS ─────────────────────────────────────────────────────────────────
@app.callback(
    Output('last-updated',   'children'),
    Output('strategy-cards', 'children'),
    Output('combined-card',  'children'),
    Output('equity-chart',   'figure'),
    Output('drawdown-chart', 'figure'),
    Output('monthly-chart',  'figure'),
    Input('interval',        'n_intervals'),
)
def update_dashboard(n):
    now       = datetime.now().strftime('%Y-%m-%d %H:%M SGT')
    all_trades = []
    cards      = []
    eq_fig     = go.Figure()
    dd_fig     = go.Figure()

    for name, cfg in STRATEGIES.items():
        filepath = os.path.join(DATA_DIR, cfg['file'])
        df       = load_trades(filepath)
        stats    = compute_stats(df, cfg['capital'])
        color    = cfg['color']
        capital  = cfg['capital']

        # Strategy card
        pnl_color = '#2ecc71' if stats['total_pnl'] >= 0 else '#e74c3c'
        cards.append(html.Div(style={
            'backgroundColor':'#161b22','borderRadius':'10px','padding':'20px',
            'flex':'1','minWidth':'220px','borderLeft':f'4px solid {color}'
        }, children=[
            html.H4(name, style={'color':color,'marginBottom':'10px','fontSize':'14px'}),
            html.Div([
                html.Div([html.Span('P&L', style={'color':'#8b949e','fontSize':'12px'}),
                          html.H3(f'${stats["total_pnl"]:+,.2f}', style={'color':pnl_color,'margin':'2px 0'})]),
                html.Div([html.Span('Return', style={'color':'#8b949e','fontSize':'12px'}),
                          html.P(f'{stats["total_return"]:+.2f}%', style={'color':pnl_color,'margin':'2px 0'})]),
                html.Div([html.Span('Sharpe', style={'color':'#8b949e','fontSize':'12px'}),
                          html.P(f'{stats["sharpe"]}', style={'margin':'2px 0'})]),
                html.Div([html.Span('Max DD', style={'color':'#8b949e','fontSize':'12px'}),
                          html.P(f'${stats["max_dd"]:,.2f}', style={'color':'#e74c3c','margin':'2px 0'})]),
                html.Div([html.Span('Win Rate', style={'color':'#8b949e','fontSize':'12px'}),
                          html.P(f'{stats["win_rate"]}%', style={'margin':'2px 0'})]),
                html.Div([html.Span('Trades', style={'color':'#8b949e','fontSize':'12px'}),
                          html.P(f'{stats["n_trades"]}', style={'margin':'2px 0'})]),
            ], style={'display':'grid','gridTemplateColumns':'1fr 1fr','gap':'8px'}),
        ]))

        # Equity curve
        dates, eq = equity_curve(df, capital)
        if dates:
            eq_fig.add_trace(go.Scatter(
                x=dates, y=eq, name=name, line=dict(color=color, width=2),
                hovertemplate='%{x}<br>$%{y:,.2f}<extra>' + name + '</extra>'
            ))
            # Drawdown
            pnl_arr = df['dollar_pnl'].values
            cum     = np.cumsum(pnl_arr)
            peak    = np.maximum.accumulate(cum)
            dd      = cum - peak
            dd_fig.add_trace(go.Scatter(
                x=dates, y=dd.tolist(), name=name,
                fill='tozeroy', line=dict(color=color, width=1),
                hovertemplate='%{x}<br>$%{y:,.2f}<extra>' + name + '</extra>'
            ))

        # Collect all trades
        if not df.empty:
            df['strategy'] = name
            all_trades.append(df)

    # Equity chart layout
    eq_fig.update_layout(
        paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
        font=dict(color='#e6edf3'), legend=dict(bgcolor='#161b22'),
        xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d', tickprefix='$'),
        hovermode='x unified', margin=dict(l=50,r=20,t=20,b=40)
    )
    eq_fig.add_hline(y=10000, line_dash='dash', line_color='#8b949e', annotation_text='Capital $10,000')

    # Drawdown chart layout
    dd_fig.update_layout(
        paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
        font=dict(color='#e6edf3'), legend=dict(bgcolor='#161b22'),
        xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d', tickprefix='$'),
        hovermode='x unified', margin=dict(l=50,r=20,t=20,b=40)
    )

    # Combined stats card
    combined_card = html.Div()
    if all_trades:
        df_all       = pd.concat(all_trades, ignore_index=True)
        total_pnl    = df_all['dollar_pnl'].sum()
        total_trades = len(df_all)
        win_rate     = round((df_all['dollar_pnl'] > 0).mean() * 100, 1)
        pnl_color    = '#2ecc71' if total_pnl >= 0 else '#e74c3c'
        combined_card = html.Div(style={
            'backgroundColor':'#161b22','borderRadius':'10px','padding':'20px',
            'borderLeft':'4px solid #58a6ff','display':'flex','gap':'40px','flexWrap':'wrap'
        }, children=[
            html.H4('📊 COMBINED', style={'color':'#58a6ff','width':'100%','margin':'0 0 10px 0'}),
            html.Div([html.Span('Total P&L', style={'color':'#8b949e','fontSize':'12px'}),
                      html.H3(f'${total_pnl:+,.2f}', style={'color':pnl_color,'margin':'2px 0'})]),
            html.Div([html.Span('Total Trades', style={'color':'#8b949e','fontSize':'12px'}),
                      html.H3(f'{total_trades}', style={'margin':'2px 0'})]),
            html.Div([html.Span('Combined Win Rate', style={'color':'#8b949e','fontSize':'12px'}),
                      html.H3(f'{win_rate}%', style={'margin':'2px 0'})]),
        ])

    # Monthly heatmap (combined)
    monthly_fig = go.Figure()
    if all_trades:
        df_all = pd.concat(all_trades, ignore_index=True)
        df_all['year']  = pd.to_datetime(df_all['date']).dt.year
        df_all['month'] = pd.to_datetime(df_all['date']).dt.strftime('%b')
        pivot = df_all.groupby(['year','month'])['dollar_pnl'].sum().unstack(fill_value=0)
        month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        cols  = [m for m in month_order if m in pivot.columns]
        pivot = pivot[cols]
        monthly_fig = go.Figure(go.Heatmap(
            z=pivot.values.tolist(),
            x=pivot.columns.tolist(),
            y=[str(y) for y in pivot.index.tolist()],
            colorscale='RdYlGn', zmid=0,
            text=[[f'${v:,.0f}' for v in row] for row in pivot.values],
            texttemplate='%{text}',
            hovertemplate='%{y} %{x}: $%{z:,.2f}<extra></extra>'
        ))
        monthly_fig.update_layout(
            paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
            font=dict(color='#e6edf3'),
            margin=dict(l=60,r=20,t=20,b=40)
        )

    return (
        f'Last updated: {now}',
        cards,
        combined_card,
        eq_fig,
        dd_fig,
        monthly_fig,
    )

@app.callback(
    Output('trade-table', 'children'),
    Input('strategy-filter', 'value'),
    Input('interval', 'n_intervals'),
)
def update_table(strategy_filter, n):
    all_trades = []
    for name, cfg in STRATEGIES.items():
        filepath = os.path.join(DATA_DIR, cfg['file'])
        df = load_trades(filepath)
        if not df.empty:
            df['strategy'] = name
            all_trades.append(df)

    if not all_trades:
        return html.P('No trades yet — run the strategies tonight!',
                      style={'color':'#8b949e','textAlign':'center','padding':'20px'})

    df_all = pd.concat(all_trades, ignore_index=True)

    if strategy_filter != 'ALL':
        df_all = df_all[df_all['strategy'] == strategy_filter]

    if df_all.empty:
        return html.P('No trades for selected strategy.', style={'color':'#8b949e'})

    # Format columns for display
    show_cols = ['date','strategy','symbol','direction','shares',
                 'est_entry','stop','dollar_pnl']
    show_cols = [c for c in show_cols if c in df_all.columns]
    df_show   = df_all[show_cols].copy().sort_values('date', ascending=False)

    # Format numbers
    if 'dollar_pnl' in df_show.columns:
        df_show['dollar_pnl'] = df_show['dollar_pnl'].apply(lambda x: f'${x:+,.2f}')
    if 'est_entry' in df_show.columns:
        df_show['est_entry'] = df_show['est_entry'].apply(lambda x: f'${x:.2f}')
    if 'stop' in df_show.columns:
        df_show['stop'] = df_show['stop'].apply(lambda x: f'${x:.2f}')

    df_show.columns = [c.replace('_',' ').title() for c in df_show.columns]

    return dash_table.DataTable(
        data=df_show.to_dict('records'),
        columns=[{'name':c,'id':c} for c in df_show.columns],
        style_table={'overflowX':'auto'},
        style_header={'backgroundColor':'#21262d','color':'#58a6ff','fontWeight':'bold','border':'1px solid #30363d'},
        style_cell={'backgroundColor':'#0d1117','color':'#e6edf3','border':'1px solid #21262d',
                    'fontSize':'13px','padding':'8px','textAlign':'center'},
        style_data_conditional=[
            {'if':{'filter_query':'{Dollar Pnl} contains "+"'},'color':'#2ecc71'},
            {'if':{'filter_query':'{Dollar Pnl} contains "-"'},'color':'#e74c3c'},
        ],
        page_size=20,
        sort_action='native',
        filter_action='native',
    )

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
