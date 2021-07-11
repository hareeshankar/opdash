import requests
import json
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
from datetime import datetime
import time
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from smartapi import SmartConnect #or from smartapi.smartConnect import SmartConnect
#import smartapi.smartExceptions(for smartExceptions)
external_stylesheets=['styles.css']
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
server = app.server

#SmartConnect
obj=SmartConnect(api_key="YZQSSF4H ")
#login api call
data = obj.generateSession("SPPA1005","QWEASD@01")
refreshToken= data['data']['refreshToken']
#print(refreshToken)
#fetch the feedtoken
feedToken=obj.getfeedToken()

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
payload={}
headers = {}
response = requests.request("GET", url, headers=headers, data=payload)
#print(response.text)
data = response.json()
df = pd.DataFrame(data)
#dfBN = pd.DataFrame(df[df['name']=="BANKNIFTY"])
#dfNIF = pd.DataFrame(df[df['name']=="NIFTY"])
dfinstru = df[df["exch_seg"]=="NFO"]
#print(df)
#Dash app code starts here
instru=dfinstru["name"].unique()
inst=[]
sT = np.arange((round(0.3*int(100))-(round(0.3*int(100))%100)),
                 round(1.7*int(100)),100)
bsop=[{'label':'Buy ','value':'BUY'},{'label':'Sell','value':'SELL'}]
cpop=[{'label':'CE ','value':'CE'},{'label':'PE','value':'PE'},{'label':'Stock','value':'Stock'}]
for x in instru:
    y={'label':x,'value':x}
    inst.append(y)
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(id='instru',options=inst,value='',placeholder='Select Index',style={'width':200,'marginBottom':10},searchable=True),
        html.Output(id='Spot',children='Spot Price:',style={'width':200,'padding': 10,'marginBottom':10}),
        dcc.Dropdown(id='CP',options=cpop,value='',placeholder='CE or PE or Stock',style={'width':200,'marginBottom':10}),
        html.Div([
            html.Div(id='exdd'),
            html.Div(id='STdd'),
            #dcc.Dropdown(id='CP',options=cpop,value='',placeholder='CE or PE',style={'width':200,'marginBottom':10}),
            html.Div(id='prelot',className='box1')
        ],id="wrapper1",style={'visibility':'hidden'}),
        html.Div([
            dcc.Input(id='Quantity',value='',type="number",placeholder='Quantity',style={'width':200,'padding': 10,'marginBottom':10}),
            dcc.Input(id='Price',value='',type="number",placeholder='Stock Price',style={'width':200,'padding': 10,'marginBottom':10}),
        ],className='box2',id="wrapper2",style={'visibility':'hidden'}),
        dcc.Dropdown(id='BS',options=bsop,value='',placeholder='Buy or Sell',style={'width':200,'marginBottom':10}),
        html.Button('Add Row', id='addleg', n_clicks=0,style={'width':200,'padding':10,'marginBottom':10}),
    ],className='box'),
    html.Div([
        dash_table.DataTable(
            id='strategy-table',
            columns=[{
                'name': '{}'.format(x),
                'id': 'id-{}'.format(x),
                'deletable': False,
                'renamable': False
            } for x in ['Buy or Sell','QTY','Strike','CE or PE / EQ',' Days to Expiry','Premium']],
            data=[],
            row_deletable=True,
            editable=True
        ),
        html.Div([
            html.P('Max Profit: ',className="alert alert-success radius",id='maxprof'),
            html.P('Max Loss: ',className="alert alert-error radius",id='maxloss'),
            html.P('Break Even: ',className="alert alert-info radius",id='brkeven')
        ],className='inforow'),
        dcc.Graph(id='payoff-graph')
    ],className='graph')
],className='container')

@app.callback(
    Output('exdd','children'),
    Input('instru','value')
)
def update_expiry(instru):
    exopc=[]
    if instru is not None:
        dfex =  df[df['name']==instru]
        dfex = dfex[dfex['exch_seg']!="NSE"]
        dfex["dttime"] = pd.to_datetime(dfex["expiry"])
        dfex = dfex.sort_values("dttime")
        dfex = dfex[dfex["dttime"] > datetime.now()]
        dfexU = dfex["expiry"].unique()
        for x in dfexU:
            y={'label':x,'value':x}
            exopc.append(y)
        app.logger.info(exopc)
    elem = dcc.Dropdown(id='expiry',options=exopc,value='',placeholder='Select Expiry',style={'width':200,'marginBottom':10},searchable=True)
    return elem

@app.callback(
   Output(component_id='wrapper2', component_property='style'),
   [Input(component_id='CP', component_property='value')])
def show_hide_element(cp):
    if cp == 'Stock':
        return {'visibility': 'visible'}
    else:
        return {'visibility':'hidden','order':5}
@app.callback(
   Output(component_id='wrapper1', component_property='style'),
   [Input(component_id='CP', component_property='value')])
def show_hide_element(cp):
    if ( cp == 'CE' or cp == 'PE'):
        return {'visibility': 'visible'}
    else:
        return {'visibility':'hidden','order':5}

@app.callback(
Output('Spot','children'),
Input('instru','value')
)
def update_spot(instru):
    app.logger.info(instru)
    if instru is not None:
        dfspot = df[df['name']==instru]
        dfspot = dfspot[dfspot['exch_seg']=="NSE"]
        if instru != "NIFTY" and instru != "BANKNIFTY":
            ltpsymbol = instru + "-EQ"
        else:
            ltpsymbol = instru
        ltptokendf = dfspot[dfspot["symbol"]==ltpsymbol]
        ltptokendf = ltptokendf["token"]
        ltptoken = ltptokendf.values[0]
        response = obj.ltpData("NSE",ltpsymbol,ltptoken)
        ltp = response["data"]["ltp"]
        return ("Spot Price : " + str(ltp))
    else:
        return "Spot Price : "

@app.callback(
    Output('STdd','children'),
    Input('expiry','value'),
    State('instru','value'),prevent_initial_call=True
)
def update_strike(expiry,instru):
    stopc=[]
    if instru and expiry is not None:
        dfst =  df[df['name']==instru]
        dfst = dfst[dfst['expiry']==expiry]
        dfst = dfst[dfst['exch_seg']!="NSE"]
        dfst["dttime"] = pd.to_datetime(dfst["expiry"])
        dfst = dfst.sort_values("dttime")
        dfst = dfst[dfst["dttime"] > datetime.now()]
        dfst["strike"] = dfst['strike'].astype(float)
        dfst = dfst.sort_values("strike")
        dfst["newstr"] = dfst["strike"]/100
        dfst = dfst[dfst["newstr"]!=-0.01]
        dfstU = dfst["newstr"].unique()
        for x in dfstU:
            y={'label':x,'value':x}
            stopc.append(y)
        app.logger.info(stopc)
    elem = dcc.Dropdown(id='strike',options=stopc,value='',placeholder='Select Strike',style={'width':200,'marginBottom':10},searchable=True)
    return elem

@app.callback(
    Output('prelot','children'),
    Input('CP','value'),
    Input('strike','value'),State('instru','value'),State('expiry','value')
    )
def update_prelot(CP, strike, instru, expiry):
    if (CP == 'CE' or CP == 'PE'):
        if ( strike != ""):
            if ( expiry != ""):
                if (instru != ""):
                    x = expiry[0:5]
                    y = expiry[5:9]
                    yint = int(y)
                    yint = yint-2000
                    ystr = str(yint)
                    exmod = x + ystr
                    symbol = instru + exmod + str(strike) + CP
                    app.logger.info(symbol)
                    dflot = df[df["symbol"]==symbol]
                    lotsz = dflot["lotsize"]
                    lotsizevar = lotsz.values[0]
                    lotdd = []
                    lotddop = []
                    for i in range (1,100,1):
                        lotdd.append(i*int(lotsizevar))
                    app.logger.info(lotdd)
                    for x in lotdd:
                        y={'label':x,'value':x}
                        lotddop.append(y)
                    #app.logger.info(lotddop)
                    lotszstr = "Lot Size : " + str(lotsizevar)
                    app.logger.info(lotsizevar)
                    tokendf = dflot["token"]
                    token = tokendf.values[0]
                    response = obj.ltpData("NFO",symbol,token)
                    ltp = response["data"]["ltp"]
                    premstr = "Premium : " + str(ltp)
                    app.logger.info(premstr)
                    layout = html.Div([
                                html.Output(id='Premium',children=premstr,style={'width':200,'padding': 10}),html.Br(),
                                #html.Label(id='Quantity',children='Quantity',style={'width':200,'padding': 10,'marginBottom':10}),
                                dcc.Dropdown(id='lotsize',options=lotddop,placeholder="Lot Size",value='',
                                style={'width':200,'marginBottom':10})
                    ],className='box1')
                    return layout
    if (CP == 'Stock'):
        layout = html.Div([
                    html.Output(id='Premium',children='Premium : ',style={'width':200,'padding': 10,'marginBottom':10,'visibility':'hidden'}),
                    dcc.Dropdown(id='lotsize',options=[],placeholder="Lot Size",value='',disabled=True,
                    style={'width':200,'marginBottom':10,'visibility':'hidden'}),
        ],className='box1')
        return layout

@app.callback(
    Output('strategy-table', 'data'),
    Input('addleg', 'n_clicks'),
    State('strategy-table', 'data'),
    State('strategy-table','columns'),
    State('BS','value'),State('expiry','value'),State('strike','value'),State('CP','value'),
    State('Spot','children'),State('Premium','children'),State('lotsize','value'),
    State('Quantity','value'),State('Price','value')
    )
def add_row(n_clicks, rows, columns, BS, EX, ST, CP, Spot,Pre,lots,QTY,price):
    if ( n_clicks > 0  ):
            pres = Pre[10:len(Pre)]
            spt = Spot[13:len(Spot)]
            sptfloat = float(spt)
            if( CP =='Stock'):
                ST = sptfloat
                row = [ BS,QTY,ST,CP,EX,price]
            else:
                row = [ BS,lots,ST,CP,EX,pres]
            cols=[]
            for c in columns:
                cols.append(c['id'])
            app.logger.info(row)
            app.logger.info(cols)
            dict1={}
            if float(row[5]) != 0.0:
                for i in range(len(row)):
                    #data1.append({cols[i]:row[i]})
                    dict1[cols[i]]=row[i]
                app.logger.info([dict1])
            else:
                return None

            #data1=pd.DataFrame(dict1)
            #data1.append({c['id']: row[i] for i, c in zip(range(len(row)),columns)},ignore_index=True)
            if rows is None:
                rows=[dict1]
            else:
                rows.append(dict1)

            return rows

@app.callback(
    Output('payoff-graph', 'figure'),
    Output('maxprof','children'),
    Output('maxloss','children'),
    Output('brkeven','children'),
    Input('strategy-table', 'data'),
    Input('strategy-table', 'columns'),
    State('Spot','children')
    )
def display_output(rows, columns,Spot):
    spt = Spot[13:len(Spot)]
    sptfloat = float(spt)
    lists = [[row.get(c['id'], None) for c in columns] for row in rows]
    sT = np.arange((round(0.3*sptfloat)-(round(0.3*sptfloat)%100)),
                     round(1.7*sptfloat),.1)
    newsT = np.arange((round(0.3*sptfloat)-(round(0.3*sptfloat)%100)),
                     round(1.7*sptfloat),.1)
    #global PAYOFF
    PAYOFF = np.where(sT > 0, 0, 0)
    newPF = np.where(newsT > 0, 0,0)
    mprof = 'Max Profit: '
    mloss = 'Max Loss: '
    brke = 'Break Even: '
    dfpayf=pd.DataFrame()
    dfpfrnd=pd.DataFrame()
    dfpffinal=pd.DataFrame()
    app.logger.info('lists')
    app.logger.info(lists)
    for x in lists:
        app.logger.info(x)
        pf = np.where(sT > 0, 0, 0)
        if (x[5] != ""):
            print("entered if for option leg")
            BorS = x[0]
            CorP = x[3]
            QTY = int(x[1])
            STR = int(x[2])
            EXP = x[4]
            PRE = float(x[5])
            if CorP == "CE" and BorS == "BUY":
                pf = (np.where(sT > STR, sT - STR, 0) - PRE) * QTY
            if CorP == "CE" and BorS == "SELL":
                pf = (np.where(sT > STR , STR - sT, 0) + PRE) * QTY
            if CorP == "PE" and BorS == "BUY":
                pf = (np.where(sT < STR, STR - sT, 0) - PRE) * QTY
            if CorP == "PE" and BorS == "SELL":
                pf = (np.where(sT < STR, sT - STR, 0) + PRE) * QTY
            if CorP == "Stock" and BorS == "BUY":
                pf = (np.where(True,sT-PRE,0)) * QTY
            if CorP == "Stock" and BorS == "SELL":
                pf = (np.where(True,PRE-sT,0)) * QTY
        PAYOFF = PAYOFF + pf
        ## Below newly added after last deployment in heroku ## Latest work is ocuiv3
    #if PAYOFF.size:
    #    app.logger.info(PAYOFF)

    ######################## Break Even Calculation ####
    dfpppp = pd.DataFrame()
    pfsign = np.sign(PAYOFF)
    pfsignchange = ((np.roll(pfsign, 1) - pfsign) != 0).astype(int)
    pfsignchange[0]=0
    dfpppp['pf'] = PAYOFF.tolist()
    dfpppp['sign'] = pfsignchange.tolist()
    dfpppp.to_csv('dfpppp.csv')
    indx = []
    indxprev = []
    indxcnt = 0
    for x in pfsignchange:
        if x == 1:
            indx.append(indxcnt)
            indxprev.append(indxcnt-1)
        indxcnt = indxcnt + 1
    app.logger.info('indx and indxprev')
    app.logger.info(indx)
    app.logger.info(indxprev)
    dfindxxx = pd.DataFrame()
    dfindxxx['ind']=indx
    dfindxxx['indp']=indxprev
    for x in indx:
        stavg = (sT[x]+sT[x-1])/2
        newsT = np.insert(sT,x,stavg)
        newPF = np.insert(PAYOFF,x,0)
    dfpayf['sT'] = newsT.tolist()
    dfpayf['pf'] = newPF.tolist()
    dfpfrnd['sT'] = sT.tolist()
    pfrnd = np.round(PAYOFF,2)
    dfpfrnd['pf'] =pfrnd.tolist()
    dfpayf.to_csv('dfpayf.csv')
    #dfind = dfpayf.index[(dfpayf['pf'] > -1) & (dfpayf['pf'] < 1)].tolist()
    dfind = dfpfrnd.index[(dfpfrnd['pf'] == 0)].tolist()
    if len(dfind) > 0:
        for x in dfind:
            bflt =dfpfrnd['sT'].iloc[x]
            brkstr = str("{:.2f}".format(bflt))
            brke = brke + brkstr + " "
        app.logger.info('brke ' + brke)
        dfpffinal=dfpfrnd
    else:
        dfind = dfpayf.index[(dfpayf['pf'] == 0)].tolist()
        for x in dfind:
            bflt =dfpayf['sT'].iloc[x]
            brkstr = str("{:.2f}".format(bflt))
            brke = brke + brkstr + " "
        app.logger.info('brke ' + brke)
        dfpffinal=dfpayf

    ######################## Max prof Max Loss Calculation ####
    maxpf = dfpffinal['pf'].max()
    minpf = dfpffinal['pf'].min()
    rymin = 0
    rymax = 0
    pflen = len(dfpffinal['pf'])
    dfmxind = dfpffinal.index[(dfpffinal['pf'] == maxpf)].tolist()
    dfmnind = dfpffinal.index[(dfpffinal['pf'] == minpf)].tolist()
    if len(dfmxind) > 1:
        mprof = mprof + str("{:.2f}".format(maxpf))
        rymax = maxpf*2
    else:
        if (dfpffinal['pf'].iloc[0]==maxpf or dfpffinal['pf'].iloc[pflen-1]==maxpf):
            mprof = mprof + "Unlimited"
            rymax = maxpf * .25
        else:
            mprof = mprof + str("{:.2f}".format(maxpf))
            rymax = maxpf * 2
    if len(dfmnind) > 1:
        mloss = mloss + str("{:.2f}".format(minpf))
        rymin = minpf*2
    else:
        if (dfpffinal['pf'].iloc[0]==minpf or dfpffinal['pf'].iloc[pflen-1]==minpf):
            mloss = mloss + "Unlimited"
            rymin = minpf * .25
        else:
            mloss = mloss + str("{:.2f}".format(minpf))
            rymin = minpf * 2
    ############################ Create Figure object ####
    if len(dfind) > 0:
        #fig = px.line(
        #x=sT, y=pfrnd,
        #title="Strategy Payoff",template="none"
        #)
        #fig.update_yaxes(zerolinecolor="#123")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sT, y=pfrnd,
                    mode='lines',
                    name='Expiry Payoff',
                    line=dict(color='crimson', width=2)
                    ))
        fig.update_layout(title='Strategy Payoff',
                   xaxis_title='Underlying Price',
                   yaxis_title='Profit / Loss',
                   template='none',
                   autosize=False,
                   margin=dict(
                        autoexpand=False,
                        l=50,
                        r=20,
                        t=50,
                    )
                   )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        fig.update_xaxes(range=[sptfloat*.5, sptfloat*1.5])
        fig.update_yaxes(range=[rymin,rymax])
        app.logger.info("rymin + rymax ")
        app.logger.info(rymin)
        app.logger.info(rymax)
    else:
        fig = px.line(
        x=newsT, y=newPF,
        title="Strategy Payoff",template="none"
        )
        fig.update_yaxes(zerolinecolor="#123")

    if len(lists) == 0:
        fig : []
        mprof = 'Max Profit: '
        mloss = 'Max Loss: '
        brke = 'Break Even: '

    return fig, mprof, mloss, brke
    #{
        #'data': [{
        #    'type': 'line',
        #    'y': PAYOFF,
        #    'x': sT
        #}]
    #}

if __name__ == '__main__':
    app.run_server(debug=True)
