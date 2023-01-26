#!/usr/bin/env python
# coding: utf-8

# # Annual emissions broken down by sector
# 
# Here we can present the visualisations and discuss them.
# 
# This page can be easily printed to a pdf using the buttons on the top right hand side of this web page.
# 
# This page is just a markdown file that has been slightly adapted to look nicer in Jupyter Books - e.g. removing the cell inputs/ outputs to only show the interactive graph below.

# In[1]:


import pandas as pd
from itertools import takewhile
import plotly.graph_objects as go
import base64
import numpy as np
import glob

from fig_mods.getdata import load_ch4
from fig_mods.getdata import load_n2o
from fig_mods.getdata import load_fgas

from fig_mods import path
from fig_mods.nextstep import read_invent_ODS
from fig_mods.nextstep import read_invent

import plotly.express as px
import base64
import xarray as xr
import matplotlib.pyplot as plt
from numpy import loadtxt

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from fig_mods.nextstep import areagrid


# In[2]:


dfCH4 = (read_invent("GBR", "2022", "Table10s3"))
dfN2O = (read_invent("GBR", "2022", "Table10s4"))


# In[3]:


invent_fgas_total = pd.read_csv('data/UK_NIR_2022_co2e_all_gases_yearly.csv', index_col="year")
invent_fgas_total.index = pd.to_datetime(invent_fgas_total.index, format='%Y')

fgases = ["HFC-23", "HFC-125", "HFC-134a", "HFC-143a", "HFC-152a", "HFC-227ea", "HFC-245fa", "HFC-32", "HFC-365mfc", "HFC-43-10mee"]
dfODS = pd.DataFrame()

for name in (fgases):
    a = read_invent_ODS("2022", name)
    b = a.rename(columns={"F.  Product uses as substitutes for ODS(2)": name})
    
    dfODS = pd.concat([dfODS, b], axis=1)
    
dfODS["ODS"] = dfODS.sum(axis=1) / 1000
dfODS["Total"] = invent_fgas_total.sum(axis=1) / 1000


# In[4]:


def read_intem(species):

    fname = path() / f"data/intem/Table_UK_2021_{species}.txt"

    with open(fname, "r") as f:
        headiter = takewhile(lambda s: s.startswith('#'), f)
        header = list(headiter)

    df = pd.read_csv(fname,
                    sep=",",
                    skipinitialspace=True,
                    index_col="Year",
                    comment="#")

    df = df.apply(pd.to_numeric, errors='ignore')
    df.index = pd.to_datetime(df.index, format='%Y')
    df.rename(columns = {"Invent2021": "Invent"}, inplace=True)

    for h in header:
        if "GAS" in h:
            species_str = h.split(":")[1].strip()
        if "UNITS" in h:
            species_units = h.split(":")[1].strip()
        if "GWP" in h:
            species_gwp = float(h.split(":")[1].strip())
    
    return df, species_str, species_units, species_gwp


# In[5]:


df_ch4, _, _, _ = read_intem("ch4")
df_n2o, _, _, _ = read_intem("n2o")


# In[6]:


# Get F-gases
fgas = [f"hfc{hfc}" for hfc in ["23", "32", "125", "134a", "143a", "152a", "227ea", "245fa", "365mfc", "4310mee"]] +     [f"pfc{pfc}" for pfc in ["14", "116", "218", "318"]] +     ["sf6"]

df_fgas, _, _, _ = read_intem(fgas[0])

uncert = df_fgas["InTEM_std"]**2

for fg in fgas[1:]:
    dfs, _, _, _ = read_intem(fg)
    df_fgas += dfs
    uncert += df_fgas["InTEM_std"]**2

df_fgas["InTEM_std"] = np.sqrt(uncert.values)


# In[7]:


df_fgas["InTEM_ODS"] = (dfODS["ODS"] / dfODS["Total"]) * df_fgas["InTEM"]
df_fgas["InTEM_ODS_std"] = (dfODS["ODS"] / dfODS["Total"]) * df_fgas["InTEM_std"]


# In[8]:


alpha = 0.6
color_ch4 = ["rgba(86, 119, 194, 1)", f"rgba(86, 119, 194, {alpha})"]
color_n2o = ["rgba(99, 182, 137, 1)", f"rgba(99, 182, 137, {alpha})"]
color_fgas = ["rgba(187, 103, 93, 1)", f"rgba(187, 103, 93, {alpha})"]

fig = go.Figure()

mo_logo = base64.b64encode(open("metoffice_logo.png", 'rb').read())
uob_logo = base64.b64encode(open("uob_logo.png", 'rb').read())

def errorbars(df, var, color, dash, name, showlegend=True):

    error_minus=df[var] - df[f"{var}_std"]
    error_plus=df[var] + df[f"{var}_std"]

    fig.add_trace(go.Scatter(
        x=df.index,
        y=error_minus.values,
        fill=None,
        mode='lines',
        line=dict(color=color[1], width=0.1),
        showlegend=False,
        hoverinfo='skip'
        ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=error_plus.values,
        fill="tonexty",
        mode='lines',
        line=dict(color=color[1], width=0.1),
        showlegend=False,
        hoverinfo='skip'
        ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[var],
        mode="lines",
        line=dict(color=color[0], dash=dash),
        showlegend=showlegend,
        name=name,
        hovertemplate = 'InTEM %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
        ))


# errorbars(df_ch4.loc[:"2011-01-01"], "InTEM", color_ch4, None, "CH<sub>4</sub>")
# errorbars(df_ch4.loc["2012-01-01":], "InTEM", color_ch4, None, "CH<sub>4</sub>", showlegend=False)
errorbars(df_ch4, "InTEM", color_ch4, None, "Methane")
fig.add_trace(go.Scatter(
    x=df_ch4.index,
    y=df_ch4["Invent"],
    line=dict(color=color_ch4[0], dash="dot", width=5),
    showlegend=False,
    hovertemplate = 'Inventory %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    ))

errorbars(df_n2o, "InTEM", color_n2o, None, "Nitrous oxide")
# errorbars(df_n2o[:"2011-01-01"], "InTEM", color_n2o, None, "N<sub>2</sub>O")
# errorbars(df_n2o["2012-01-01":], "InTEM", color_n2o, None, "N<sub>2</sub>O", showlegend=False)
fig.add_trace(go.Scatter(
    x=df_n2o.index,
    y=df_n2o["Invent"],
    line=dict(color=color_n2o[0], dash="dot", width=4),
    showlegend=False,
    hovertemplate = 'Inventory %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    #name="N<sub>2</sub>O"
    ))

errorbars(df_fgas["2012-01-01":], "InTEM", color_fgas, None, "F-gases")
fig.add_trace(go.Scatter(
    x=df_fgas.index,
    y=df_fgas["Invent"],
    line=dict(color=color_fgas[0], dash="dot", width=4),
    showlegend=False,
    hovertemplate = 'Inventory %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    #name="F-gases"
    ))

# Add fake line for InTEM
fig.add_trace(go.Scatter(
    x=[pd.Timestamp("1900-01-01"), pd.Timestamp("1900-01-02")],
    y=[0., 0.],
    line=dict(color="black", dash=None),
    marker=dict(opacity=0., size=0),
    showlegend=True,
    name="Atmospheric data"
    ))

# Add fake line for inventory legend
fig.add_trace(go.Scatter(
    x=[pd.Timestamp("1900-01-01"), pd.Timestamp("1900-01-02")],
    y=[0., 0.],
    line=dict(color="black", dash="dot", width=4),
    marker=dict(opacity=0., size=0),
    showlegend=True,
    name="Inventory (2021)"
    ))

fig.update_layout(
    # title={"text": "UK non-CO<sub>2</sub> reenhouse gas emissions 2010 - 2020",
    #         "xanchor": "left",
    #         "x": 0.01},
    yaxis_title="CO<sub>2</sub>-equivalent emissions (Gt yr⁻¹)",
    template="none",
    autosize=False,
    width=1000,
    height=600,
    legend=dict(
        yanchor="top",
        y=0.77,
        xanchor="right",
        x=0.99,
        traceorder="normal"),
    margin=dict(l=100, r=50, t=50, b=50),
    # paper_bgcolor='rgba(0,0,0,0)',
    # plot_bgcolor='rgba(0,0,0,0)'
)

fig.layout.font.size=20
fig.layout.font.family="Arial"

fig.update_xaxes(range=[pd.Timestamp("1990-01-01"),
                        pd.Timestamp("2020-01-01")])

fig.update_yaxes(range=[0,
                        153])

# Add UKMO logo
fig.add_layout_image(
    dict(
        source='data:image/png;base64,{}'.format(mo_logo.decode()),
        xref="x domain",
        yref="y domain",
        x=1.015, y=0.9,
        sizex=0.25,
        sizey=0.3,
        xanchor="right", yanchor="top"
    )
)

# Add UoB logo
fig.add_layout_image(
    dict(
        source='data:image/png;base64,{}'.format(uob_logo.decode()),
        xref="x domain",
        yref="y domain",
        x=0.99, y=0.99,
        sizex=0.2,
        sizey=0.3,
        xanchor="right", yanchor="top"
    )
)

# Add shading to 2-site period
fig.add_vrect(x0="1990-01-01", x1="2011-06-01", 
              #annotation_text=None, annotation_position="left",
              fillcolor="grey", opacity=0.1, line_width=0)

# Text annotation denoting period
fig.add_annotation(
    xref="x",
    yref="y",
    x="2014-01-01",
    y=45,
    align="left",
    text="DECC network",
    showarrow=False,
    font=dict(
        family="Arial",
        size=15,
        color="Grey",
        )
)
fig.add_annotation(
    xref="x",
    yref="y",
    x="2011-06-01",
    arrowwidth=2,
    arrowcolor="grey",
    align="right",
    y=40,
    ax=150,
    ay=0,
    arrowside="start",
    text=None
)

# Text annotation denoting 2-site period
fig.add_annotation(
    xref="x",
    yref="y",
    x="2008-09-01",
    y=40,
    align="right",
    text="Mace Head, Cabauw",
    showarrow=False,
    font=dict(
        family="Arial",
        size=15,
        color="Grey",
        )
)
fig.add_annotation(
    xref="x",
    yref="y",
    x="2011-06-01",
    arrowwidth=2,
    arrowcolor="grey",
    align="right",
    y=35,
    ax=-150,
    ay=0,
    arrowside="start",
    text=None
)


# # Text annotation for CH4
# fig.add_annotation(
#     xref="paper",
#     yref="y",
#     x=1.01,
#     y=df_ch4.loc["2020-01-01", "InTEM"],
#     xanchor="left",
#     align="left",
# #    text="<b>CH<sub>4</sub></b>",
#     text="<b>Methane</b>",
#     showarrow=False,
#     font=dict(
#         family="Arial",
#         size=20,
#         color=color_ch4[0]
#         )
# )

# # Text annotation for N2O
# fig.add_annotation(
#     xref="paper",
#     yref="y",
#     x=1.01,
#     y=df_n2o.loc["2020-01-01", "InTEM"],
#     xanchor="left",
#     align="left",
# #    text="<b>CH<sub>4</sub></b>",
#     text="<b>Nitrous<br>oxide</b>",
#     showarrow=False,
#     font=dict(
#         family="Arial",
#         size=20,
#         color=color_n2o[0]
#         )
# )

# # Text annotation for F-gases
# fig.add_annotation(
#     xref="paper",
#     yref="y",
#     x=1.01,
#     y=df_fgas.loc["2020-01-01", "InTEM"],
#     xanchor="left",
#     align="left",
# #    text="<b>CH<sub>4</sub></b>",
#     text="<b>F-gases</b>",
#     showarrow=False,
#     font=dict(
#         family="Arial",
#         size=20,
#         color=color_fgas[0]
#         )
# )

fig.write_image("InTEM_CO2e_2021.pdf")
fig.write_image("InTEM_CO2e_2021.png")
fig.write_image("InTEM_CO2e_2021.svg")
fig.write_html("InTEM_CO2e_2021.html")

fig.show()


# In[9]:


ds_flux = xr.open_dataset("data/flux_MetOffice-InTEM_ch4_MHD_TAC_RGL_TTA_BSD_HFD_CBW_WAO_JFJ_CMN_ZEP_TRN_LUT_2012-2021_Dec2022.nc")

## Set the longitude and latitude to only include the UK;

min_lon = -12
min_lat = 50

max_lon = 2 
max_lat = 62 

ds_flux = ds_flux.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))


# In[10]:


lat = np.arange(50.041, 61.975002, 0.234)
lon = np.arange(-11.66, 1.716001, 0.352)
area = areagrid(lat, lon)
#area


# In[11]:


## Create a function to produce a mean flux reading for each year;

def weighted_temporal_mean(ds, var):

    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds[var]

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")
    
    return obs_sum / ones_out


# In[12]:


Intem_new_year = weighted_temporal_mean(ds_flux, "flux_prior")
Intem2012 = Intem_new_year[dict(time=0)] * area
Intem2013 = Intem_new_year[dict(time=1)] * area
Intem2014 = Intem_new_year[dict(time=2)] * area
Intem2015 = Intem_new_year[dict(time=3)] * area


# In[13]:


def intem_by_year(year):
        
    Intem_new_year = weighted_temporal_mean(ds_flux, "flux_prior")
    
    intem = Intem_new_year[dict(time=year)] * area
    

    return intem


# In[14]:


def read_invent_ch4(year, species):
    import pandas as pd
    from itertools import takewhile
    import plotly.graph_objects as go
    import base64
    import numpy as np
    import glob
    import xarray as xr
    
    csv = glob.glob(f"data/ch4-edgarv7-ukghg-yearly/ch4-edgarv7-ukghg-europe-{year}/ch4-edgarv7-ukghg-{species}_EUROPE_{year}.nc")
    
    flux = xr.open_dataset(csv[0])
    
    min_lon = -12
    min_lat = 50

    max_lon = 2 
    max_lat = 62 
    
    cropped_ds = flux.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))
    
    cropped_ds1 = cropped_ds[dict(time=0)] * area
    
    flux1 = flux[dict(time=0)] #* area
    
    return cropped_ds1


# In[15]:


years = ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]

for name in years:
    
    a = read_invent_ch4(name, "domcom")
    b = read_invent_ch4(name, "energyprod")
    c = read_invent_ch4(name, "offshore")
    d = read_invent_ch4(name, "othertrans")
    e = read_invent_ch4(name, "roadtrans")
    f = read_invent_ch4(name, "total")
    
    fract = 100*(a.flux + b.flux + c.flux + d.flux + e.flux / f.flux)
    
    fract = np.clip(fract, 0, 100)
    
    test = fract.reindex_like(Intem2012, method='nearest', tolerance=0.01)
    
    
    if name == "2012":
        tenerg12 = test/100 * intem_by_year(0)
        te12 = tenerg12.sum()
        
        fe12 = fract.mean()/100
        
    if name == "2013":
        tenerg13 = test/100 * intem_by_year(1)
        te13 = tenerg13.sum()
        
        fe13 = fract.mean()/100
        
    if name == "2014":
        tenerg14 = test/100 * intem_by_year(2)
        te14 = tenerg14.sum()
        
        fe14 = fract.mean()/100
        
    if name == "2015":
        tenerg15 = test/100 * intem_by_year(3)
        te15 = tenerg15.sum()
        
        fe15 = fract.mean()/100
    
    if name == "2016":
        tenerg16 = test/100 * intem_by_year(4)
        te16 = tenerg16.sum()
        
        fe16 = fract.mean()/100
        
    if name == "2017":
        tenerg17 = test/100 * intem_by_year(5)
        te17 = tenerg17.sum()
        
        fe17 = fract.mean()/100
        
    if name == "2018":
        tenerg18 = test/100 * intem_by_year(6)
        te18 = tenerg18.sum()
        
        fe18 = fract.mean()/100
        
    if name == "2019":
        tenerg19 = test/100 * intem_by_year(7)
        te19 = tenerg19.sum()
        
        fe19 = fract.mean()/100
        
    if name == "2020":
        tenerg20 = test/100 * intem_by_year(8)
        te20 = tenerg20.sum()
        
        fe20 = fract.mean()/100

    
te = xr.concat((te12, te13, te14, te15, te16, te17, te18, te19, te20), dim="time")
te = te * 16 * 31536000 * 28 / 10000000000000
te = te.to_pandas()

fe = xr.concat((fe12, fe13, fe14, fe15, fe16, fe17, fe18, fe19, fe20), dim="time")

fe = fe.to_pandas()


# In[16]:


years = ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]

for name, in zip(years):
    
    a = read_invent_ch4(name, "indcom")
    c = read_invent_ch4(name, "indproc")
    b = read_invent_ch4(name, "total")
    
    fract = 100*(a.flux + c.flux / b.flux)
    
    fract = np.clip(fract, 0, 100)
    
    test = fract.reindex_like(Intem2012, method='nearest', tolerance=0.01)
    
    if name == "2012":
        tind12 = test/100 * intem_by_year(0)
        ti12 = tind12.sum()
        
        find12 = fract.mean()/100
        
    if name == "2013":
        tind13 = test/100 * intem_by_year(1)
        ti13 = tind13.sum()
        
        find13 = fract.mean()/100
        
    if name == "2014":
        tind14 = test/100 * intem_by_year(2)
        ti14 = tind14.sum()
        
        find14 = fract.mean()/100
        
    if name == "2015":
        tind15 = test/100 * intem_by_year(3)
        ti15 = tind15.sum()
        
        find15 = fract.mean()/100
        
    if name == "2016":
        tind16 = test/100 * intem_by_year(4)
        ti16 = tind16.sum()
        
        find16 = fract.mean()/100
        
    if name == "2017":
        tind17 = test/100 * intem_by_year(5)
        ti17 = tind17.sum()
        
        find17 = fract.mean()/100
        
    if name == "2018":
        tind18 = test/100 * intem_by_year(6)
        ti18 = tind18.sum()
        
        find18 = fract.mean()/100
        
    if name == "2019":
        tind19 = test/100 * intem_by_year(7)
        ti19 = tind19.sum()
        
        find19 = fract.mean()/100
        
    if name == "2020":
        tind20 = test/100 * intem_by_year(8)
        ti20 = tind20.sum()
        
        find20 = fract.mean()/100
    
tind = xr.concat((ti12, ti13, ti14, ti15, ti16, ti17, ti18, ti19, ti20), dim="time")
tind = tind * 16 * 31536000 * 28 / 1000000000000
tind = tind.to_pandas()

find = xr.concat((find12, find13, find14, find15, find16, find17, find18, find19, find20), dim="time")

find = find.to_pandas()


# In[17]:


years = ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]

for name, in zip(years):
    
    a = read_invent_ch4(name, "agric")
    b = read_invent_ch4(name, "total")
    
    fract = 100*(a.flux / b.flux)
    
    fract = np.clip(fract, 0, 100)
    
    test = fract.reindex_like(Intem2012, method='nearest', tolerance=0.01)
    
    if name == "2012":
        tagri12 = test/100 * intem_by_year(0)
        ta12 = tagri12.sum()
        
        fa12 = fract.mean()/100
        
    if name == "2013":
        tagri13 = test/100 * intem_by_year(1)
        ta13 = tagri13.sum()
        
        fa13 = fract.mean()/100
        
    if name == "2014":
        tagri14 = test/100 * intem_by_year(2)
        ta14 = tagri14.sum()
        
        fa14 = fract.mean()/100
        
    if name == "2015":
        tagri15 = test/100 * intem_by_year(3)
        ta15 = tagri15.sum()
        
        fa15 = fract.mean()/100
        
    if name == "2016":
        tagri16 = test/100 * intem_by_year(4)
        ta16 = tagri16.sum()
        
        fa16 = fract.mean()/100
        
    if name == "2017":
        tagri17 = test/100 * intem_by_year(5)
        ta17 = tagri17.sum()
        
        fa17 = fract.mean()/100
        
    if name == "2018":
        tagri18 = test/100 * intem_by_year(6)
        ta18 = tagri18.sum()
        
        fa18 = fract.mean()/100
        
    if name == "2019":
        tagri19 = test/100 * intem_by_year(7)
        ta19 = tagri19.sum()
        
        fa19 = fract.mean()/100
        
    if name == "2020":
        tagri20 = test/100 * intem_by_year(8)
        ta20 = tagri20.sum()
        
        fa20 = fract.mean()/100
    
ta = xr.concat((ta12, ta13, ta14, ta15, ta16, ta17, ta18, ta19, ta20), dim="time")
ta = ta * 16 * 31536000 * 28 / 1000000000000
ta = ta.to_pandas()

fa = xr.concat((fa12, fa13, fa14, fa15, fa16, fa17, fa18, fa19, fa20), dim="time")

fa = fa.to_pandas()


# In[18]:


years = ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]

for name, in zip(years):
    
    a = read_invent_ch4(name, "waste")
    b = read_invent_ch4(name, "total")
    
    fract = 100*(a.flux / b.flux)
    
    fract = np.clip(fract, 0, 100)
    
    test = fract.reindex_like(Intem2012, method='nearest', tolerance=0.01)
    
    if name == "2012":
        twaste12 = test/100 * intem_by_year(0)
        tw12 = twaste12.sum()
        
        fw12 = fract.mean()/100
        
    if name == "2013":
        twaste13 = test/100 * intem_by_year(1)
        tw13 = twaste13.sum()
        
        fw13 = fract.mean()/100
        
    if name == "2014":
        twaste14 = test/100 * intem_by_year(2)
        tw14 = twaste14.sum()
        
        fw14 = fract.mean()/100
        
    if name == "2015":
        twaste15 = test/100 * intem_by_year(3)
        tw15 = twaste15.sum()
        
        fw15 = fract.mean()/100
        
    if name == "2016":
        twaste16 = test/100 * intem_by_year(4)
        tw16 = twaste16.sum()
        
        fw16 = fract.mean()/100
        
    if name == "2017":
        twaste17 = test/100 * intem_by_year(5)
        tw17 = twaste17.sum()
        
        fw17 = fract.mean()/100
        
    if name == "2018":
        twaste18 = test/100 * intem_by_year(6)
        tw18 = twaste18.sum()
        
        fw18 = fract.mean()/100
        
    if name == "2019":
        twaste19 = test/100 * intem_by_year(7)
        tw19 = twaste19.sum()
        
        fw19 = fract.mean()/100
        
    if name == "2020":
        twaste20 = test/100 * intem_by_year(8)
        tw20 = twaste20.sum()
        
        fw20 = fract.mean()/100
        
twaste = xr.concat((tw12, tw13, tw14, tw15, tw16, tw17, tw18, tw19, tw20), dim="time")
twaste = twaste * 16 * 31536000 * 28 / 1000000000000
twaste = twaste.to_pandas()
#twaste.reindex(InTEM_total.index)
twaste.fillna(0, inplace=True)

fw = xr.concat((fw12, fw13, fw14, fw15, fw16, fw17, fw18, fw19, fw20), dim="time")

fw = fw.to_pandas()


# In[19]:


species = ["1. Energy", "2.  Industrial processes", "3.  Agriculture", "4. Land use, land-use change and forestry", "5.  Waste"]
ch4_fract = pd.DataFrame(df_ch4)

for name in species:
    data = pd.DataFrame((dfCH4.loc[name] / dfCH4.loc["Total CH4 emissions with CH4 from LULUCF"]), columns=[name]) 
    data.index = pd.to_datetime(data.index, format='%Y')
    
    ch4_fract = pd.concat([ch4_fract, data], axis=1).reindex(df_ch4.index)
    ch4_fract.dropna


# In[20]:


species = ["1. Energy", "2.  Industrial processes", "3.  Agriculture", "4. Land use, land-use change and forestry", "5.  Waste"]
ch4_gt = pd.DataFrame()

for name in species:
    d = pd.DataFrame((ch4_fract[name] * ch4_fract["InTEM"]), columns=[name])
    d.index = pd.to_datetime(d.index, format='%Y')
    
    d_std = pd.DataFrame((ch4_fract[name] * ch4_fract["InTEM_std"]), columns=[name+"_std"])
    d_std.index = pd.to_datetime(d_std.index, format='%Y')
    
    ch4_gt = pd.concat([ch4_gt, d, d_std], axis=1)


# In[21]:


ch4_gt_all = ch4_gt
year = ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]

for date in year:
    ch4_gt_all.loc[date, "1. Energy"] = te.loc[date]
    ch4_gt_all.loc[date, "2.  Industrial processes"] = tind.loc[date]
    ch4_gt_all.loc[date, "3.  Agriculture"] = ta.loc[date]
    #ch4_gt_all.loc[date, "4. Land use, land-use change and forestry"] = tLULUCF.loc[date]
    ch4_gt_all.loc[date, "5.  Waste"] = twaste.loc[date]
    
    ch4_gt_all.loc[date, "1. Energy_std"] = ch4_fract.loc[date, "InTEM_std"] * (fe.loc[date])
    ch4_gt_all.loc[date, "2.  Industrial processes_std"] = ch4_fract.loc[date, "InTEM_std"] * (find.loc[date])
    ch4_gt_all.loc[date, "3.  Agriculture_std"] = ch4_fract.loc[date, "InTEM_std"] * (fa.loc[date])
    #ch4_gt_all.loc[date, "4. Land use, land-use change and forestry"] = tLULUCF.loc[date]
    ch4_gt_all.loc[date, "5.  Waste_std"] = ch4_fract.loc[date, "InTEM_std"] * (fw.loc[date])


# In[22]:


species = ["1. Energy", "2.  Industrial processes", "3.  Agriculture", "4. Land use, land-use change and forestry", "5.  Waste"]
n2o_fract = pd.DataFrame(df_n2o)

for name in species:
    data = pd.DataFrame((dfN2O.loc[name] / dfN2O.loc["Total direct N2O emissions with N2O from LULUCF"]), columns=[name]) 
    data.index = pd.to_datetime(data.index, format='%Y')
    
    n2o_fract = pd.concat([n2o_fract, data], axis=1).reindex(df_n2o.index)
    n2o_fract.dropna


# In[23]:


species = ["1. Energy", "2.  Industrial processes", "3.  Agriculture", "4. Land use, land-use change and forestry", "5.  Waste"]
n2o_gt = pd.DataFrame()

for name in species:
    d = pd.DataFrame((n2o_fract[name] * n2o_fract["InTEM"]), columns=[name])
    d.index = pd.to_datetime(d.index, format='%Y')
    
    d_std = pd.DataFrame((n2o_fract[name] * n2o_fract["InTEM_std"]), columns=[name+"_std"])
    d_std.index = pd.to_datetime(d_std.index, format='%Y')
    
    n2o_gt = pd.concat([n2o_gt, d, d_std], axis=1)
    n2o_gt.dropna


# In[24]:


species = ["1. Energy", "2.  Industrial processes", "3.  Agriculture", "4. Land use, land-use change and forestry", "5.  Waste"]
Invent_total = pd.DataFrame()

for name in species:
    s = pd.DataFrame(dfN2O.loc[name] + dfCH4.loc[name], columns=[name])
    
    Invent_total = pd.concat([Invent_total, s], axis=1)
    #Invent_total.index = pd.to_datetime(Invent_total.index, format='%Y')


# In[25]:


InTEM_total = ch4_gt_all + n2o_gt
#InTEM_total = InTEM_total.drop(index=(["2012", "2013", "2014", "2015"]))
InTEM_total = InTEM_total.reindex(ch4_gt.index)
#InTEM_total.index = pd.to_datetime(InTEM_total.index, format='%Y')
#InTEM_total.fillna(0)
#InTEM_total.drop(["1990"], inplace=True)
#InTEM_total


# In[26]:


## zoomed in version

Invent_total = Invent_total.drop(["1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999", 
                   "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009",
                   "2010", "2011"])

InTEM_total = InTEM_total.drop(["1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999", 
                  "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", 
                  "2010", "2011"])

df_fgas = df_fgas.drop(["1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999", 
                  "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", 
                  "2010", "2011"])


# In[27]:


fig2 = go.Figure()

mo_logo = base64.b64encode(open("metoffice_logo.png", 'rb').read())
uob_logo = base64.b64encode(open("uob_logo.png", 'rb').read())

alpha = 0.6
color_1 = ["rgba(86, 119, 194, 1)", f"rgba(86, 119, 194, {alpha})"]
color_2 = ["rgba(99, 182, 137, 1)", f"rgba(99, 182, 137, {alpha})"]
color_3 = ["rgba(238,92,66, 1)", f"rgba(238,92,66, {alpha})"]
color_4 = ["rgba(255,215,0, 1)", f"rgba(255,215,0, {alpha})"]
color_5 = ["rgba(142,229,238, 1)", f"rgba(142,229,238, {alpha})"]
color_6 = ["rgba(169,169,169, 1)", f"rgba(169,169,169, {alpha})"]
color_7 = ["rgba(0, 0, 0, 1)", f"rgba(0, 0, 0, {alpha})"]

date_range = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
#date_range = range=[pd.Timestamp("2012-01-01"), pd.Timestamp("2020-01-01")]
data_range1 = [2012, 2013, 2014, 2015]

def errorbars(df, var, color, dash, name, showlegend=False):

    error_minus=df[var] - df[f"{var}_std"]
    error_plus=df[var] + df[f"{var}_std"]

    fig2.add_trace(go.Scatter(
        x=date_range,
        y=error_minus.values,
        fill=None,
        mode='lines',
        line=dict(color=color[1], width=0.1),
        showlegend=False,
        hoverinfo='skip'
        ))

    fig2.add_trace(go.Scatter(
        x=date_range,
        y=error_plus.values,
        fill="tonexty",
        mode='lines',
        line=dict(color=color[1], width=0.1),
        showlegend=False,
        hoverinfo='skip'
        ))
    
    fig2.add_trace(go.Scatter(
        x=date_range,
        y=df[var],
        mode="lines",
        line=dict(color=color[0], dash="dot", width = 0.1),
        showlegend=showlegend,
        name=name,
        hovertemplate = 'InTEM %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
        ))

fig2.add_trace(go.Scatter(
    x=date_range,
    y=Invent_total["1. Energy"],
    mode='lines',
    line=dict(color=color_1[0], width=3.),
    showlegend=True,
    hovertemplate = 'Inventory Energy %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Energy and Transport",
    ))

fig2.add_trace(go.Scatter(
    x=date_range,
    y=Invent_total["2.  Industrial processes"],
    mode='lines',
    line=dict(color=color_2[0], width=3.),
    showlegend=True,
    hovertemplate = 'Inventory Industrial Processes %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Industrial processes",
    ))

fig2.add_trace(go.Scatter(
    x=date_range,
    y=Invent_total["3.  Agriculture"],
    mode='lines',
    line=dict(color=color_3[0], width=3.),
    showlegend=True,
    hovertemplate = 'Inventory Agriculture %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Agriculture",
    ))

fig2.add_trace(go.Scatter(
    x=date_range,
    y=Invent_total["4. Land use, land-use change and forestry"],
    mode='lines',
    line=dict(color=color_4[0], width=3.),
    showlegend=True,
    hovertemplate = 'Inventory LULUCF %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Land use, Land-use change and forestry",
    ))

fig2.add_trace(go.Scatter(
    x=date_range,
    y=Invent_total["5.  Waste"],
    mode='lines',
    line=dict(color=color_5[0], width=3.),
    showlegend=True,
    hovertemplate = 'Inventory Waste %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Waste",
    ))

errorbars(InTEM_total, "1. Energy", color_1, None, "InTEM")
fig2.add_trace(go.Scatter(
    x=date_range,
    y=InTEM_total["1. Energy"],
    mode='lines',
    line=dict(color=color_1[0], dash="dot", width=5),
    showlegend=False,
    hovertemplate = 'InTEM Energy %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Energy",
    ))

errorbars(InTEM_total, "2.  Industrial processes", color_2, None, "InTEM")
fig2.add_trace(go.Scatter(
    x=date_range,
    y=InTEM_total["2.  Industrial processes"],
    mode='lines',
    line=dict(color=color_2[0], dash="dot", width=5),
    showlegend=False,
    hovertemplate = 'InTEM Industrial Processes %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Industrial processes",
    ))

errorbars(InTEM_total, "3.  Agriculture", color_3, None, "InTEM")
fig2.add_trace(go.Scatter(
    x=date_range,
    y=InTEM_total["3.  Agriculture"],
    mode='lines',
    line=dict(color=color_3[0], dash="dot", width=5),
    showlegend=False,
    hovertemplate = 'InTEM Agriculture %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Agriculture",
    ))

errorbars(InTEM_total, "4. Land use, land-use change and forestry", color_4, None, "InTEM")
fig2.add_trace(go.Scatter(
    x=date_range,
    y=InTEM_total["4. Land use, land-use change and forestry"],
    mode='lines',
    line=dict(color=color_4[0], dash="dot", width=5),
    showlegend=False,
    hovertemplate = 'InTEM LULUCF %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Land use, Land-use change and forestry",
    ))

errorbars(InTEM_total, "5.  Waste", color_5, None, "InTEM")
fig2.add_trace(go.Scatter(
    x=date_range,
    y=InTEM_total["5.  Waste"],
    mode='lines',
    line=dict(color=color_5[0], dash="dot", width=5),
    showlegend=False,
    hovertemplate = 'InTEM Waste %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Waste",
    ))

fig2.add_trace(go.Scatter(
    x=date_range,
    y=dfODS["ODS"],
    mode='lines',
    line=dict(color=color_6[0], width=3.),
    showlegend=True,
    hovertemplate = 'Inventory Product uses as substitutes for ODS %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Product uses as substitutes for ODS",
    ))

errorbars(df_fgas, "InTEM_ODS", color_6, None, "InTEM")
fig2.add_trace(go.Scatter(
    x=date_range,
    y=df_fgas["InTEM_ODS"],
    mode='lines',
    line=dict(color=color_6[0], dash = "dot",  width=5),
    showlegend=False,
    hovertemplate = 'InTEM F Gases %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="F Gases",
    ))

#fig2.add_trace(go.Scatter(
#    x=date_range,
#    y=te + n2o_gt["1. Energy"],
#    mode='lines',
#    line=dict(color=color_1[0], dash="dot", width=5),
#    showlegend=False,
#    hovertemplate = 'InTEM Energy %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
#    name="Energy",
#    ))

#fig2.add_trace(go.Scatter(
#    x=date_range,
#    y=tind + n2o_gt["2.  Industrial processes"],
#    mode='lines',
#    line=dict(color=color_2[0], dash="dot", width=5),
#    showlegend=False,
#    hovertemplate = 'InTEM Industrial %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
#    name="Industrial Processes",
#    ))

#fig2.add_trace(go.Scatter(
#    x=date_range,
#    y=ta + n2o_gt["3.  Agriculture"],
#    mode='lines',
#    line=dict(color=color_3[0], dash="dot", width=5),
#    showlegend=False,
#    hovertemplate = 'InTEM Agriculture %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
#    name="Agriculture",
 #   ))

#fig2.add_trace(go.Scatter(
#    x=date_range,
 #   y=tLULUCF + n2o_gt["4. Land use, land-use change and forestry"],
  #  mode='lines',
   # line=dict(color=color_4[0], dash="dot", width=5),
#    showlegend=False,
#    hovertemplate = 'InTEM LULUCF %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
#    name="LULUCF",
#    ))

#fig2.add_trace(go.Scatter(
#    x=date_range,
#    y=twaste + n2o_gt["5.  Waste"],
#    mode='lines',
#    line=dict(color=color_1[0], dash="dot", width=5),
#    showlegend=False,
#    hovertemplate = 'InTEM Waste %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
#    name="Waste",
#    ))


fig2.add_trace(go.Scatter(
    x=[pd.Timestamp("1900-01-01"), pd.Timestamp("1900-01-02")],
    y=[0., 0.],
    line=dict(color="black", dash="dot", width=4),
    marker=dict(opacity=0., size=0),
    showlegend=True,
    name="InTEM"
    ))
    
fig2.update_layout(
    yaxis_title="Annual Emissions (GtCO₂-equ)",
    template="simple_white",
    autosize=False,
    width=550*2,
    height=400*2,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        traceorder="normal",
        orientation="h",
        font_size=18),
    margin=dict(l=55, r=10, t=10, b=40),
    # paper_bgcolor='rgba(0,0,0,0)',
    # plot_bgcolor='rgba(0,0,0,0)'
)
    


fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='Lightgrey')

fig2.layout.font.size=20
fig2.layout.font.family="Arial"

fig2.update_xaxes(range=[pd.Timestamp("2012-01-01"),
                        pd.Timestamp("2020-01-01")])

fig2.update_yaxes(range=[0,
                        65])

# Add UKMO logo
fig2.add_layout_image(
    dict(
        source='data:image/png;base64,{}'.format(mo_logo.decode()),
        xref="x domain",
        yref="y domain",
        x=0.225, y=0.92,
        sizex=0.2,
        sizey=0.25,
        xanchor="right", yanchor="top"
    )
)

# Add UoB logo
fig2.add_layout_image(
    dict(
        source='data:image/png;base64,{}'.format(uob_logo.decode()),
        xref="x domain",
        yref="y domain",
        x=0.2, y=0.99,
        sizex=0.15,
        sizey=0.25,
        xanchor="right", yanchor="top"
    )
)

fig2.write_image("Annual_Emissions_by_sector1.png")
#fig2.write_image("Annual_Emissions_by_sector.pdf")
#fig2.write_html("Annual_Emissions_by_sector.html")

fig2.show()


# In[ ]:




