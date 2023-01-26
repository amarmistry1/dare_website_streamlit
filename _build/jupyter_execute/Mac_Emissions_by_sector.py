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


# In[2]:


dfCH4 = load_ch4()
dfN2O = load_n2o()
dfODS = load_fgas()


# In[3]:


dfch4_new = (read_invent("GBR", "2022", "Table10s4"))


# In[4]:


fgas1 = [f"{hfc}" for hfc in ["23", "32", "125", "134a", "143a", "152a", "227ea", "245fa", "365mfc", "4310mee"]] +     [f"pfc{pfc}" for pfc in ["14", "116", "218", "318"]] +     ["sf6"]


# In[5]:


dftest = read_invent_ODS("2022", fgas1[0])


# In[30]:


dftest = read_invent_ODS("2022", "HFC-23") + read_invent_ODS("2022", "HFC-32")
dftest


# Read in InTEM data

# In[11]:


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


# In[12]:


df_ch4, _, _, _ = read_intem("ch4")
df_n2o, _, _, _ = read_intem("n2o")


# In[15]:


# Get F-gases
fgas = [f"hfc{hfc}" for hfc in ["23", "32", "125", "134a", "143a", "152a", "227ea", "245fa", "365mfc", "4310mee"]] +     [f"pfc{pfc}" for pfc in ["14", "116", "218", "318"]] +     ["sf6"]

df_fgas, _, _, _ = read_intem(fgas[0])

uncert = df_fgas["InTEM_std"]**2

for fg in fgas[1:]:
    dfs, _, _, _ = read_intem(fg)
    df_fgas += dfs
    uncert += df_fgas["InTEM_std"]**2

df_fgas["InTEM_std"] = np.sqrt(uncert.values)


# In[18]:


df_fgas["InTEM_ODS"] = (dfODS["ODS"] / dfODS["Total"]) * df_fgas["InTEM"]
df_fgas["InTEM_ODS_std"] = (dfODS["ODS"] / dfODS["Total"]) * df_fgas["InTEM_std"]
#df_fgas


# In[9]:


energy_ch4_fraction = pd.DataFrame(dfCH4.loc["1. Energy"] / dfCH4.loc["Total CH4 emissions with CH4 from LULUCF"], columns=["Energy_Fraction"])
energy_ch4_fraction.index = pd.to_datetime(energy_ch4_fraction.index, format='%Y')

industry_ch4_fraction = pd.DataFrame(dfCH4.loc["2.  Industrial processes"] / dfCH4.loc["Total CH4 emissions with CH4 from LULUCF"], columns=["Industry_Fraction"])
industry_ch4_fraction.index = pd.to_datetime(industry_ch4_fraction.index, format='%Y')

agri_ch4_fraction = pd.DataFrame(dfCH4.loc["3.  Agriculture"] / dfCH4.loc["Total CH4 emissions with CH4 from LULUCF"], columns=["Agri_Fraction"])
agri_ch4_fraction.index = pd.to_datetime(agri_ch4_fraction.index, format='%Y')

lulucf_ch4_fraction = pd.DataFrame(dfCH4.loc["4. Land use, land-use change and forestry"] / dfCH4.loc["Total CH4 emissions with CH4 from LULUCF"], columns=["LULUCF_Fraction"])
lulucf_ch4_fraction.index = pd.to_datetime(lulucf_ch4_fraction.index, format='%Y')

waste_ch4_fraction = pd.DataFrame(dfCH4.loc["5.  Waste"] / dfCH4.loc["Total CH4 emissions with CH4 from LULUCF"], columns=["Waste_Fraction"])
waste_ch4_fraction.index = pd.to_datetime(waste_ch4_fraction.index, format='%Y')


# In[14]:


frames = [energy_ch4_fraction, industry_ch4_fraction, agri_ch4_fraction, lulucf_ch4_fraction, waste_ch4_fraction, df_ch4]
df_ch4_new = pd.concat(frames, axis=1).reindex(df_n2o.index)

df_ch4_new["InTEM_Energy"] = df_ch4_new["InTEM"] *df_ch4_new["Energy_Fraction"]
df_ch4_new["InTEM_Energy_std"] = df_ch4_new["InTEM_std"] *df_ch4_new["Energy_Fraction"]

df_ch4_new["InTEM_Industry"] = df_ch4_new["InTEM"] *df_ch4_new["Industry_Fraction"]
df_ch4_new["InTEM_Industry_std"] = df_ch4_new["InTEM_std"] *df_ch4_new["Industry_Fraction"]

df_ch4_new["InTEM_Agriculture"] = df_ch4_new["InTEM"] *df_ch4_new["Agri_Fraction"]
df_ch4_new["InTEM_Agriculture_std"] = df_ch4_new["InTEM_std"] *df_ch4_new["Agri_Fraction"]

df_ch4_new["InTEM_LULUCF"] = df_ch4_new["InTEM"] *df_ch4_new["LULUCF_Fraction"]
df_ch4_new["InTEM_LULUCF_std"] = df_ch4_new["InTEM_std"] *df_ch4_new["LULUCF_Fraction"]

df_ch4_new["InTEM_Waste"] = df_ch4_new["InTEM"] *df_ch4_new["Waste_Fraction"]
df_ch4_new["InTEM_Waste_std"] = df_ch4_new["InTEM_std"] *df_ch4_new["Waste_Fraction"]


# Breakdown N2O

# In[23]:


energy_n2o_fraction = pd.DataFrame(dfN2O.loc["1. Energy"] / dfN2O.loc["Total direct N2O emissions with N2O from LULUCF"], columns=["Energy_Fraction"])
energy_n2o_fraction.index = pd.to_datetime(energy_n2o_fraction.index, format='%Y')

industry_n2o_fraction = pd.DataFrame(dfN2O.loc["2.  Industrial processes"] / dfN2O.loc["Total direct N2O emissions with N2O from LULUCF"], columns=["Industry_Fraction"])
industry_n2o_fraction.index = pd.to_datetime(industry_n2o_fraction.index, format='%Y')

agri_n2o_fraction = pd.DataFrame(dfN2O.loc["3.  Agriculture"] / dfN2O.loc["Total direct N2O emissions with N2O from LULUCF"], columns=["Agri_Fraction"])
agri_n2o_fraction.index = pd.to_datetime(agri_n2o_fraction.index, format='%Y')

lulucf_n2o_fraction = pd.DataFrame(dfN2O.loc["4. Land use, land-use change and forestry"] / dfN2O.loc["Total direct N2O emissions with N2O from LULUCF"], columns=["LULUCF_Fraction"])
lulucf_n2o_fraction.index = pd.to_datetime(lulucf_n2o_fraction.index, format='%Y')

waste_n2o_fraction = pd.DataFrame(dfN2O.loc["5.  Waste"] / dfN2O.loc["Total direct N2O emissions with N2O from LULUCF"], columns=["Waste_Fraction"])
waste_n2o_fraction.index = pd.to_datetime(waste_n2o_fraction.index, format='%Y')


# In[24]:


frames = [energy_n2o_fraction, industry_n2o_fraction, agri_n2o_fraction, lulucf_n2o_fraction, waste_n2o_fraction, df_n2o]
df_n2o_new = pd.concat(frames, axis=1).reindex(df_n2o.index)

df_n2o_new["InTEM_Energy"] = df_n2o_new["InTEM"] *df_n2o_new["Energy_Fraction"]
df_n2o_new["InTEM_Energy_std"] = df_n2o_new["InTEM_std"] *df_n2o_new["Energy_Fraction"]

df_n2o_new["InTEM_Industry"] = df_n2o_new["InTEM"] *df_n2o_new["Industry_Fraction"]
df_n2o_new["InTEM_Industry_std"] = df_n2o_new["InTEM_std"] *df_n2o_new["Industry_Fraction"]

df_n2o_new["InTEM_Agriculture"] = df_n2o_new["InTEM"] *df_n2o_new["Agri_Fraction"]
df_n2o_new["InTEM_Agriculture_std"] = df_n2o_new["InTEM_std"] *df_n2o_new["Agri_Fraction"]

df_n2o_new["InTEM_LULUCF"] = df_n2o_new["InTEM"] *df_n2o_new["LULUCF_Fraction"]
df_n2o_new["InTEM_LULUCF_std"] = df_n2o_new["InTEM_std"] *df_n2o_new["LULUCF_Fraction"]

df_n2o_new["InTEM_Waste"] = df_n2o_new["InTEM"] *df_n2o_new["Waste_Fraction"]
df_n2o_new["InTEM_Waste_std"] = df_n2o_new["InTEM_std"] *df_n2o_new["Waste_Fraction"]


# Sum of N2O and CH4

# In[25]:


energy_both = dfN2O.loc["1. Energy"] + dfCH4.loc["1. Energy"]
industrial_both = dfN2O.loc["2.  Industrial processes"] + dfCH4.loc["2.  Industrial processes"]
agricultural_both = dfN2O.loc["3.  Agriculture"] + dfCH4.loc["3.  Agriculture"]
LULUCF_both = dfN2O.loc["4. Land use, land-use change and forestry"] + dfCH4.loc["4. Land use, land-use change and forestry"]
waste_both = dfN2O.loc["5.  Waste"] + dfCH4.loc["5.  Waste"]


# In[27]:


energy_both_intem = pd.DataFrame(df_n2o_new["InTEM_Energy"] + df_ch4_new["InTEM_Energy"], columns=["InTEM_Energy"])
energy_both_intem["InTEM_Energy_std"] = df_n2o_new["InTEM_Energy_std"] + df_ch4_new["InTEM_Energy_std"]
energy_both_intem.index = pd.to_datetime(energy_both_intem.index, format='%Y')

industrial_both_intem = pd.DataFrame(df_n2o_new["InTEM_Industry"] + df_ch4_new["InTEM_Industry"], columns=["InTEM_Industry"])
industrial_both_intem["InTEM_Industry_std"] = df_n2o_new["InTEM_Industry_std"] + df_ch4_new["InTEM_Industry_std"]
industrial_both_intem.index = pd.to_datetime(industrial_both_intem.index, format='%Y')

agricultural_both_intem = pd.DataFrame(df_n2o_new["InTEM_Agriculture"] + df_ch4_new["InTEM_Agriculture"], columns=["InTEM_Agriculture"])
agricultural_both_intem["InTEM_Agriculture_std"] = df_n2o_new["InTEM_Agriculture_std"] + df_ch4_new["InTEM_Agriculture_std"]
agricultural_both_intem.index = pd.to_datetime(agricultural_both_intem.index, format='%Y')

LULUCF_both_intem = pd.DataFrame(df_n2o_new["InTEM_LULUCF"] + df_ch4_new["InTEM_LULUCF"], columns=["InTEM_LULUCF"])
LULUCF_both_intem["InTEM_LULUCF_std"] = df_n2o_new["InTEM_LULUCF_std"] + df_ch4_new["InTEM_LULUCF_std"]
LULUCF_both_intem.index = pd.to_datetime(LULUCF_both_intem.index, format='%Y')

waste_both_intem = pd.DataFrame(df_n2o_new["InTEM_Waste"] + df_ch4_new["InTEM_Waste"], columns=["InTEM_Waste"])
waste_both_intem["InTEM_Waste_std"] = df_n2o_new["InTEM_Waste_std"] + df_ch4_new["InTEM_Waste_std"]
waste_both_intem.index = pd.to_datetime(waste_both_intem.index, format='%Y')


# In[28]:


fig2 = go.Figure()

alpha = 0.6
color_1 = ["rgba(86, 119, 194, 1)", f"rgba(86, 119, 194, {alpha})"]
color_2 = ["rgba(99, 182, 137, 1)", f"rgba(99, 182, 137, {alpha})"]
color_3 = ["rgba(238,92,66, 1)", f"rgba(238,92,66, {alpha})"]
color_4 = ["rgba(255,215,0, 1)", f"rgba(255,215,0, {alpha})"]
color_5 = ["rgba(142,229,238, 1)", f"rgba(142,229,238, {alpha})"]
color_6 = ["rgba(169,169,169, 1)", f"rgba(169,169,169, {alpha})"]
color_7 = ["rgba(0, 0, 0, 1)", f"rgba(0, 0, 0, {alpha})"]

date_range = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
#date_range = ["2019-03-01", "2019-06-01"]

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
    y=energy_both,
    mode='lines',
    line=dict(color=color_1[0], width=3.),
    showlegend=True,
    hovertemplate = 'Inventory Energy %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Energy and Transport",
    ))

fig2.add_trace(go.Scatter(
    x=date_range,
    y=industrial_both,
    mode='lines',
    line=dict(color=color_2[0], width=3.),
    showlegend=True,
    hovertemplate = 'Inventory Industrial Processes %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Industrial processes",
    ))

fig2.add_trace(go.Scatter(
    x=date_range,
    y=agricultural_both,
    mode='lines',
    line=dict(color=color_3[0], width=3.),
    showlegend=True,
    hovertemplate = 'Inventory Agriculture %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Agriculture",
    ))

fig2.add_trace(go.Scatter(
    x=date_range,
    y=LULUCF_both,
    mode='lines',
    line=dict(color=color_4[0], width=3.),
    showlegend=True,
    hovertemplate = 'Inventory LULUCF %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Land use, Land-use change and forestry",
    ))

fig2.add_trace(go.Scatter(
    x=date_range,
    y=waste_both,
    mode='lines',
    line=dict(color=color_5[0], width=3.),
    showlegend=True,
    hovertemplate = 'Inventory Waste %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Waste",
    ))

errorbars(energy_both_intem, "InTEM_Energy", color_1, None, "InTEM")
fig2.add_trace(go.Scatter(
    x=date_range,
    y=energy_both_intem["InTEM_Energy"],
    mode='lines',
    line=dict(color=color_1[0], dash="dot", width=5),
    showlegend=False,
    hovertemplate = 'InTEM Energy %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Energy",
    ))

errorbars(industrial_both_intem, "InTEM_Industry", color_2, None, "InTEM")
fig2.add_trace(go.Scatter(
    x=date_range,
    y=industrial_both_intem["InTEM_Industry"],
    mode='lines',
    line=dict(color=color_2[0], dash="dot", width=5),
    showlegend=False,
    hovertemplate = 'InTEM Industrial Processes %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Industrial processes",
    ))

errorbars(agricultural_both_intem, "InTEM_Agriculture", color_3, None, "InTEM")
fig2.add_trace(go.Scatter(
    x=date_range,
    y=agricultural_both_intem["InTEM_Agriculture"],
    mode='lines',
    line=dict(color=color_3[0], dash="dot", width=5),
    showlegend=False,
    hovertemplate = 'InTEM Agriculture %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Agriculture",
    ))

errorbars(LULUCF_both_intem, "InTEM_LULUCF", color_4, None, "InTEM")
fig2.add_trace(go.Scatter(
    x=date_range,
    y=LULUCF_both_intem["InTEM_LULUCF"],
    mode='lines',
    line=dict(color=color_4[0], dash="dot", width=5),
    showlegend=False,
    hovertemplate = 'InTEM LULUCF %{x|%Y}: %{y:.0f} Gt yr⁻¹<extra></extra>',
    name="Land use, Land-use change and forestry",
    ))

errorbars(waste_both_intem, "InTEM_Waste", color_5, None, "InTEM")
fig2.add_trace(go.Scatter(
    x=date_range,
    y=waste_both_intem["InTEM_Waste"],
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
        traceorder="normal"),
    margin=dict(l=55, r=10, t=10, b=40),
    # paper_bgcolor='rgba(0,0,0,0)',
    # plot_bgcolor='rgba(0,0,0,0)'
)
    


fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='Lightgrey')

fig2.layout.font.size=20
fig2.layout.font.family="Arial"

#fig2.write_image("Annual_Emissions_by_sector.png")
#fig2.write_image("Annual_Emissions_by_sector.pdf")
#fig2.write_html("Annual_Emissions_by_sector.html")

fig2.show()


# In[2]:


from dash import dcc


# In[43]:


dcc.Graph(id='fig2', style={'width': '90vw', 'height': '90vh'}) 


# In[49]:


from IPython.display import display, HTML, IFrame, Image


# Here is a diagram showing the locations of all the sites associated with the DECC network:

# In[50]:


display(Image("UKDECCnetwork_2021.png", width = 700, height = 800))


# In[3]:


import xarray as xr


# In[4]:


ds_flux = xr.open_dataset("data/flux_data.nc")
ds_flux


# In[5]:


Intem_new = xr.DataArray.to_dataframe(ds_flux.country_post_mean)
Intem_new.reset_index(drop=False, inplace=True)
Intem_new.set_index("time", inplace=True)
Intem_new.index = pd.to_datetime(Intem_new.index, format='%Y')
Intem_new.drop("country", axis=1,  inplace=True)

Intem_new_std = xr.DataArray.to_dataframe(ds_flux.country_post_std)
Intem_new_std.reset_index(drop=False, inplace=True)
Intem_new_std.set_index("time", inplace=True)
Intem_new_std.index = pd.to_datetime(Intem_new_std.index, format='%Y')
Intem_new_std.drop("country", axis=1,  inplace=True)

Intem_new["country_post_mean_std"] = Intem_new_std["country_post_std"]


# In[164]:


Intem_new["InTEM_new_ODS"] = (df_test["ODS"] / df_test["Total"]) * Intem_new["country_post_mean"]
#df_fgas["InTEM_new_ODS_std"] = (df_test["ODS"] / df_test["Total"]) * df_fgas["InTEM_std"]


# In[6]:


def weighted_temporal_mean(ds, var):

    month_length = ds_flux.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds_flux[var]

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")
    
    return obs_sum / ones_out


# In[7]:


Intem_new_year = weighted_temporal_mean(ds_flux, "flux_prior")
Intem_new_year_std = weighted_temporal_mean(ds_flux, "country_post_std")


# In[8]:


Intem_new_year_series = Intem_new_year.to_series()
Intem_new_year_df = Intem_new_year_series.to_frame()
Intem_new_year_df.reset_index(drop=False, inplace=True)
Intem_new_year_df.set_index("time", inplace=True)
Intem_new_year_df.index = pd.to_datetime(Intem_new_year_df.index, format='%Y')
#Intem_new_year_df.drop("country", axis=1,  inplace=True)

#Intem_new_year_series_std = Intem_new_year_std.to_series()
#Intem_new_year_std_df = Intem_new_year_series_std.to_frame()
#Intem_new_year_std_df.reset_index(drop=False, inplace=True)
#Intem_new_year_std_df.set_index("time", inplace=True)
#Intem_new_year_std_df.index = pd.to_datetime(Intem_new_year_df.index, format='%Y')
#Intem_new_year_std_df.drop("country", axis=1,  inplace=True)

#Intem_new_year_df["std"] = Intem_new_year_std_df[0]
#Intem_new_year_df["InTEM_ODS"] = (df_test["ODS"] / df_test["Total"]) *Intem_new_year_df[0]
#Intem_new_year_df["InTEM_ODS_std"] = (df_test["ODS"] / df_test["Total"]) *Intem_new_year_std_df[0]

#Intem_new_year_df


# In[10]:


df2012 = Intem_new_year[dict(time=0)]
#df2012


# In[11]:


df2012_series = df2012.to_series()
df2012_df = df2012_series.to_frame()
#df2012_df.reset_index(drop=False, inplace=True)
#df2012_df.set_index("time", inplace=True)
#df2012_df.index = pd.to_datetime(df2012_df.index, format='%Y')
df2012_df


# In[12]:


def read_intem_ch4(year, species):
    import pandas as pd
    from itertools import takewhile
    import plotly.graph_objects as go
    import base64
    import numpy as np
    import glob
    import xarray as xr
    
    csv = glob.glob(f"data/ch4-ukghg-{species}_EUROPE_{year}.nc")
    
    flux = xr.open_dataset(csv[0])
    
    return flux


# In[18]:


total2012 = read_intem_ch4("2012", "total")
total2012_df = xr.DataArray.to_dataframe(total2012.flux)
total2012


# In[14]:


agri2012 = read_intem_ch4("2012", "agric")
agri2012_df = xr.DataArray.to_dataframe(agri2012.flux)

domcom2012 = read_intem_ch4("2012", "domcom")
domcom2012_df = xr.DataArray.to_dataframe(domcom2012.flux)

energyprod2012 = read_intem_ch4("2012", "energyprod")
energyprod2012_df = xr.DataArray.to_dataframe(energyprod2012.flux)

indcom2012 = read_intem_ch4("2012", "indcom")
indcom2012_df = xr.DataArray.to_dataframe(indcom2012.flux)

indproc2012 = read_intem_ch4("2012", "indproc")
indproc2012_df = xr.DataArray.to_dataframe(indproc2012.flux)

natural2012 = read_intem_ch4("2012", "natural")
natural2012_df = xr.DataArray.to_dataframe(natural2012.flux)

offshore2012 = read_intem_ch4("2012", "offshore")
offshore2012_df = xr.DataArray.to_dataframe(offshore2012.flux)

othertrans2012 = read_intem_ch4("2012", "othertrans")
othertrans2012_df = xr.DataArray.to_dataframe(othertrans2012.flux)

roadtrans2012 = read_intem_ch4("2012", "roadtrans")
roadtrans2012_df = xr.DataArray.to_dataframe(roadtrans2012.flux)

waste2012 = read_intem_ch4("2012", "waste")
waste2012_df = xr.DataArray.to_dataframe(waste2012.flux)


# In[15]:


df2012_df["agri2012"] = agri2012_df["flux"]
df2012_df["total2012"] = total2012_df["flux"]
df2012_df["domcom2012"] = domcom2012_df["flux"]
df2012_df["energyprod2012"] = energyprod2012_df["flux"]
df2012_df["indcom2012"] = indcom2012_df["flux"]
df2012_df["indproc2012"] = indproc2012_df["flux"]
df2012_df["natural2012"] = natural2012_df["flux"]
df2012_df["offshore2012"] = offshore2012_df["flux"]
df2012_df["othertrans2012"] = othertrans2012_df["flux"]
df2012_df["roadtrans2012"] = roadtrans2012_df["flux"]
df2012_df["waste2012"] = waste2012_df["flux"]
df2012_df


# In[16]:


df2012_df.sum()


# In[17]:


df2012_df["agri_fraction"] = df2012_df["agri2012"] / df2012_df["total2012"]
df2012_df


# In[19]:


agri_fraction = agri2012.flux / total2012.flux
testag = xr.DataArray.to_dataframe(agri_fraction)
testag
testag.sum()
agri_fraction


# In[58]:


Intem_new_year


# In[59]:


agri_fraction


# In[57]:


agri_fract_intem = Intem_new_year * agri_fraction
agri_fract_intem_t0 = agri_fract_intem[dict(time=0)]
agri_fract_intem_t0.flux.plot()


# In[3]:


Intem_agri = xr.DataArray.to_dataframe(agri_flux.flux)
Intem_agri


# In[7]:


agrifp_t0 = agri_flux[dict(time=5)]
agrifp_t0.flux.plot()


# In[6]:





# In[ ]:




