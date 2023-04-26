#!/usr/bin/env python
# coding: utf-8



# In[39]:

import streamlit as st
import pandas as pd
from itertools import takewhile
import plotly.graph_objects as go
import plotly.express as px
import base64
import numpy as np
import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy import loadtxt

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#from fig_mods.nextstep import areagrid

def areagrid(lat, lon):
        
  re=6367500.0	#radius of Earth in m
  
  dlon=abs(np.mean(lon[1:] - lon[0:-1]))*np.pi/180.
  dlat=abs(np.mean(lat[1:] - lat[0:-1]))*np.pi/180.
  theta=np.pi*(90.-lat)/180.
  
  area=np.zeros((len(lat), len(lon)))
  
  for latI in range(len(lat)):
    if theta[latI] == 0. or np.isclose(theta[latI], np.pi):
      area[latI, :]=(re**2)*abs(np.cos(dlat/2.)-np.cos(0.))*dlon
    else:
      lat1=theta[latI] - dlat/2.
      lat2=theta[latI] + dlat/2.
      area[latI, :]=((re**2)*(np.cos(lat1)-np.cos(lat2))*dlon)

  return area


ds_flux = xr.open_dataset("data/flux_MetOffice-InTEM_ch4_MHD_TAC_RGL_TTA_BSD_HFD_CBW_WAO_JFJ_CMN_ZEP_TRN_LUT_2012-2021_Dec2022.nc")

## Set the longitude and latitude to only include the UK;

min_lon = -12
min_lat = 50

max_lon = 2 
max_lat = 62 

ds_flux = ds_flux.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))


# In[4]:


lat = np.arange(50.041, 61.975002, 0.234)
lon = np.arange(-11.66, 1.716001, 0.352)
area = areagrid(lat, lon)
#area


# In[5]:


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


# In[6]:


Intem_new_year = weighted_temporal_mean(ds_flux, "flux_prior")
Intem2012 = Intem_new_year[dict(time=0)] * area
Intem2013 = Intem_new_year[dict(time=1)] * area
Intem2014 = Intem_new_year[dict(time=2)] * area
Intem2015 = Intem_new_year[dict(time=3)] * area


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
        
    return cropped_ds1


# In[9]:


test2012 = read_invent_ch4("2012", "total")
#test2012


# In[10]:


def plot_sector(year):
    
    fig, axs = plt.subplots(4, 3, figsize = (13, 13), subplot_kw={'projection':cartopy.crs.PlateCarree()})
    
    species = ["total", "agric", "domcom", "energyprod", "indcom", "indproc", "offshore", "othertrans", "roadtrans", "waste"]
    
    fontsizes = {"title":10, "labels": 10, "axis":10}
    
    for name, ax in zip(species, axs.flatten()):
        
        a = read_invent_ch4(year, name)
        b = read_invent_ch4(year, "total")
        
        fract = 100*(a.flux / b.flux)
    
        fract = np.clip(fract, 0, 100)
        
        test = fract.reindex_like(Intem2012, method='nearest', tolerance=0.01)
        
        if year == "2012":
            data = test/100 * Intem2012
            x = 'Reds'
        elif year == "2013":
            data = test/100 * Intem2013
            x = 'Blues'
        elif year == "2014":
            data = test/100 * Intem2014
            x = 'Greens'
        elif year == "2015":
            data = test/100 * Intem2015
            x = 'Purples'
                
        a = ax.pcolormesh(data.lon, data.lat, data, cmap=plt.cm.get_cmap(x))
        ax.set_extent([-12,3, 49.9,60], crs=cartopy.crs.PlateCarree())        
        ax.coastlines(resolution='50m', color='black', linewidth=1)
        
        ax.set_xticks([-12, -8, -4, 0], crs=cartopy.crs.PlateCarree())
        lon_formatter = LongitudeFormatter(number_format='.1f',
                                    degree_symbol='',
                                    dateline_direction_label=True)
        ax.xaxis.set_major_formatter(lon_formatter)        
        # set y-ticks for the plot, and format          
        ax.set_yticks([50, 54, 58], crs=cartopy.crs.PlateCarree())
        lat_formatter = LatitudeFormatter(number_format='.1f',  degree_symbol='',)
        ax.yaxis.set_major_formatter(lat_formatter)             
        ax.tick_params(axis='both', which='major', labelsize=fontsizes["axis"]) 
        
        fig.colorbar(a, ax=ax, pad=0.05, shrink=0.7)
        
        ax.set_title("2013" +"\n"+ name, fontsize = fontsizes["title"])
        ax.set_ylabel("Latitude (degrees)", fontsize = fontsizes["labels"])
        ax.set_xlabel("Longitude (degrees)", fontsize = fontsizes["labels"])
        
    fig.tight_layout()


# In[48]:

    
fig, axs = plt.subplots(4, 3, figsize = (13, 13), subplot_kw={'projection':cartopy.crs.PlateCarree()})
    
species = ["total", "agric", "domcom", "energyprod", "indcom", "indproc", "offshore", "othertrans", "roadtrans", "waste"]
    
fontsizes = {"title":10, "labels": 10, "axis":10, "suptitle":16} 
    
for name, ax in zip(species, axs.flatten()):
        
    a = read_invent_ch4("2013", name)
    b = read_invent_ch4("2013", "total")
        

    fract = 100*(a.flux / b.flux)

    fract = np.clip(fract, 0, 100)

    test = fract.reindex_like(Intem2012, method='nearest', tolerance=0.01)
        
   
    test1 = test * Intem2013
    data = test1 / Intem2013 #* 100
    x = 'Reds'

                
    a = ax.pcolormesh(data.lon, data.lat, data, cmap=plt.cm.get_cmap(x))
    ax.set_extent([-12,3, 49.9,60], crs=cartopy.crs.PlateCarree())        
    ax.coastlines(resolution='50m', color='black', linewidth=1)
        
    ax.set_xticks([-12, -8, -4, 0], crs=cartopy.crs.PlateCarree())
    lon_formatter = LongitudeFormatter(number_format='.1f',
                                    degree_symbol='',
                                    dateline_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)        
        # set y-ticks for the plot, and format          
    ax.set_yticks([50, 54, 58], crs=cartopy.crs.PlateCarree())
    lat_formatter = LatitudeFormatter(number_format='.1f',  degree_symbol='',)
    ax.yaxis.set_major_formatter(lat_formatter)             
    ax.tick_params(axis='both', which='major', labelsize=fontsizes["axis"]) 
    
    ax.set_title("2013" +"\n"+ name, fontsize = fontsizes["title"])
    fig.colorbar(a, ax=ax, pad=0.01, shrink=0.9)
    ax.set_ylabel("Latitude (degrees)", fontsize = fontsizes["labels"])
    ax.set_xlabel("Longitude (degrees)", fontsize = fontsizes["labels"])
    
fig.suptitle("Percentage of Total Methane Emissions (CH\u2084) by Sector (2013)", fontsize=fontsizes["suptitle"], ha="center")
    
    #fig.subplots_adjust(hspace=0.5, wspace=0.5)
        
fig.tight_layout()

st.title("Methodology")

st.markdown('''
This section will discuss the methodology surrounding the 2012-2015 data. 
 
It is important to note that DARE_UK does not currently have the capabilities to record emission contributions from individual economic sectors, although we are currently working towards this. 
 
As such, the sectoral emissions presented here are estimated using emissions data published in the UK Greenhouse Gas Inventory. 
 
By analysing data published in the UK GHG Inventory, we are able to estimate the percentage of average yearly emissions emitted by each economic sector from each individual grid-cell. We have then applied these same percentages from each grid-cell to the data published by the DARE_UK network between 2012-2022, as shown below.  
 
These sectors are then further categorised into 4 major economic sectors as such:
     
* Energy and Transport = Domestic combustion (domcom), Energy Production (energyprod), Offshore (offshore), Road Transport (roadtrans) and Other Transport (othertrans)
* Industiral Processes = Industrial combustion (indcom) and Industrial Production (indprod)
* Agriculture = Agriculture (agric)
* Waste = Waste  (waste)
''')

    
st.pyplot(fig, use_container_width=True)


# In[49]:


#plot_sector_perc("2013")

