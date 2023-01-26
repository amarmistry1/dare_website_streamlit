###############################################################################
def read_invent(country, year, species):
    import pandas as pd
    from itertools import takewhile
    import plotly.graph_objects as go
    import base64
    import numpy as np
    import glob
    
    xls = glob.glob(f"data/{country}_{year}*.xlsx")
    df = pd.read_excel(xls[0], sheet_name=species, header=4)
    
    df.set_index("GREENHOUSE GAS SOURCE AND SINK CATEGORIES", inplace=True)
    df.dropna(inplace=True)

    df.drop(labels = ["Base year(1)", "Change from base to latest reported year"],axis=1,inplace = True)
    
    df1 = df.apply(pd.to_numeric, errors='coerce')
    
    if species == "Table10s3":
        df1 = (df1 * 28) / 1000
        df1.dropna(inplace=True)
    elif species == "Table10s4":
        df1 = (df1 * 265) / 1000
        df1.dropna(inplace=True)
    
    
    return df1

################################################################################
def read_invent_ODS(year, species):
    import pandas as pd
    from itertools import takewhile
    import plotly.graph_objects as go
    import base64
    import numpy as np
    import glob
    
    csv = glob.glob(f"data/UK_NIR_{year}_{species}.csv")  
        
    df = pd.read_csv(csv[0], index_col="GREENHOUSE GAS SOURCE AND SINK CATEGORIES")
    
    ODS = df.loc["F.  Product uses as substitutes for ODS(2)"]    
    
    dfODS = pd.DataFrame(ODS)
    dfODS.set_index(pd.to_datetime(dfODS.index, format='%Y'), inplace=True)

    if species == "HFC-23":
        dfODS = (dfODS * 14800) / 1000
        dfODS.dropna(inplace=True)
    elif species == "HFC-32":
        dfODS = (dfODS * 675) / 1000
        dfODS.dropna(inplace=True)
    elif species == "HFC-43-10mee":
        dfODS.replace("NA,NO", 0, inplace=True)
        dfODS = (dfODS * 1640) / 1000
        dfODS.dropna(inplace=True) 
    elif species == "HFC-125":
        dfODS = (dfODS * 3500) / 1000
        dfODS.dropna(inplace=True) 
    elif species == "HFC-134a":
        dfODS = (dfODS * 1430) / 1000
        dfODS.dropna(inplace=True) 
    elif species == "HFC-143a":
        dfODS = (dfODS * 4470) / 1000
        dfODS.dropna(inplace=True) 
    elif species == "HFC-152a":
        dfODS = (dfODS * 124) / 1000
        dfODS.dropna(inplace=True) 
    elif species == "HFC-227ea":
        dfODS = (dfODS * 3220) / 1000
        dfODS.dropna(inplace=True) 
    elif species == "HFC-245fa":
        dfODS = (dfODS * 1030) / 1000
        dfODS.dropna(inplace=True) 
    elif species == "HFC-365mfc":
        dfODS = (dfODS * 794) / 1000
        dfODS.dropna(inplace=True) 
    
    return dfODS

########################################################################

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
    
#########################################################################

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:23:48 2014
@author: chxmr
"""
import numpy as np

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

    