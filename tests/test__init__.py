################################################################################
from pathlib import Path

def test_path():

    return Path(__file__).parents[1]

################################################################################
def test_load_ch4():
    import pandas as pd
    from itertools import takewhile
    import plotly.graph_objects as go
    import base64
    import numpy as np
    
    xlsCH4 = './../data/GBR_2022_2020_10052022_142545.xlsx'
    dfCH4 = pd.read_excel(xlsCH4, 'Table10s3', header=4)
    dfCH4.set_index("GREENHOUSE GAS SOURCE AND SINK CATEGORIES", inplace=True)
    dfCH4.dropna(inplace=True)

    dfCH4.drop(labels = ["D.  Non-energy products from fuels and solvent use", "G.  Other product manufacture and use ", "C.  Rice cultivation", "D.  Agricultural soils", "E.  Prescribed burning of savannas", "F.  Other land", "E.  Other ","6.  Other (as specified in summary 1.A)", "International bunkers", "Navigation", "Multilateral operations", "Aviation"], axis=0, inplace = True)

    dfCH4.drop(labels = ["Base year(1)", "Change from base to latest reported year"],axis=1,inplace = True)
    
    dfCH4_CO2e = dfCH4 / 1000 * 28

    return dfCH4_CO2e

################################################################################
def test_load_n2o():
    import pandas as pd
    from itertools import takewhile
    import plotly.graph_objects as go
    import base64
    import numpy as np
    
    xlsN2O = './../data/GBR_2022_2020_10052022_142545.xlsx'
    dfN2O = pd.read_excel(xlsN2O, 'Table10s4', header=4)
    dfN2O.set_index("GREENHOUSE GAS SOURCE AND SINK CATEGORIES", inplace=True)
    dfN2O.dropna(inplace=True)

    dfN2O.drop(labels = ["D.  Non-energy products from fuels and solvent use",  "E.  Prescribed burning of savannas", "F.  Other land", "E.  Other ","6.  Other (as specified in summary 1.A)", "International bunkers", "Navigation", "Multilateral operations", "Aviation", "Indirect N2O", "H.  Other "], axis=0, inplace = True)
    dfN2O.drop(labels = ["Base year(1)", "Change from base to latest reported year"],axis=1,inplace = True)
    
    dfN2O_CO2e = dfN2O / 1000 * 265
    
    return dfN2O_CO2e

################################################################################
def test_load_fgas():
    import pandas as pd
    from itertools import takewhile
    import plotly.graph_objects as go
    import base64
    import numpy as np
    
    invent_fgas = pd.read_csv('./../data/UK_NIR_2022_co2e_all_gases_yearly.csv')
    invent_fgas.set_index("year", inplace=True)
    
    invent_fgas_1 = invent_fgas / 1000
    invent_fgas_1["Total"] = invent_fgas_1.sum(axis=1)
    invent_fgas_1.index = pd.to_datetime(invent_fgas_1.index, format='%Y')

    hfc23 = pd.read_csv('./../data/UK_NIR_2022_HFC-23.csv')
    hfc23.set_index("GREENHOUSE GAS SOURCE AND SINK CATEGORIES", inplace=True)
    hfc23.fillna(0, inplace=True)

    hfc32 = pd.read_csv('./../data/UK_NIR_2022_HFC-32.csv')
    hfc32.set_index("GREENHOUSE GAS SOURCE AND SINK CATEGORIES", inplace=True)
    hfc32.fillna(0, inplace=True)

    hfc4310mee = pd.read_csv('./../data/UK_NIR_2022_HFC-43-10mee.csv')
    hfc4310mee.set_index("GREENHOUSE GAS SOURCE AND SINK CATEGORIES", inplace=True)
    hfc4310mee.replace("NA,NO", 0, inplace=True)

    hfc125 = pd.read_csv('./../data/UK_NIR_2022_HFC-125.csv')
    hfc125.set_index("GREENHOUSE GAS SOURCE AND SINK CATEGORIES", inplace=True)
    hfc125.fillna(0, inplace=True)

    hfc134a = pd.read_csv('./../data/UK_NIR_2022_HFC-134a.csv')
    hfc134a.set_index("GREENHOUSE GAS SOURCE AND SINK CATEGORIES", inplace=True)
    hfc134a.fillna(0, inplace=True)

    hfc143a = pd.read_csv('./../data/UK_NIR_2022_HFC-143a.csv')
    hfc143a.set_index("GREENHOUSE GAS SOURCE AND SINK CATEGORIES", inplace=True)
    hfc143a.fillna(0, inplace=True)

    hfc152a = pd.read_csv('./../data/UK_NIR_2022_HFC-152a.csv')
    hfc152a.set_index("GREENHOUSE GAS SOURCE AND SINK CATEGORIES", inplace=True)
    hfc152a.fillna(0, inplace=True)

    hfc227ea = pd.read_csv('./../data/UK_NIR_2022_HFC-227ea.csv')
    hfc227ea.set_index("GREENHOUSE GAS SOURCE AND SINK CATEGORIES", inplace=True)
    hfc227ea.fillna(0, inplace=True)

    hfc245fa = pd.read_csv('./../data/UK_NIR_2022_HFC-245fa.csv')
    hfc245fa.set_index("GREENHOUSE GAS SOURCE AND SINK CATEGORIES", inplace=True)
    hfc245fa.fillna(0, inplace=True)

    hfc365mfc = pd.read_csv('./../data/UK_NIR_2022_HFC-365mfc.csv')
    hfc365mfc.set_index("GREENHOUSE GAS SOURCE AND SINK CATEGORIES", inplace=True)
    hfc365mfc.fillna(0, inplace=True)
    
    
    ODS = (hfc23.loc["F.  Product uses as substitutes for ODS(2)"] / 1000 *14800) + (hfc32.loc["F.  Product uses as substitutes for ODS(2)"] / 1000 *675) +(hfc4310mee.loc["F.  Product uses as substitutes for ODS(2)"] / 1000 *1640) +(hfc125.loc["F.  Product uses as substitutes for ODS(2)"] / 1000 *3500) +(hfc134a.loc["F.  Product uses as substitutes for ODS(2)"] / 1000 *1430) +(hfc143a.loc["F.  Product uses as substitutes for ODS(2)"] / 1000 *4470) +(hfc152a.loc["F.  Product uses as substitutes for ODS(2)"] / 1000 *124) +(hfc227ea.loc["F.  Product uses as substitutes for ODS(2)"] / 1000 *3220) +(hfc245fa.loc["F.  Product uses as substitutes for ODS(2)"] / 1000 *1030) +(hfc365mfc.loc["F.  Product uses as substitutes for ODS(2)"] / 1000 *794)
    
    dfODS = pd.DataFrame(ODS)
    dfODS.set_index(pd.to_datetime(dfODS.index, format='%Y'), inplace=True)
           
    dfODS["ODS"] = dfODS["F.  Product uses as substitutes for ODS(2)"] / 1000
    
    dfODS["Total"] = invent_fgas_1["Total"]
    
    return dfODS
    
    
################################################################################
def test_read_invent():
    import pandas as pd
    from itertools import takewhile
    import plotly.graph_objects as go
    import base64
    import numpy as np
    
    species = "Table10s3"
    
    xls = './../data/GBR_2022_2020_10052022_142545.xlsx'
    df = pd.read_excel(xls, sheet_name=species, header=4)    

    df.set_index("GREENHOUSE GAS SOURCE AND SINK CATEGORIES", inplace=True)
    df.dropna(inplace=True)

    df.drop(labels = ["Base year(1)", "Change from base to latest reported year"],axis=1,inplace = True)
    
    return df

    assert(type(df).__name__ == 'DataFrame')

################################################################################

def test_read_intem(species):

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

