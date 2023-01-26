# ****************************************************************************
# Methane source sector correspondence used for embedding UKGHG emissions,
# that are aggregated by SNAP sector, into EDGAR v7.0 emissions, that use
# the IPCC sector specifications.
# 
# Embedded emissions follow the UKGHG sector apportionment - we cannot,
# for example, disaggregate 'waste' emissions by landfill, incineration etc.
# 
# Naming conventions follow the UK NAEI/UKGHG sectoral specifications.
# 
# Correspondence between sectors follows Saboya et al. (2022) - see for 
# more information on how this is done.
#
# Created: 9 Dec. 2022 
# Contact: Eric Saboya, eric.saboya@bristol.ac.uk
# ****************************************************************************
# 
#  UKGHG sector   <---->   EDGAR sectors
#  --------------------------------------------
#  agric          <---->   AGS,AWB,ENF,MNM
#  domcom         <---->   RCO
#  energyprod     <---->   ENE
#  indcom         <---->   IND
#  indproc        <---->   CHE,IRO
#  natural        <---->   none
#  offshore       <---->   REF_TRF,PRO
#  othertrans     <---->   TNR_Aviation_CDS,TNR_Aviation_CRS,TNR_Aviation_LTO,TNR_Other,TNR_Ship
#  roadtrans      <---->   TRO_noRES
#  waste          <---->   SWD_INC,SWD_LDF,WWT
# ****************************************************************************

