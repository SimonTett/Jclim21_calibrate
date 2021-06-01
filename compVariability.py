"""
Compute variability for comparision with 1% runs.
Do this for 140 and 70 year timescales from all LongControl runs
"""
import pandas as pd
import os
import collections
import PaperLib
import iris
import numpy as np
import pathlib

def compVar(direct, file, verbose=False):
    """
    Compute 70-year and 140-year variance from 2nd order poly fits in 140 year chunks.
    :param direct: dictionary to read
    :param file: titles of variance to read
    :return: var of 70-year and 140-year fits.
    """

    if verbose:
        print("Processing %s in %s" % (file, direct))

    if file is 'NetFlux':
        ts = PaperLib.comp_net(direct)
    elif file is "ClearNetFlux":
        ts = PaperLib.comp_net(direct, clear=True)
    elif file is "CRF":
        ts = PaperLib.comp_crf(direct)
    else:
        try:
            ts = PaperLib.readFile(direct, file)
        except FileNotFoundError:
            print("Failed to load %s %s " % (direct, file))
            return None
    years = ts.coord('year').points
    # loop in chunks of 140 years extacting variance and fitting.
    nstep = 140
    result=[]
    for start_yr in range(years.min(),years.max()-nstep,nstep//4):
        #print("Start_yr is ",start_yr)
        lfn=lambda yr: start_yr <= yr.point < (start_yr + nstep)
        timeConstraint = iris.Constraint(coord_values={'year': lfn})
        sample = ts.extract(timeConstraint)
        result.append(PaperLib.comp_fit(sample, order=2, year=np.array([70,140])+start_yr))
    result=np.array(result)
    result = result[:,:,-1 ]# just want the last element which is gm.
    return result.var(0) # return variance




runInfo = pd.read_excel('OptClim_lookup.xlsx', index_col=0)  # read in meta variance on runs
worked = runInfo.Status == 'Succeeded' # ones that worked
runInfo=runInfo[worked]
longCtl = runInfo.LongControl[~runInfo.LongControl.isnull()]

titles=collections.OrderedDict([
        ('ts_t15.pp',("Global Average SAT","K")),
        ('ts_t15_land.pp',("Global Average Land SAT","K")),
        ('ts_sst.pp', ("Global Average SST", "K")),
        ('ts_ot.pp',("Volume Average Ocean Temperature",r"$^\circ$C")),
        ('ts_rtoaswu.pp',("RSR","Wm$^{-2}$")),
        ('ts_rtoalwu.pp', ("OLR", "Wm$^{-2}$")),
        ('NetFlux', ("Net", "Wm$^{-2}$")),
        ('CRF', ("Cloud Rad. Forcing", "Wm$^{-2}$")),
        ('ts_rtoaswuc.pp',("Clear Sky RSR","Wm$^{-2}$")),
        ('ts_rtoalwuc.pp', ("Clear Sky OLR", "Wm$^{-2}$")),
        ('ts_t50.pp',("Global Average 500 hPa T","K")),
        ('ts_rh50.pp', ("Global Average 500 hPa RH", "%")),
        #('ts_nao.pp',('NAO',"hPa",0.01)),
        ('ts_cet.pp',('CET','K')),
        ('ts_nino34.pp', ('CET', 'K')),
        ('ts_ice_extent.pp',("Ice Extent","10$^6$ km$^2$",1.0e-12)),
        ('ts_aice.pp', ("Ice Area", "10$^6$ km$^2$", 1.0e-12)),
        ('ts_snow_land.pp',("Snow Area","10$^6$ km$^2$",1.0e-12)),
        ('ts_nathcstrength.pp',("AMOC","Sv",-1)),
        ('ts_tppn_land.pp',("Land Precipitaiton","Sv",86400.)),
        ('ts_wme.pp',('Windmixing Energy',r'W,${-2}$')),
        ('ts_mld.pp',('Mixed Layer Depth','m')),
        ('ts_cloud.pp',('Cloud Fraction','')),
])

variance=collections.OrderedDict() # holding place for variance.
for indx,ctl in longCtl.items(): # iterate over long ctls
    ctlPath = pathlib.Path(ctl) # easier to have it as a path
    direct = PaperLib.OptClimPath/ 'grl17_coupled' / ctlPath / 'A' / (ctlPath.name + '.000100')
    variance[indx]=dict() # set up empty dict. For python 3.6ish and up dicts are ordered.
    for k,info in titles.items():
        try:
            scale=info[2]
        except IndexError:
            scale=1.0
        var = compVar(direct,k,verbose=False)
        if var is not None: # store the variance (0 = 70 year fit; 1 = 140 year fit)
            key=os.path.splitext(k)[0]
            variance[indx][key + '_yr70'] = var[0]
            variance[indx][key + '_yr140'] = var[1]
    print("Computed variance for %s"%(indx))
variance= pd.DataFrame(variance).T
# save the variance as a csv file
variance.to_csv(os.path.join(PaperLib.dataPath, 'internalVar.csv')) # save the variance as a CSV file.



