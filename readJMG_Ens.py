"""
Read and compute summary stats from Jonathan's IC ensemble
Changes from my case:
1) Have reference as mean -- might be worth asking JMG for time dependence.. (and also ask him how he perturbed runs
2) Runs are for 100 years rather than my 40.
"""


import PaperLib
import os
import glob
import collections
import iris
import pandas as pd
import numpy as np
import readDataLib

use2xCO2=True

if use2xCO2:
    ensDirs = glob.glob(os.path.join(PaperLib.time_cache, 'xjwgy?.000100'))
    outFile = "JMG_2xCO2_ens_data.csv"
    CO2=2.0
    CO2title='_2xCO2'
    print("Only got 20 years for 2xCO2 runs")
    #raise NotImplementedError("Not enough data for 2xCO2")
    
else:
    ensDirs=glob.glob(os.path.join(PaperLib.time_cache,'xfqqp?.000100'))
    outFile = "JMG_ens_data.csv"
    CO2=4.0
    CO2title='_4xCO2'

refDir=os.path.join(PaperLib.time_cache,'xfqqe.014000')


def readData(name,dir,scale):
    """
    Read data in
    :param name: name of var to read in.
    :return:
    """
    if name is 'NetFlux':
        ts = PaperLib.comp_net(dir)
    elif name is 'ClrNetFlux':
        ts = PaperLib.comp_net(dir,clear=True)
    else:  # just read the data
        file=os.path.join(dir, name)
        try:
            ts = PaperLib.read_pp(file) * scale
        except IOError:
            print ("Failed to read ",file)
            ts = None


    return ts

titles=collections.OrderedDict([
        ('ts_cet.pp', ('CET', 'K')),
        ('ts_t15.pp',("Global Average SAT","K")),
        ('ts_sst.pp', ("Global Average SST", "K")),
        ('ts_ot.pp',("Volume Average Ocean Temperature",r"$^\circ$C")),
        ('ts_rtoaswu.pp',("RSR","Wm$^{-2}$")),
        ('ts_rtoalwu.pp', ("OLR", "Wm$^{-2}$")),
        ('NetFlux', ("Net", "Wm$^{-2}$")),
        ('ClrNetFlux', ("Clear Sky Net", "Wm$^{-2}$")),
        ('ts_rtoaswuc.pp',("Clear Sky RSR","Wm$^{-2}$")),
        ('ts_rtoalwuc.pp', ("Clear Sky OLR", "Wm$^{-2}$")),
        ('ts_t50.pp',("Global Average 500 hPa T","K")),
        ('ts_nao.pp',('NAO',"hPa",0.01)),
        ('ts_tppn.pp',("Land Precipitation","kg/m^2/sec")),
        ('ts_aice.pp', ("Ice Area", "10$^6$ km$^2$", 1.0e-12)),
        ('ts_nathcstrength.pp',("AMOC","Sv",-1)),

    ])

# read in reference data
refData=collections.OrderedDict()
for file,values in titles.items():
    try:  # work out scaling.
        scale = values[2]
    except IndexError:
        scale = 1.0
    ref=readData(file,refDir,scale)
    key = os.path.splitext(file)[0]  # remove the .pp bit.
    refData[key]=ref


# iterate over ens members

allData = collections.OrderedDict()
for member in ensDirs:
    NF = PaperLib.comp_net(member)# compute the net flux.
    min_yr = NF.coord('year').points.min()
    lfn = lambda yr: min_yr <= yr.point < (min_yr + 40)
    timeConstraint = iris.Constraint(coord_values={'year': lfn})

    NF = NF.extract(timeConstraint)
    ensMemberValues = collections.OrderedDict()  #
    ensMemberDelta = collections.OrderedDict()  #
    eqValues= collections.OrderedDict()
    for file, values in titles.items():

        try:  # work out scaling.
            scale = values[2]
        except IndexError:
            scale = 1.0
        key = os.path.splitext(file)[0]  # remove the .pp bit.
        ts=readData(file, member, scale)# .extract(iris.Constraint(lambda: y ))
        if ts is not None: # extract 40 years worth.
            ts = ts.extract(timeConstraint)
        else:
            print("Failed to read in %s",file)

        ensMemberValues[key] = ts

        # something weird with some datasets so need to act on the data part..
        delta = ts.copy()
        delta.data = (ts.data - refData[key].data)
        # extract gm value where we have it.
        if (delta.ndim > 1) and (delta.shape[1] > 1):
            gmConstraint = iris.Constraint(site_number=3.0)
            # and only want the first 40 years (at most -- if run less than 40 years will get that)
            delta = delta.extract(gmConstraint)
        # delta = ts-refData[key] # remove long-term mean value. and crashes...IRIS sucks... doing ts -2 crashes...

        ensMemberDelta[key]= delta
        try:
            reg = np.polyfit(NF.data.squeeze(), delta.data, 1)
        except TypeError:
            print("Failed %s file:%s" % (member, file))
            raise  # re-raise the error.
        eqValue = np.polyval(reg, 0.0)

        eqValues[key+CO2title] = eqValue

    allData[member]=eqValues
    allData[member].update(PaperLib.forceFeedback(delta_CO2=ensMemberDelta, CO2=CO2)) # compute ECS, feedback and forcings
# now let's make a dataframe w
ICens=pd.DataFrame(allData).T

# and save it.
ICens.to_csv(os.path.join(PaperLib.dataPath, outFile))





