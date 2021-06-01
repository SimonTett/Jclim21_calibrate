import pathlib
import PaperLib
import collections
import readDataLib
import os.path
import matplotlib.pyplot as plt
import numpy as np
import iris

def forceFeedback(delta_CO2=None, CO2=2):
    """
    Compute forcing and feedback from 2xCO2 (or 4xCo2) experiments.
    Does by fitting straight line to flux vs radn.
    :param delta_CO2:
    :param CO2 -- how much CO2 is increased compared to Ctl. Default is 2.
    :return: hash containing:
    F_2xCO2 -- Forcing for 2XCO2
    ECS_2xCO2   -- Equlibrium Climate Sens for doubling CO2
    F_2xCO2_SW -- SW Forcing for  2XCO2
    F_2xCO2_SWC -- Clear Sky SW Forcing for  2XCO2
    F_2xCO2_LW -- LW Forcing for  2XCO2
    F_2xCO2_LWC -- Clear Sky LW Forcing for  2XCO2
    lambda_2xCO2 -- Feedback  for 2XCO2
    lambda_2xCO2C -- Clear Sky Feedback  for 2XCO2
    lambda_2xCO2_SW -- SW Feedback for  2XCO2
    lambda_2xCO2_SWC -- Clear Sky SW Feedback for  2XCO2
    lambda_2xCO2_LW -- LW Feedback for  2XCO2
    lambda_2xCO2_LWC -- Clear Sky LW Feedback for  2XCO2

    If delta_CO2 is None then all values are set to np.nan

    """
    import math

    def polyval_deriv(polyval):
        """
        Compute derivative of a polynomial.  Polynomial is k_n x^n, k_(n-1) x^(n-1), ....
        so deriv is n k_n x^(n-1), (n-1) k_(n-1 )x^n-2,....
            :param polyval -- polyval (produced from polyfit) to computer deriv.

        """
        deriv = polyval[0:-1] * np.arange(len(polyval) - 1, 0, -1)  # regression is highest order power first. So use usual rule for deriv.
        return deriv

    result = collections.OrderedDict()  # where we store results
    fitOrder = 1  # order of fit.
    gmConstraint = iris.Constraint(site_number=3.0)
    try:
        temp = delta_CO2['ts_t15']
    except KeyError:
        temp = None

    name = '_' + repr(CO2).rstrip(('.0')) + 'xCO2'

    if temp is None:
        result['F' + name] = 5.92 * math.log(CO2)  ## default value for HadCM3 from JMG scaled by CO2 conc.
        result['ECS' + name] = np.nan
        result['lambda' + name] = np.nan
    else:  # got the data we need!
        temp = temp.extract(gmConstraint)  # global-mean temperature
        netFlux = delta_CO2['NetFlux']
        clear_netFlux = delta_CO2['ClrNetFlux']
        reg = np.polyfit(temp.data, netFlux.data, fitOrder)
        reg_clear = np.polyfit(temp.data, clear_netFlux.data, fitOrder)
        try:# work out the forcing for NXCO2
            r = np.polyval(reg, 0.0)[0]
            r_clear = np.polyval(reg_clear, 0.0)[0]
        except IndexError:
            r = np.polyval(reg, 0.0)
            r_clear = np.polyval(reg_clear, 0.0)
        result['F' + name] = r
        result['Fclear' + name] = r_clear
        roots = np.roots(reg.ravel())  # and the ECS (when net flux is zero)
        ecs = roots[(roots > 0.0) & np.isreal(roots)].min()
        result['ECS' + name] = ecs
        # to get feedbacks need deriv at ECS wrt temp. So just take the polyval.
        result['lambda' + name] = -np.polyval(polyval_deriv(reg), ecs)
        result['lambda' + name+'_C'] = -np.polyval(polyval_deriv(reg_clear), ecs) # -ve as lambda is change in outward flux per K.


    # now to compute SW/LW/SWC/LWC terms
    for index, name2 in zip(['ts_rtoalwu', 'ts_rtoaswu', 'ts_rtoalwuc', 'ts_rtoaswuc'],
                            ['LW', 'SW', 'LWC', 'SWC']):
        fName = 'F' + name + '_' + name2
        lName = 'lambda' + name + '_' + name2
        if temp is not None:
            reg = np.polyfit(temp.data, delta_CO2[index].data, fitOrder)
            result[fName] = -np.polyval(reg, 0.0)  # work out the forcing for XCO2. Flip sign as fluxes are outgoing.
            result[lName] = np.polyval(polyval_deriv(reg), ecs)   # feedback at 2xCO2 per K warming.

        else:  # no data so set to Nan
            result[fName] = np.nan
            result[lName] = np.nan

    # strip all numpy stuff out -- causes trouble when making pandas series/dataframes..
    for key, v in result.items():
        if isinstance(v, np.ndarray) and len(v) == 1: result[key] = v[0]  # just first value.
    return result

fourTimesCO2Dir = pathlib.Path(PaperLib.time_cache)/'xnilb.000100'
twoTimesCO2Dir = pathlib.Path(PaperLib.time_cache)/'xncyb.000100'
ctlDir = pathlib.Path(PaperLib.time_cache)/'xhivd.000100'

titles=collections.OrderedDict([
        ('ts_t15.pp',("Global Average SAT","K")),
        ('ts_t15_land.pp',("Global Average Land SAT","K")),
        ('ts_sst.pp', ("Global Average SST", "K")),
        ('ts_ot.pp',("Volume Average Ocean Temperature",r"$^\circ$C")),
        ('ts_rtoaswu.pp',("RSR","Wm$^{-2}$")),
        ('ts_rtoalwu.pp', ("OLR", "Wm$^{-2}$")),
        ('NetFlux', ("Net", "Wm$^{-2}$")),
        ('ClrNetFlux', ("Clear-Sky Net", "Wm$^{-2}$")),
        ('ts_rtoaswuc.pp',("Clear Sky RSR","Wm$^{-2}$")),
        ('ts_rtoalwuc.pp', ("Clear Sky OLR", "Wm$^{-2}$")),
        ('ts_t50.pp',("Global Average 500 hPa T","K")),
        ('ts_rh50.pp', ("Global Average 500 hPa RH", "%")),
        ('ts_nao.pp',('NAO',"hPa",0.01)),
        #('ts_soi.pp',('SOI',"hPa",0.01)),
        ('ts_cet.pp',('CET','K')),
        ('ts_nino34.pp', ('CET', 'K')),
        ('ts_ice_extent.pp',("Ice Extent","10$^6$ km$^2$",1.0e-12)),
        ('ts_aice.pp', ("Ice Area", "10$^6$ km$^2$", 1.0e-12)),
        ('ts_snow_land.pp',("Snow Area","10$^6$ km$^2$",1.0e-12)),
        ('ts_nathcstrength.pp',("AMOC","Sv",-1)),
        ('ts_tppn_land.pp',("Land Precipitation","mm/day",86400.)),
        ('ts_wme.pp',('Windmixing Energy',r'W,${-2}$')),
        ('ts_mld.pp',('Mixed Layer Depth','m')),
        # ('t15.pp', ('1.5m Temperature', 'K')),
        # ('t15_land.pp', ('1.5m Land Temperature', 'K')),
        # ('t15_min_land.pp', ('1.5m Ann Min Land Temperature', 'K')),
        # ('t15_max_land.pp', ('1.5m Ann Max Land Temperature', 'K')),
    # analysis only really works with timeseries.
        #('precip_land.pp', ('Land Precipitation', 'mm/day', 86400.)),
        #('ot.pp',('Ocean Temperature', 'C')), # TODO fix -- read does not work.
        #('os.pp',('Ocean Salinity', 'psu'))
                            ])

ts4xCO2=dict()
ts2xCO2 = dict()
for name in titles.keys():
    key = os.path.splitext(name)[0] # remove the .pp bit.
    ts4xCO2[key] = readDataLib.compDelta(ctlDir, fourTimesCO2Dir, name)
    ts2xCO2[key] = readDataLib.compDelta(ctlDir, twoTimesCO2Dir, name)

forcing4 = forceFeedback(ts4xCO2,CO2=4)
forcing4_old = PaperLib.forceFeedback(ts4xCO2,CO2=4)

forcing = forceFeedback(ts2xCO2,CO2=2)
forcing_old = PaperLib.forceFeedback(ts2xCO2,CO2=2)


## plot data
fig,axis = plt.subplots(2,2,num='scatter',sharex='all',clear=True)
for ax,title in zip(axis.flatten(),['NetFlux','ClrNetFlux','ts_rtoalwu','ts_rtoaswu']):
    ax.scatter(ts4xCO2['ts_t15'].data,ts4xCO2[title].data,color='red',marker='*')
    ax.scatter(ts2xCO2['ts_t15'].data, ts2xCO2[title].data,color='blue',marker='o')
    ax.set_title(title)
axis[0,0].set_xlim(0,6.5)

