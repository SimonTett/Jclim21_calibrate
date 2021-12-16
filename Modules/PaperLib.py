"""
module containing useful functions and defaults used to produce plots for paper.
"""
interactive = True
pltType = 'paper'  # paper to make paper. talk to make 300 dpi pngs
import collections
import functools
import inspect
import os
import re
import pathlib
import seaborn as sns

import scipy.stats
# gives us cache function -- which is used to cache reads.
# and partial functions.
from dask.cache import Cache

# setup cache stuff.
cache = Cache(2e9)  # 1 gbyte of cache.
cache.register()  # now use it and this really does speed things up!
os.environ['HOME'] = 'M:'
try:
    from importlib import reload  # so we can reload
except ImportError:
    pass

import iris

import matplotlib as mpl
# mpl.use('QT4Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import iris.coord_categorisation
import iris.fileformats.pp as pp
import iris.util

# import StudyConfig

if interactive and not plt.isinteractive():
    plt.ion()  # now we are interactive.
    print("Warning -- in interactive mode")

os.environ['OPTCLIMTOP'] = r'M:\analyse\optclim\CurrentCode'  # really should be set in the env.

# rename -- to support old names going to new names!
column_rename = {'netrad_global': 'Net Flux',
                 'RADB': 'Net Flux',
                 'netflux_global': 'Net Flux',
                 'OutLW_NHX': 'olr_nhx',
                 'OutSW_NHX': 'rsr_nhx',
                 'LAt_NHX': 'lat_nhx',
                 'LPrecip_NHX': 'lprecip_nhx',
                 'Mslp_NHX': 'mslp_nhx_dgm',
                 'Temp_NHX': 'temp@500_nhx',
                 'RH_NHX': 'rh@500_nhx',
                 'OutLW_TROP': 'olr_tropics',
                 'OutSW_TROP': 'rsr_tropics',
                 'LAt_TROP': 'lat_tropics',
                 'LPrecip_TROP': 'lprecip_tropics',
                 'Mslp_TROP': 'mslp_tropics_dgm',
                 'Temp_TROP': 'temp@500_tropics',
                 'RH_TROP': 'rh@500_tropics',
                 'OutLW_SHX': 'olr_shx',
                 'OutSW_SHX': 'rsr_shx',
                 'LAt_SHX': 'lat_shx',
                 'LPrecip_SHX': 'lprecip_shx',
                 'Temp_SHX': 'temp@500_shx',
                 'RH_SHX': 'rh@500_shx'}
# set up rename to take new names to old names (which then get taken back to new names)
varRename = dict()
for v, k in column_rename.items():
    varRename[v] = k
varRename['RADB'] = 'netflux_global'

dir_rewrite = {'/exports/work/geos_cesd/OptClim':
                   'm:/analyse/optclim/data_files'}

import platform

if platform.uname()[1] in ['GEOS-D-0892']:  # where we need to get data from other places
    OptClimPath = pathlib.Path(r'\\csce.datastore.ed.ac.uk\csce\geos\groups\OPTCLIM')  # Datastore.
elif 'geos.ed.ac.uk' in platform.uname()[1]:
    OptClimPath = pathlib.Path(r'/exports/csce/datastore/geos/groups/OPTCLIM')  # Datastore.
else:
    OptClimPath = pathlib.Path('c:/users/stett2/data/optclim')

DFOLSpath = OptClimPath/'DFOLS19'


rootData = OptClimPath / 'data_files'
time_cache = os.path.join(os.path.expanduser("~"), 'time_cache')  # where midl processed data lives
dataPath = pathlib.Path('data')  ## where processed data files live.
# dataPath = rootData
# set up std graphical look and feel. Change here to change all figures generated.
mpl.rcdefaults()  # restore default values
plt.style.use('ggplot')

if pltType == 'talk':
    sns.set_context("talk", font_scale=1.0)
    defFigType = '.png'  # Default type for all figures
    markersize = 10
    linewidth = 2.5
    dpi = 300
    # mpl.rcParams['font.size'] = 20
    mpl.rcParams['savefig.dpi'] = 300
else:
    sns.set_context("paper")  ##, font_scale=1.0)
    defFigType = '.pdf'  # Default type for all figures
    markersize = 8
    linewidth = 2
    dpi = None

# graphics
sns.set_style("whitegrid")
fsize = (11.8, 7.2)
col2xCO2 = 'green'
col4xCO2 = 'blue'

def dfInv(df, rcond=1e-15, pseudoInv=True):
    """
    Compute inverse using np.linalg.pinv wrapping things backup as a dataframe...
    :param df: dataframe
    :param rcond -- rcond value for pinv
    :param pseudoInv (Defualt True) if True use Pseudoinv else use normal inverse
    :return: return inverse
    """

    if pseudoInv:
        inv = np.linalg.pinv(df, rcond=rcond)
    else:
        inv = np.linalg.inv(df)
    inv = pd.DataFrame(inv, index=df.index, columns=df.columns)
    return inv


def dfEig(df):
    """
    Compute eigenvalues using np.linalg.eig wrapping things backup as a dataframe...
    :param df: dataframe
    :return: return eigenValues and eigenVectors as dataframes
    """

    eigenValues, eigenVectors = np.linalg.eig(df)
    col = np.arange(0, len(eigenValues))
    eigenVectors = pd.DataFrame(eigenVectors, index=df.index, columns=col)
    eigenValues = pd.Series(eigenValues, index=col)
    return eigenValues, eigenVectors


def dfDiag(df):
    """
    Return diagonal of df as Series
    :param df: dataframe
    :return: Diagonal dataframe
    """

    result = pd.Series(np.diag(df), index=df.columns)
    return result

def compParamCov(obsCov, jacAtm):
    """
    Compute the parameter covariance error
    :param obsCov: observational error -- should include obs error + 2x internal var.
    :param jacAtm: Atmospheric Jacobian
    :return: parameter covariance
    """

    invCov = dfInv(obsCov,pseudoInv=False)
    invCov = jacAtm @ invCov
    Hessian = invCov @ jacAtm.transpose()  # matrix to be inverted
    invM = dfInv(Hessian,pseudoInv=False)
    transMat = invM @ invCov  # transformation matrix
    paramCov = (transMat.dot(obsCov)).dot(transMat.T)  # covariance of parameters

    return paramCov

def combCovar(multiNorm1, multiNorm2):
    """
    Combine two multi-normal distributions using the standard bayesian formula
    see https://en.wikipedia.org/wiki/Conjugate_prior
    :param multiNorm1: a scipy.stats.multi_normal object
    :param multiNorm2:a scipy.stats.multi_normal object
    :return: scipy.stats.multi_normal object
    """
    invFn = np.linalg.pinv
    precision1 = invFn(multiNorm1.cov)
    precision2 = invFn(multiNorm2.cov)

    covar = invFn(precision1 + precision2)
    mn = covar @ ((precision1 @ multiNorm1.mean) + (precision2 @ multiNorm2.mean))
    return scipy.stats.multivariate_normal(mean=mn, cov=covar,allow_singular=True)

def compPredictionVar(jacobian, paramNorm, stdParam, stdValues):
    """
    Compute the covariance of predictions
     :param jacobian: Jacobian being used.
    :param paramNorm: frozen scipy.stats.multivariate_normal object for the multi-variate distribution of the parameters.
    :param stdParam: standard parameter setting.
    :param stdValues: Standard values (of whatever jacobian is)

    :return: a scipy.stats.multivariate_normal object with the mean and std dev of
    """
    mn = stdValues + jacobian.transpose() @ (stdParam - paramNorm.mean)
    cov = (jacobian.values.T @ paramNorm.cov) @ jacobian.values
    return scipy.stats.multivariate_normal(mean=mn, cov=(cov+cov.T)/2., allow_singular=True)

def limitParamCov(paramDist, stdParam, stdValues, jacobian, minSamp=1000, paramLimit=None, verbose=False):
    """
    Compute the covariance of the coupled model when parameters are limited to (by default) (0,1)
    :param paramDist: scipy.stats.multivariate_normal object -- multi-variate normal dist of parameter values.
    :param stdParam: standard parameters
    :param stdValues: standard model values
    :param jacobian: Jacobian of d model values/ d param
    :param minSamp: Minimum number of samples to generate.
    :param paramLimit: Limits on params (pass in a 2 x nParam numpy array) --
        default is None which sets limits to 0,1 for each param
    :param verbose: Optional -- default False. If True be a bit verbose.
    :return: a multivariate normal dist.
    """

    np.random.seed(123456)  # always make sure have the same RNG seed
    if paramLimit is None:
        paramLimit = np.zeros([2,len(stdParam)])
        paramLimit[1,:] = 1


    sampCount = 0  # no of samples
    nSamp = int(minSamp) # initial estimate of how many samples to generate

    samp = np.zeros((0, len(stdParam)))
    cnt = np.zeros(len(stdParam))  # total number of OK samples for each parameter
    nGen = 0  # no of samples generated

    while sampCount < minSamp:
        lsamp = paramDist.rvs(nSamp)
        nGen += nSamp  # running count of how many samples have been generated.
        L = ((lsamp >= paramLimit[0,:]) & (lsamp <= paramLimit[1,:]))  # where all generated values are OK
        cnt += L.sum(0)  # increment count of OK for individual parameters
        OK = L.all(axis=1)  # good cases
        sampCount += OK.sum()  # total no of samples we've generated.
        samp = np.append(samp, lsamp[OK, :], 0)  # list of OK samples!
        nSamp *= 10  # next time generate 10 times as many.

    fractParamOK = cnt / float(nGen)  # what fraction of parameters are good..
    fractOK = sampCount / float(nGen)  # what fraction of samples are good
    if verbose:
        print(f"Sample Fract. {fractOK} Paramfract: {fractParamOK} ")
    # compute the (linear) climate change for the  cases in the range 0..1

    samp = pd.DataFrame(samp, columns=stdParam.index)  # convert all samples into dataFrame
    change = (samp - stdParam) @ jacobian  # compute the linear sensitivity
    covDelta = np.cov(change.transpose())  # compute covariance
    result = scipy.stats.multivariate_normal(mean=stdValues + change.mean(), cov=covDelta, allow_singular=True)

    return result


def sym_colors(df, text=False):
    """
    Return a list of colours
    :param df -- the dataframe  -- needs to have Optimised & Ensemble and set
    :param text if True (Default is False) return text colour.
    """

    ens_colours = {'CMIP5': 'darkBlue', 'CMIP6': 'cornflowerblue', 'CE7': 'grey', 'DF14': 'orange',
                   'CE14': 'indianred',
                   'ICE': 'grey'}
    ens_text_colours = {'CMIP5': 'black', 'CMIP6': 'darkblue', 'CE7': 'black', 'DF14': 'black',
                        'CE14': 'black',
                        'ICE': 'black'}

    optimise = dict(y='lightblue', n='red')  # force optimise to lowercase.

    if text:
        lookup_colours = ens_text_colours
        stdColour = 'black'
    else:
        lookup_colours = ens_colours
        stdColour = 'grey'
    colours = dict()  # list of colors
    for name, row in df.iterrows():
        colour = lookup_colours.get(row.Ensemble)
        if colour is None:
            colour = optimise.get(str(row.Optimised).lower(), None)
        if name == 'Standard':
            colour = stdColour
        colours[name] = colour

    return pd.Series(colours, name='Colour')

def read_CMIP(json_file, label):
    """

    :param json_file:
    :param label:
    :return:
    """

    if label == 'CMIP5':
        obs_dir = OptClimPath / 'grl17_coupled/extras/CMIP5_process'
        PI_dir = OptClimPath / 'grl17_coupled/extras/CMIP5/PI'
        translate_sheet = 'AtmosCplCMIP5'

    elif label == 'CMIP6':
        obs_dir = OptClimPath / 'grl17_coupled/extras/CMIP6_process'
        PI_dir = OptClimPath / 'grl17_coupled/extras/CMIP6/PI'
        translate_sheet = 'AtmosCplCMIP6'

    else:
        raise Exception(f"Unknown label: {label}")

    amip_to_coup = {'CanAM4': 'CanESM2', 'FGOALS_g2': 'FGOALS-g2',
                    'HadGEM2-A': 'HadGEM2-ES'}  # only really needed for CMIP5
    coup_names = {'FGOALS_g2': 'FGOALS-g2'}  # names to translate coupled model names to
    files = list(obs_dir.glob('*.nc'))
    if len(files) == 0:
        raise Exception(f"No AMIP files found for {obs_dir}")
    AMIP = NCtoDF(files, json_file, labels=label, index_rename=amip_to_coup)

    PI_files = list(PI_dir.glob('*.nc'))
    if len(PI_files) == 0:
        raise Exception(f"No PI  files found for {PI_dir}")
    PI = readPI(PI_files, label).rename(index=coup_names)

    CMIPx = AMIP.merge(PI, how='inner', left_index=True,
                       right_index=True)  # merge in the PI info but only want cases where have both T & COST
    coup_info = pd.read_excel(dataPath / 'CMIPdata.xls', sheet_name=label, header=0, index_col=0)
    CMIPx = CMIPx.merge(coup_info, how='outer', left_index=True, right_index=True)
    # add in ECS_4xCO2 which is just double ECS_2xCo2 as 2xCO2 computed from 4xCo2
    CMIPx.loc[:, 'ECS_4xCO2'] = 2 * CMIPx.ECS_2xCO2
    # fix names for ensembles and set them to be Int (with missing msk)
    CMIPx = CMIPx.rename(columns=dict(NENS_x='Natmos', NENS_y='Ncoup')).astype(dict(Natmos='Int64', Ncoup='Int64'))
    return CMIPx


def areaAvg(cube, xCoord='longitude', yCoord='latitude'):
    """
    Area average a cube
    :param cube: cube to be area averaged
    :param xCoord: Optional name of x coordinate -- default is longitude
    :param yCoord: Optional name of y coordinate -- default is latitude
    :return: area averaged.
    """

    grid_areas = iris.analysis.cartography.area_weights(cube)
    result = cube.collapsed([xCoord, yCoord], iris.analysis.MEAN, weights=grid_areas)
    return result

def area_wt(dataArray, longitude_bounds=None, latitude_bounds=None):
    """
    compute the area wt for xarray cubes. Needs to make some guesses at the co-ord names
    to do so.
    :param dataArray -- dataArray for which weights to be computed
    :param longitude_bounds -longitude bounds. If not provided then will work out assuming equally spaced longitudes.
    :param latitude_bounds -- latitude bounds. If not  provided then will work out assuming equally spaced latitudes.
      For bounds coord should be nx,by,m nbounds -- nbounds =0 - upper bound, nbound=1 -- lower bound
    """
    import xarray
    radius_earth = 6.371e6
    lon = dataArray.longitude
    lat = dataArray.latitude
    if longitude_bounds is None:
        dx = np.abs(lon - np.roll(lon, 1))
        dx[0] = dx[-1]
        longitude_bounds = np.array([lon - dx / 2, lon + dx / 2]).T

    if latitude_bounds is None:
        dy = np.abs(lat - np.roll(lat, 1))
        dy[0] = dy[1]
        latitude_bounds = np.array([(lat + dy / 2).clip(-90, 90), (lat - dy / 2).clip(-90, 90)]).T

    # Want to compute $(\lambda_1-\lambda_0)*(\int_\theta_0^\theta_1 \cos\theta d\theta)$
    # problem is on the UM grid when have points at +/- 90. I assume there are 1/2 the normal size.
    # something like this (s.latitude+np.roll(s.latitude,1))/2 sort of does it.
    # so compute bottom and top bounds than do the integration (sin(top)-sin(bottom))
    # clip bottom and top at +/-pi/2 (or sin at -1,1)
    lon_bounds = np.expand_dims(np.deg2rad(longitude_bounds),axis=1)
    sin_lat_bounds = np.expand_dims(np.sin(np.deg2rad(latitude_bounds)),axis=0) # sin neeed
    area_wt = (lon_bounds[:,:, 1] - lon_bounds[:, :,0]) * \
              (sin_lat_bounds[:, :, 0] - sin_lat_bounds[:,:, 1]) * radius_earth ** 2
    # convert back to xarray datarray
    area_wt = xarray.DataArray(area_wt,
                               coords=dict(longitude=lon,latitude=lat))
    return area_wt
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
        deriv = polyval[0:-1] * np.arange(len(polyval) - 1, 0,
                                          -1)  # regression is highest order power first. So use usual rule for deriv.
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
        try:  # work out the forcing for NXCO2
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
        result['lambda' + name + '_C'] = -np.polyval(polyval_deriv(reg_clear),
                                                     ecs)  # -ve as lambda is change in outward flux per K.

    # now to compute SW/LW/SWC/LWC terms
    for index, name2 in zip(['ts_rtoalwu', 'ts_rtoaswu', 'ts_rtoalwuc', 'ts_rtoaswuc'],
                            ['LW', 'SW', 'LWC', 'SWC']):
        fName = 'F' + name + '_' + name2
        lName = 'lambda' + name + '_' + name2
        if temp is not None:
            reg = np.polyfit(temp.data, delta_CO2[index].data, fitOrder)
            result[fName] = -np.polyval(reg, 0.0)  # work out the forcing for XCO2. Flip sign as fluxes are outgoing.
            result[lName] = np.polyval(polyval_deriv(reg), ecs)  # feedback at 2xCO2 per K warming.

        else:  # no data so set to Nan
            result[fName] = np.nan
            result[lName] = np.nan

    # strip all numpy stuff out -- causes trouble when making pandas series/dataframes..
    for key, v in result.items():
        if isinstance(v, np.ndarray) and len(v) == 1: result[key] = v[0]  # just first value.
    return result


def forceFeedback_old(delta_CO2=None, CO2=2):
    """
    Compute forcing and feedback from 2xCO2 (or 4xCo2) experiments. Old -- broken code.
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
        try:  # work out the forcing for NXCO2
            r = np.polyval(reg, 0.0)[0]
            r_clear = np.polyval(reg_clear, 0.0)[0]
        except IndexError:
            r = np.polyval(reg, 0.0)
            r_clear = np.polyval(reg_clear, 0.0)
        result['F' + name] = r
        result['Fclear' + name] = r
        roots = np.roots(reg.ravel())  # and the ECS (when net flux is zero)
        ecs = roots[(roots > 0.0) & np.isreal(roots)].min()
        # print("ECS is ",ecs,'Alt = ',np.squeeze(-reg[1]/reg[0]))
        result['ECS' + name] = ecs
        l = (result['F' + name] - np.polyval(reg, ecs)) / ecs
        l_clr = (result['F' + name] - np.polyval(reg_clear, ecs)) / ecs
        try:
            result['lambda' + name] = l[0]  # feedback at W/m^2 per K warming.
            result['lambda' + name + '_C'] = l_clr[0]  # feedback at W/m^2 per K warming.
        except IndexError:
            result['lambda' + name] = l
            result['lambda' + name + '_C'] = l_clr

    # now to compute SW/LW/SWC/LWC terms
    for index, name2 in zip(['ts_rtoalwu', 'ts_rtoaswu', 'ts_rtoalwuc', 'ts_rtoaswuc'],
                            ['LW', 'SW', 'LWC', 'SWC']):
        fName = 'F' + name + '_' + name2
        lName = 'lambda' + name + '_' + name2
        if temp is not None:
            reg = np.polyfit(temp.data, delta_CO2[index].data, fitOrder)
            force = -np.polyval(reg, 0.0)  # work out the forcing for 2XCO2. Flip sign as fluxes are outgoing.
            feedback = (force - np.polyval(reg, ecs)) / ecs  # feedback at 2xCO2 per K warming.
            try:
                result[fName] = force[0]
                result[lName] = feedback[0]
            except IndexError:
                result[fName] = force
                result[lName] = feedback
        else:  # no data so set to Nan
            result[fName] = np.nan
            result[lName] = np.nan

    # strip all numpy stuff out -- causes trouble when making pandas series/dataframes..
    for key, v in result.items():
        if isinstance(v, np.ndarray) and len(v) == 1: result[key] = v[0]  # just first value.
    return result


def read_data(sevenParam=False, fourteenParam=False, sensCases=False, dfols=False):
    """
    Read in dataframe info.
    :param severnParam (optional - default is False). If True return only the 7-parameter cases.
    :param fourteenParam (optional -- default is False). If True return only the 14-parameter cases
    :param sensCases (optional -- default is False). If True return only the sensitivity cases
    :return: a data frame!
    """

    allTsData = pd.read_csv(os.path.join(dataPath, "all_ts_data.csv"), index_col=0)
    name = None
    # TODO modif readData so choice is a string!
    if np.sum([sevenParam, fourteenParam, sensCases, dfols]) > 1:
        raise Exception("Do  not set more than one of sevenParam, fourteenParam, sensCases & dfols to True")

    L7param = allTsData.index.str.match('HadAM3-7') & ~allTsData.index.str.match('-lc')
    L14param = allTsData.index.str.match('HadAM3-14')
    Ldfolsparam = allTsData.index.str.match('HadAM3-DFO14')
    if sevenParam:
        allTsData = allTsData[L7param]
        name = 'sevenParm'

    if fourteenParam:
        allTsData = allTsData[L14param]
        name = 'fourteenParam'

    if sensCases:
        l = ~(L7param | L14param | Ldfolsparam)
        allTsData = allTsData[l]
        name = 'sensCases'

    if dfols:
        allTsData = allTsData[Ldfolsparam]
        name = 'dfols'
    runInfo = pd.read_excel('OptClim_lookup.xlsx', index_col=0, dtype={'shortText': str})  # read in meta data on runs
    worked = runInfo.Status == 'Succeeded'
    # and merge them
    allTsData = allTsData.merge(runInfo[worked], left_index=True, right_index=True)
    # and add names.
    if name is not None:
        allTsData.loc[:, 'Collection'] = name

    return allTsData


def readAtmos(runInfo, verbose=False):
    """
    Read data from atmospheric files and return two pandas DataFrames
    :param runInfo =-- run info pandas datadframe -- provides all the info wanted!
    """
    # work out the files we want to read

    atmosCols = ['Atmosphere Run#%d' % x for x in range(1, 5)]  # columns for Atmospheric runs
    atmos = []
    for study in runInfo.index:
        jsonFile = os.path.join(rootData, runInfo.loc[study, 'JsonFile'])
        experiments = runInfo.loc[study, atmosCols]  # get experiment names
        m = experiments.notnull()
        labels = [x for i, x in enumerate(experiments) if m[i]]
        Files = [os.path.join(rootData, "fulldiags", x, "observations.nc") for i, x in enumerate(experiments) if
                 m[i]]  # get file names
        # read em as one DataFrame renaming columns etc.
        # hack to deal with json names for first gen of runs
        if "update_" in jsonFile:
            print("WARNING: Assuming first gen JSON file. NC vars being renamed")
            ncVarRename = varRename
        else:
            ncVarRename = None
        df = NCtoDF(Files, jsonFile, labels=labels, column_rename=column_rename,
                    verbose=verbose, ncVarRename=ncVarRename)  # column_rename and varRename defined in the module.

        # extract data from dataFrame putting average into result
        atmos.append(df.mean(numeric_only=True).rename(study))
    # now make everything into a list
    atmos = pd.DataFrame(atmos)
    return atmos


def shortNames(paramNames=None, obsNames=None, mark=None):
    """
    generate short names for parameters or observables
    :param paramNames -- list of parameter names
    :param obsNames -- list of observations
    :return dict of mapping from long to short names
    provide one of param or obs names
    """
    cnt = int(paramNames is None) + int(obsNames is None)  # count of None values
    # if cnt != 1:
    #    raise ValueError("Provide ONE of paramNames and obsNames")

    result = collections.OrderedDict()
    if paramNames is not None:
        for p in paramNames:
            result[p] = p[0:3].replace("_", "")  # first three characters with _ removed.

    if obsNames is not None:
        odict = {'temp@500': 'T500', 'rh@500': 'q500', 'lprecip': 'LP', 'mslp': 'SLP', 'COST': "C", 'rh500': 'q500',
                 'temp500': 'T500'}
        for o in obsNames:
            # search for match..
            for k in odict.keys():
                m = re.match(k, o)  # note match looks for match from start of string
                if m is not None:  # matched!
                    result[o] = odict[k]  # substitute
                    break  # stop looking for matches

            if m is None:  # failed to match a regexp so use default.
                result[o] = o[0:3].replace("_", "").upper()  # first three characters with _ removed.
    if mark:  # want to mark some parameters
        for m in mark:
            try:
                result[m] += '*'  # append a *
            except KeyError:  # failed to find key which is OK!
                pass

    return result


def label(upper=False, roman=False):
    """
    Generator function to provide labels
    :param:upper (default False)
    :return: generator function
    """
    import string
    if roman:  # roman numerals
        strings = ['i', 'ii', 'iii', 'iv', 'defaultCov', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii']
    else:
        strings = [x for x in string.ascii_lowercase]

    if upper:  # upper case if requested
        strings = [x.upper() for x in strings]

    num = 0
    while True:  # keep going
        yield strings[num] + " )"
        num += 1
        num = num % len(strings)


def plotLabel(ax, labelgen, where=None):
    """
    Plot a label  on the current axis
    :param ax -- axis to put the label on
    :param labelgen -- label generator function.
    :param where -- where to put label relative to plot.
    """
    txt = next(labelgen)
    if where is None:
        x = -0.03;
        y = 1.03
    else:
        (x, y) = where
    # ax.annotate(txt, xy=(0.1, 1.03), xycoords="axes fraction")
    ax.text(x, y, txt, transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='bottom')


def latitudeLabel(value, pos):
    """
    :param values -- value of label
    """

    deg = r'$^\circ$'  # what we need for a degree symbol
    if value < 0:
        end = deg + 'S'
    elif value > 0:
        end = deg + 'N'
    else:
        end = ''

    c = mpl.ticker.ScalarFormatter()
    c.set_scientific(False)
    str = c.format_data_short(abs(value)).strip()  # trailing strip removes whitespace
    str += end
    if abs(value) < 1e-6:
        str = 'Eq'

    return str


def myName():
    """
    Return filename of calling function
    """

    frame = inspect.getouterframes(inspect.currentframe())[1]
    filename = os.path.basename(frame[1])  #
    return os.path.splitext(filename)[0]


def saveFig(fig, name=None, savedir="figures", figtype=None):
    """
    :param fig -- figure to save
    :param name (optional) set to None if undefined
    :param savedir (optional) directory to save figure to. Default is figures
    :param figtype (optional) type of figure. Defailt is ".pdf"
    """

    # set up defaults
    if figtype is None:
        figtype = defFigType
    # work out sub_plot_name.
    if name is None:
        fig_name = fig.get_label()
    else:
        fig_name = name

    outFileName = os.path.join(savedir, fig_name + figtype)
    fig.savefig(outFileName, dpi=dpi)


def readCSVdata(filename, scalings=None, remove=None):
    """
    :param filename -- name of csv file to read
    :param scalings (optional) -- sclaings to apply to data. A hash with keys being the regexp pattern and value the scaling
    :param remove (optional) if set to name then numerical values indexed by than name are removed from all elements.. data statisityics (IV_* & SV_*) are not modified.
    """

    data = pd.read_csv(filename)
    data.set_index("NAMES", inplace=True)
    cols = data.columns

    # potentially scale data
    if scalings is not None:
        for pattern, scale in scalings.iteritems():
            m = cols.str.match(pattern)
            data.loc[:, m] = data.loc[:, m] * scale  # scale data

    # give everything short names...

    rename = dict()  # shorten column names
    for c in cols:
        rename[c] = c.replace("TS_", ""). \
            replace("LAND_", "L"). \
            replace("NORTHERN_", "N").replace("SOUTHERN_", "S").replace("HEMISPHERE", "H").replace("GLOBAL", "G")

    data.rename(columns=rename, inplace=True)
    data.loc[:, 'experiment'] = data.index

    # potentially remove reference data
    ref = None
    if remove is not None:
        subset = data.select_dtypes(include=[np.number])
        col = subset.columns
        col = col[~col.str.match("^[IS]V_")]  # extract the columns that are not variability.
        ref = data.loc[remove, :].copy()  # extract the reference value
        data.loc[:, col] = subset - subset.loc[remove,
                                    :]  # remove the reference from numerical values that are not variances

    return data, ref


# end of readData()

def readNCdata(files, jsonFile, labels,
               column_rename=None, verbose=False, ncVarRename=None, scale=True):
    """
    readNC data and modify index
    :param files: files to read
    :param jsonFile: configuration file for study
    :param column_rename: (optional) column names to rename
    :param verbose: default False -- if True be verbose
    :param ncVarRename: rename netcdf variables
    :param labels: labels to use.
    :return:
    """
    # only keep the paths that exist
    filesGot, labelsGot = zip(*[x for x in zip(files, labels) if os.path.isfile(x[0])])

    df = NCtoDF(filesGot, jsonFile, column_rename=column_rename, verbose=verbose,
                ncVarRename=ncVarRename, labels=labelsGot)
    df.loc[:, 'experiment'] = [os.path.basename(os.path.dirname(n)) for n in filesGot]

    df.loc[:, 'label'] = labelsGot
    df.set_index('label', inplace=True, verify_integrity=True)
    return df


def readPI(files, source=None):
    """

    :param files: list of files to read
    :param source: source -- CMIP5 or CMIP6 to determine metadata to use
    :return: a Dataframe
    """
    series = []
    for file in files:
        ds = iris.load_cube(str(file), 'air_temperature')
        value = ds.data[-100:].mean().astype('float')  # average over the last 100 years
        if source == 'CMIP6':
            name = ds.attributes['source_id']
        elif source == 'CMIP5':
            name = ds.attributes['model_id']
        else:
            raise Exception(f"Unknown Source: {source}")
        s = pd.Series(dict(CTL_ts_t15=value, model=name), name=file.name)
        # s = pd.
        series.append(s)

    df = pd.DataFrame(series)
    # now average up the models.

    grp = df.groupby(by='model')
    dfg = grp.mean()
    dfg.loc[:, 'NENS'] = grp.count()  # count ens
    dfg.loc[:, 'Ensemble'] = source  # store the ensemble info
    # sorted alphabetically. Upper case then lower case.
    return dfg


def NCtoDF(files, jsonFile, column_rename=None, index_rename=None, verbose=False, ncVarRename=None, labels=None,
           scale=None):
    """
    Read observations.nc files and put them all into one large dataframe
    :param files:  file names to read which will be expanded using glob
    :param jsonFile:  name of jsonFle
    :param column_rename: (optional) dict of column names to rewrite. Passed to dataFrame.rename()
    :param index_rename: (optional) dict to index names to rename
    :param ncVarRename : rename variables in config file
    :param scale (default None): If set scale all non OLR & RSR rows & columns by this factor . Diag is scaled scale^2
    :param verbose: (optional). If set true be more verbose.
    :param labels (optional). A list of names to use -- should be the same len as expanded files.
       If not set then labels are the same as expanded files. labels can also be the string 'CMIP5' or 'CMIP6'
       in which case the names will be worked out from the meta-data in the netcdf file.

    :return: a pandas dataFrame
    """
    import StudyConfig
    import Optimise
    import optClimLib
    import netCDF4
    studyCfg = StudyConfig.readConfig(filename=jsonFile)  # read and parse the config file
    obsNames = studyCfg.obsNames()
    if ncVarRename is None:
        obsNamesRead = obsNames
        reverseName = None
    else:
        obsNamesRead = []
        reverseName = dict()
        for name in obsNames:  # iterate over obsNames replacing name (if it exists with name in ncVarRename
            newName = ncVarRename.get(name, name)
            obsNamesRead.append(newName)
            reverseName[newName] = name  # reverse lookup

    scales = studyCfg.scales(obsNames=obsNames)
    obs = studyCfg.targets(obsNames=obsNames, scale=True)  # get the target values
    cov = studyCfg.Covariances(obsNames=obsNames, scale=True)['CovTotal']
    # possibly scale nonOLR & RSR values.
    if scale is not None:
        covObsErr = studyCfg.Covariances(obsNames=obsNames, scale=True)['CovObsErr']
        covIntVar = studyCfg.Covariances(obsNames=obsNames, scale=True)['CovIntVar']
        for obsN in ['lprecip', 'lat', 'temp@500', 'rh@500', 'mslp', 'netflux']:
            L = covObsErr.index.str.match(obsN)
            covObsErr.loc[L, :] *= 2  # scale columns
            covObsErr.loc[:, L] *= 2  # and rows
        cov = covObsErr + 2 * covIntVar

    opt = studyCfg.optimise()
    covar_cond = optClimLib.get_default(opt, 'covar_cond', None)  # get regularisation for covariance

    # Possibly regularize covariance matrix
    # TODO: make the regularisation part of the read/process covariance bloack.
    cov = cov.values  # extract cov from its pandas array.

    if covar_cond is not None:  # specified a condition number for the covariance matrix?
        cov = Optimise.regularize_cov(cov, covar_cond, trace=verbose)

    lst = []  # list holding Series from each file
    if isinstance(files, (str)):
        filesProcess = [files]
    else:
        filesProcess = files

    # lstFiles = []
    df = pd.DataFrame()  # empty dataframe

    if labels in (None, 'CMIP5', 'CMIP6'):
        labels_use = filesProcess
    else:
        labels_use = labels

    for lab, file in zip(labels_use, filesProcess):
        simObs, constraint = optClimLib.gatherNetCDF(file, obsNamesRead, trace=verbose, allowNan=True)
        # set the name.
        if labels == 'CMIP5':
            CMIP5_lookup = dict(Model77='GFDL-HIRAM-C180',
                                Model78='GFDL-HIRAM-C180')  # whnen meta-data missing use the model number to override.
            ds = netCDF4.Dataset(file,
                                 "r")  # open it up for reading using python exception handling if want to trap file not existing
            try:
                name = ds.variables[obsNamesRead[0]].getncattr('model_id')
            except AttributeError:  # failed to find it.
                print(f"Failed to find model_id for {file} working out from filename")
                model = file.name.split('_')[0]
                name = CMIP5_lookup.get(model, model)

            ds.close()
        elif labels == 'CMIP6':
            ds = netCDF4.Dataset(file,
                                 "r")  # open it up for reading using python exception handling if want to trap file not existing
            name = ds.variables[obsNamesRead[0]].getncattr('source_id')

            ds.close()
        else:
            name = lab
        simObs.name = name
        # now to get the name for C
        # constraint dead code as just wrapped into values
        if reverseName is not None:
            simObs.rename_axis(reverseName, inplace=True)  # name back to original names
        ScSimObs = simObs * scales  # scale the simulated obs.

        ScSimObs.loc['NENS'] = 1

        ScSimObs.name = name
        try:
            df.loc[name, :] += ScSimObs  # make new series which contains obs and errors.
        except KeyError:
            df = df.append(ScSimObs)

    # possibly rename columns
    if (column_rename is not None):  # rename some variables
        df.rename(columns=column_rename, inplace=True)  # rename the columns
        # commented out covariance renaming as don't think it needed. But if it is needed then uncomment.
        # allTsData.covariance.rename(columns=column_rename,index=column_rename,inplace=True)
        # allTsData.covLS.rename(columns=column_rename,index=column_rename,inplace=True)

    # compute average over Ensemble
    cols = obs.index
    nens = df.loc[:, 'NENS']
    df = df.div(nens,
                axis=0)  # this is annoying. Would like allTsData=allTsData/nens to work but interpts as rows not columns.
    df.loc[:, 'NENS'] = nens
    # add error/COST computed from MEAN values
    errList = []

    for lab in df.index:
        consErr = Optimise.calcErr(df.loc[lab, cols].values, obs.values, cov=cov)
        err = consErr
        errS = pd.Series([err[0], consErr[0]], index=['ERR', 'COST'])
        errS.name = lab
        errList.append(errS)  # series for errors

    errDF = pd.DataFrame(errList, index=df.index)

    df = pd.concat([df, errDF], axis=1)
    if index_rename is not None:  # change the index
        df = df.rename(index=index_rename)

    return df


# end of NCtoDF


def pp_time(tspp, ycoord=True):
    """
    Generate time values for pp field
    :arg tspp timeseries
    """

    if ycoord:
        time = getattr(tspp, 'y', tspp.bzy + tspp.bdy * (1 + np.arange(tspp.data.shape[0])))
        if len(time) == 1 or np.all(time == 0.0):
            time = tspp.bzy + tspp.bdy * (1 + np.arange(tspp.data.shape[0]))
    else:
        time = getattr(tspp, 'x', tspp.bzx + tspp.bdx * (1 + np.arange(tspp.data.shape[1])))
        if len(time) == 1 or np.all(time == 0.0):
            time = tspp.bzx + tspp.bdx * (1 + np.arange(tspp.data.shape[1]))

    return time


def plot_cov_ellipse(mn, cov, nstd=2, ax=None, **kwargs):
    import numpy.linalg as linalg
    from matplotlib.patches import Ellipse
    """
    FRom: https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        mn : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        cov : The 2x2 covariance matrix to base the ellipse on

        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are passed on to the ellipse patch. Useful ones are:
            Fill (default is True)
            facecolor -- which fills the ellipse
            edgecolor -- which draws the line edge.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    # print(vals)

    width, height = 2 * nstd * np.sqrt(np.abs(vals))
    ellip = Ellipse(xy=mn, width=width, height=height, angle=theta, **kwargs)

    ax.add_patch(ellip)  # change from ref code to patch. This means that autoscale works.

    # ax.autoscale_view()
    return ellip


def readFile(dir, name, realization=None):
    fullname = os.path.join(time_cache, dir, name)
    pp = read_pp(fullname, realization=realization)
    return pp


@functools.lru_cache(maxsize=5012)
def readPP(dir, name, year=None, realization=None):
    """
    Read in pp data and add some aux co-ords.
    :param dir: name of directory relative to time-cache
    :param name:name of file
    :param year (optional): if suppled will extract sub-year for that year.
    :return: cube
    """

    # f = os.path.join(time_cache, os.path.basename(dir), name)
    f = dir / name
    # convert to string as iris doesn't cope with pathlib
    f = str(f)
    try:
        cube = iris.load_cube(f)
        # add a aux co - ord with the filename
        new_coord = iris.coords.AuxCoord(f, long_name='fileName', units='no_unit')
        cube.add_aux_coord(new_coord)

        addAuxCoord(cube, realization=realization)

        # possibly extract year

        if year is not None:
            cube = cube.extract(iris.Constraint(year=year))
    # print(cube)
    except AttributeError:
        # iris load doesn't work so try read_pp
        cube = read_pp(f, realization=realization)
    return cube


@functools.lru_cache(maxsize=5012)
def read_pp(file, realization=None, verbose=False):
    """
    Read a pp file  using iris pp library and return it as a cube
    Deals with (eventually) all the brokenness of iris...
    :param file: file to read data from
    :return: pp data  returned
    """

    if file is None: return None
    if verbose:
        print("Reading data from %s" % (file))
    # if re.match('$ts_',os.path.basename(file)) is None:
    #    fld = pp.load(file)
    #    return  fld

    ts = next(pp.load(file))  # extract the timeseries
    # ts=ts[0] # extract from list
    # set the time
    timeValues = (20, 21, 22, 23)
    if ts.lbcode.ix in timeValues:  # x is time-coord
        ts.x = pp_time(ts, ycoord=False)  # this is (I think) the real fix.
    if ts.lbcode.iy in timeValues:  # y is time-coord
        ts.y = pp_time(ts)

    # fix the co-ords if they don't exist
    try:
        x = ts.x
    except AttributeError:
        x = ts.bzx + (np.arange(ts.lbnpt) + 1) * ts.bdx
        ts.x = x

    try:
        y = ts.y
    except AttributeError:
        y = ts.bzy + (np.arange(ts.lbrow) + 1) * ts.bdy
        ts.y = y
    stuff = iris.fileformats.pp_load_rules.convert(ts)  # iris 2.0
    ts.data = np.ma.masked_equal(ts.data, ts.bmdi)
    # fix std name of
    cube = iris.cube.Cube(ts.data, standard_name=stuff.standard_name,
                          long_name=stuff.long_name,
                          # var_name=stuff.var_name,
                          units=stuff.units,
                          attributes=stuff.attributes, cell_methods=stuff.cell_methods,
                          dim_coords_and_dims=stuff.dim_coords_and_dims,
                          aux_coords_and_dims=stuff.aux_coords_and_dims)  # ,aux_factories=stuff.aux_factories)
    # all to here could be replaced with cube = iris.load_cube(file) though getting hold of the meta-data
    # might be tricky.
    cube.name(file)
    # add co-ords --
    for code, name in zip([5, 13], ['model_level', 'site_number']):
        try:
            if ts.lbcode.ix == code:
                level = iris.coords.DimCoord(ts.x, long_name=name)
                try:
                    cube.add_dim_coord(level, 1)
                except ValueError:
                    pass
        except AttributeError:
            pass
        try:
            if ts.lbcode.iy == code:
                level = iris.coords.DimCoord(ts.y, long_name=name)
                try:
                    cube.add_dim_coord(level, 0)
                except ValueError:
                    pass
        except AttributeError:
            pass

    # add a aux co-ord with the filename
    new_coord = iris.coords.AuxCoord(file, long_name='fileName', units='no_unit')
    cube.add_aux_coord(new_coord)

    addAuxCoord(cube, realization=realization)
    # cube = iris.util.squeeze(cube)
    return cube


def addAuxCoord(cube, realization=None):
    """
    Add aux coords to a cube
    :param cube:
    :return:
    """

    for coord in ['time', 'latitude', 'longitude']:
        try:
            cube.coord(coord).guess_bounds()
        except (ValueError, iris.exceptions.CoordinateNotFoundError):
            pass

    # add auxilary information. year & month number
    iris.coord_categorisation.add_year(cube, 'time')
    iris.coord_categorisation.add_month_number(cube, 'time')
    # optionally add ensemble aux_coord -- from iris example doc..
    if realization is not None:
        realization_coord = iris.coords.AuxCoord(realization, 'realization')
        cube.add_aux_coord(realization_coord)
        iris.util.new_axis(cube, scalar_coord='realization')

    return cube


def comp_crf(direct):
    """
    Compute the cloud radiative forcing from ummonitor cached data
    CRF is Net  sky flux - Net  clear sky flux.

    :param direct -- the name of the directory

    """
    net = comp_net(direct)
    clr = comp_net(direct, clear=True)
    crf = net - clr

    crf.units = "W m^-2"
    crf.long_name = 'Cloud Rad. Forcing'

    return crf


def comp_net(direct, clear=False):
    """
    Compute net flux from ummonitor cached data
    :param direct: directory where pp files live
    :return: net flux
    """

    # ts_olr = read_pp(os.path.join(direct, 'ts_rtoalwu.pp'))
    # ts_rsr = read_pp(os.path.join(direct, 'ts_rtoaswu.pp'))
    # ts_insw = read_pp(os.path.join(direct, 'ts_rtoaswd.pp'))

    if clear:
        ts_olr = readFile(direct, 'ts_rtoalwuc.pp')
        ts_rsr = readFile(direct, 'ts_rtoaswuc.pp')
    else:
        ts_olr = readFile(direct, 'ts_rtoalwu.pp')
        ts_rsr = readFile(direct, 'ts_rtoaswu.pp')
    ts_insw = readFile(direct, 'ts_rtoaswd.pp')
    net = ts_insw - ts_olr - ts_rsr
    net.units = "W m^-2"
    if clear:
        net.long_name = 'Clear Sky Net Flux'
    else:
        net.long_name = 'Net Flux'

    return net


def readDelta(dir1, dir2, file):
    """
    Compute difference between file in dir1 from file in dir2
    :param dir1: Directory (as in readFile) where first data is
    :param dir2: Directory (as in readFile) where second data is
    :param file:Fiel being read in
    :return: difference between two datasets.
    """
    pp1 = readFile(dir1, file)
    pp2 = readFile(dir2, file)
    delta = comp_delta(pp1, pp2)
    return (delta)


def comp_delta(cube1, cube2):
    """
    Compute difference between two cubes. cube2 will be interpolated to same times as cube1
    :param cube1:  first cube
    :param cube2: 2nd cube
    :return: difference after interpolating cube2 to same times as cube1.
    """
    # check min and max times of cube2 are consistent with cube1
    if cube2.coord('time').bounds.min() > cube1.coord('time').bounds.min():
        print("Extrapolating below min time -- returning None for ", cube2.name())
        raise Exception("Extrapolating below min")
        return None

    if cube2.coord('time').bounds.max() < cube1.coord('time').bounds.max():
        print("Extrapolating above max time -- returning None for", cube2.name())
        raise Exception("Extrapolating above max")
        return None

    interp = cube2.interpolate([('time', cube1.coord('time').points)], iris.analysis.Linear())
    try:
        diff = (cube1 - interp)
    except ValueError:
        diff = cube1.copy()
        diff.data = cube1.data - interp.data
        # print "Fix ",exper.name,cube1.name()

    diff.units = cube1.units  # quite why this needs to be done I don't know why..
    diff.long_name = cube2.long_name
    return diff


def detrend(cubeIn, order=2, timeCoord='time'):
    """
    Remove polynomial fit from Cubein and return data with this removed
    :param cubeIn: Cube for which time trend will be removed
    :param order: polynomial order of fit
    :return: Cube with trend removed.
    """

    time = cubeIn.coord('time').points

    timeAxis = cubeIn.coord_dims(timeCoord)  # work out which axis is time.
    data = np.moveaxis(cubeIn.data, timeAxis, 0)  # data in correct orde.
    shape = data.shape
    npt = np.product(shape[1:], dtype='int64')
    # import pdb; pdb.set_trace()
    data = data.reshape(shape[0], npt)

    # watch out when not enough values. Need at least order+1 points.
    try:
        cnt = data.count(0)  # sum all non missing data,
    except AttributeError:
        cnt = np.repeat(shape[0], npt)
    L = cnt > order  # logical where we have enough pts.
    pmsk = np.ma.polyfit(time, data[:, L], order)  # result on masked subset
    shp = (pmsk.shape[0], data.shape[1])  # shape of result array
    # import pdb; pdb.set_trace()
    p = np.tile(np.nan, shp)  # array with all masked.
    p[:, L] = pmsk
    p = np.ma.masked_invalid(p)  # now as masked array
    # import pdb ; pdb.set_trace()
    trend = np.polyval(p, time).reshape(shape)
    # and final bit put time back in right place./
    trend = np.moveaxis(trend, 0, timeAxis)

    # now to remove the trend
    result = cubeIn.copy()
    result.data = result.data - trend

    return result


def regressFn(data, axis=None, **kwargs):
    """

    :param data:
    :param axis:
    :param kwargs:
    :return:
    """

    tsData = kwargs['timeseries']
    fn = functools.partial(scipy.stats.linregress, tsData)
    res = np.apply_along_axis(fn, axis, data)

    return res[:, :, 0]


# RegAgg = iris.analysis.Aggregator('regression', regressFn)


def regress(cubeIn, timeSeries, regressCoord='time', mdtol=0.5):
    """
    Do field regress -- simply apply
    :param cubeIn: input Cube
    :param timeSeries:  time series to regress
    :param regressCoord: coord to regress on (default is time)
    :param modtol (default is 0.5) maximum amount of missing data to tolerate.
    :return:
    """

    data = cubeIn.data
    # and make it a masked array..
    data = np.ma.masked_array(data)
    # find coords for TS data

    if type(regressCoord) is list:
        # work out where tgt co-ords are first.
        axes = []
        axesTS = []
        # have list of co-ords so find where they are
        for cName in regressCoord:
            axes.append(cubeIn.coord_dims(cName)[0])
            axesTS.append(timeSeries.coord_dims(cName)[0])  # find the axes we want

        # deal with timeseries. The permutation below should put things in consistent orders.
        TSdata = np.moveaxis(timeSeries.data, axesTS, np.arange(0, len(axes)))  # permute it.
        TSdata = TSdata.flatten()

        # deal with the data itself.
        naxis = len(regressCoord)
        data = np.moveaxis(data, axes, np.arange(0, len(axes)))  # permute data array so axis we want are leading
        shape = data.shape
        data = data.reshape([-1] + list(shape[naxis:]))  # flatten the data so that regression axis is dim 0.
        axis = 0  # will apply function along 0 axis.
    else:
        axis = cubeIn.coord_dims(regressCoord)[0]  # which axis do we want
        TSdata = timeSeries.data.flatten()

    fn = functools.partial(scipy.stats.linregress, TSdata)  # set up function for regression
    res = np.apply_along_axis(fn, axis, data)  # actually do the regression.
    # TODO implement missing data and then mask data if too much missing.
    cnt = data.count(axis)  # sum all non missing data,
    msk = cnt < (mdtol * TSdata.size)
    res = np.ma.array(res, mask=np.broadcast_to(msk, res.shape))
    # wrap as a cube.
    regCube = cubeIn.collapsed(regressCoord, iris.analysis.MEAN)  # generate example cube...
    result = iris.cube.CubeList()  # result will be a cube list
    # now set up date, name and units appropriately.
    for indx, (name, unit) in enumerate(zip(['slope', 'intercept', 'rvalue', 'pvalue', 'stderr'],
                                            [cubeIn.units / timeSeries.units, cubeIn.units, 1, 1,
                                             cubeIn.units / timeSeries.units])):
        c = regCube.copy(data=res[indx, ...])
        c.units = unit
        c.rename(name + ' ' + c.name())
        result.append(c)

    return result


def detrend2(cubeIn, order=2, timeCoord='time', mdtol=0.5):
    """
    Remove polynomial fit from Cubein and return data with this removed
    :param cubeIn: Cube for which time trend will be removed
    :param order: (default is 2)  polynomial order of fit
    :param timeCoord (default is 'time') name of time axis.
    :param mdtol (default is 0.5) missing data tolerance
    :return: Cube with trend removed.
    """

    time = cubeIn.coord(timeCoord).points
    timeAxis = cubeIn.coord_dims(timeCoord)[0]  # work out which axis is time.
    fitfn = functools.partial(np.polyfit, time)
    # watch out when not enough values. Need at least order+1 points.
    data = cubeIn.data
    # cnt = data.count(timeAxis) # sum all non missing data,
    fit = np.apply_along_axis(fitfn, timeAxis, cubeIn.data, order)
    trend = np.apply_along_axis(np.polyval, timeAxis, fit, time)
    # now to remove the trend
    result = cubeIn.copy()
    result.data = data - trend

    return result


def comp_fit(cubeIn, order=2, year=np.array([109, 179]), x=None, bootstrap=None, timeAvg=None, mask=False,
             fit=False, makeCube=False):
    """
    Fit Nth (Default is 2) order fn to time co-ordinate of cube then compute value at specified year
    :param cube: cube to do fitting to.
    :param order  (default 2 )-- the order to fit too
    :param year (default [111, 181]) -- the years to get values for.
    :param x (detault is None) -- if specified regress against this rather than time.
    :param bootstrap (default is None) -- if specified computes bootstrap uncertainties -- assuming guaussin
    :param timeAvg (default is None) -- if specified time-average data with this period.
    :param mask (default is False) -- if True mask data..
    :param fit (default is False) -- if True return fit params
    :param makeCube (default for now if False) -- if True wrap the result as a cube.
    :return: the fitted values at the specified year
    """

    cube = cubeIn
    if timeAvg is not None:
        cube = cubeIn.rolling_window('time', iris.analysis.MEAN, timeAvg)
        # later extract the times in the mid point of the windows..

    # need to figure out which co-ord is time..
    coordNames = [c.standard_name for c in cube.coords()]

    time_points = cube.coord('year').core_points()
    data = cube.data
    if cube.ndim == 1:
        data = data.reshape((len(data), 1))
    timeAxis = cube.coord_dims('time')  # work out which axis is time.
    if timeAxis[0] != 0:  # always want time first...not sure why sometimes a tuple..
        data = np.moveaxis(data, timeAxis, 0)
    shape = data.shape
    npt = np.product(shape[1:])
    data = data.reshape(shape[0], npt)
    if x is None:
        xx = time_points
    else:
        xx = x

    if timeAvg is not None:
        indx = np.arange(0, len(data) + 1, timeAvg)
        xx = xx[indx]
        data = data[indx, :]
    try:
        # watch out when not enough values. Need at least order+1 points.
        try:
            cnt = data.count(0)  # sum all non missing data,
        except AttributeError:
            cnt = np.repeat(shape[0], npt)
        L = cnt > order  # logical where we have enough pts.
        pmsk = np.ma.polyfit(xx, data[:, L], order)  # result on masked subset
        shp = (pmsk.shape[0], data.shape[1])  # shape of result array
        p = np.repeat(np.nan, np.product(shp)).reshape(shp)  # array with ara
        p[:, L] = pmsk
        p = np.ma.masked_invalid(p)
    except ValueError:  # polyfit failed likely because of NaN
        return [None] * order

    if fit:
        return p
    # this fails if year is an int.

    year2 = year
    if len(p.shape) > 1: year2 = np.reshape(year, (np.size(year), 1))
    result = np.polyval(p, year2).squeeze()  # compute values from fit at desired years -- note could extrapolate.
    # reform result..
    rShape = [np.size(year)]
    rShape.extend(shape[1:])
    result = result.reshape(rShape)
    # now mask result
    if mask:
        msk0 = data.mask[0, :].copy()  # mask for first column...
        result = np.ma.masked_array(result, mask=np.broadcast_to(msk0, result.shape))
    # now compute bootstrap...if wanted
    if bootstrap is not None:
        bsValues = []
        for i in range(0, bootstrap):
            npt = len(xx)
            indx = np.random.choice(npt, npt)
            p = np.polyfit(xx[indx], data[indx, :], order)
            bsValues.append(np.polyval(p, year))
        arr = np.array(bsValues)
        var = np.var(arr, 0)
        result = (result, np.sqrt(var))  # append var to the result

    if makeCube:  # wrap data as a cube.
        # easiest approach is to select required years then overwrite the data..
        # won't work if years don't actually exist.
        # should figure out how to create an iris cube from another one.
        tempCube = cube.extract(iris.Constraint(year=year))
        tempCube.data = result.squeeze()
        result = tempCube
    return result


def jitterScatter(df, ax, colours=None, marker='o', s=10):
    """
    Jitter plot a dataframe -- each column separately
    :param df: a dataframe
    :param ax:  the axis to add the scatter to
    :param colours: colours to plot each value
    :param marker:  marker -- marker to use to plot each value
    :param s: size of each marker. (as used by plot -- default is 10)
    :return: the modified axis
    """

    for indx, c in enumerate(df.columns):  # iterate over columns. indx is posn to plot at
        arr = df.loc[:, c].values  # values to plot
        x = np.repeat(indx, len(arr))  # x-coord
        x = x + np.random.randn(len(x)) * 0.1  # abit of jitter
        # iterate over values
        for xv, yv, c, m in zip(x, arr, colours, marker):
            ax.plot(xv, yv, color=c, marker=m, ms=s)

    return ax


def um_long_name(ds, verbose=False, duplicate_Fail=False):
    """
    Generate dict with long_name as key and value the dataArray for that variable.
    This makes it easy to look up variables by their long_name
    :param ds -- a dataset generated from the UM  likely using um_to_nc
    :param verbose -- be verbose if True (default is False)
    :param duplicate_Fail -- fail if duplicate long_name found.
    """
    lookup = dict()
    for var in ds.variables:
        key = ds[var].attrs.get('long_name')
        if duplicate_Fail and (key in lookup):
            raise Exception(f"Duplicate long name found {key} for var:{var}")

        if key is not None:  # got a long name
            lookup[key] = ds[var]
            if verbose:
                print(f"{var} -> {key}")

    return lookup


## make a label object
class plotLabel:
    """
    Class for plotting labels on sub-plots
    """

    def __init__(self, upper=False, roman=False,fontdict={}):
        """
        Make instance of plotLabel class
        parameters:
        :param upper -- labels in upper case if True
        :param roman -- labels use roman numbers if True
        """

        import string
        if roman:  # roman numerals
            strings = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii']
        else:
            strings = [x for x in string.ascii_lowercase]

        if upper:  # upper case if requested
            strings = [x.upper() for x in strings]

        self.strings = strings[:]
        self.num = 0
        self.fontdict=fontdict

    def label_str(self):
        """
        Return the next label
        """
        string = self.strings[self.num] + " )"
        self.num += 1
        self.num = self.num % len(self.strings)
        return string

    def plot(self, ax=None, where=None):
        """
        Plot the label on the current axis.
        :param ax -- axis to plot on. Default is current axis (using plt.gca())
        :param where -- (x,y) tuple saying where  to plot label using axis coords. Default is (-0.03,1.03)
        """

        if ax is None:
            plt_axis = plt.gca()
        else:
            plt_axis = ax
        try:
            if plt_axis.size > 1:  # got more than one element
                for a in plt_axis.flatten():
                    self.plot(ax=a, where=where)
                return
        except AttributeError:
            pass

        # now go and do the actual work!

        text = self.label_str()
        if where is None:
            x = -0.03
            y = 1.03
        else:
            (x, y) = where

        plt_axis.text(x, y, text, transform=plt_axis.transAxes,
                      horizontalalignment='right', verticalalignment='bottom',fontdict=self.fontdict)


"""
Taylor diagram (Taylor, 2001) test implementation.

http://www-pcmdi.llnl.gov/about/staff/Taylor/CV/Taylor_diagram_primer.htm
Modified by Simon Tett, 2016 & 2021.
"""

__version__ = "Time-stamp: <2012-02-17 20:59:35 ycopin>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.ticker as ticker


class TaylorDiagram(object):
    """Taylor diagram: plot model standard deviation and correlation
    to reference (data) sample in a single-quadrant polar plot, with
    r=stddev and theta=arccos(correlation).
    """

    def __init__(self, refstd=1.0, fig=None, rect=111, sdConts=None, label='_', theta_range=None, sd_range=None):
        """Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using mpl_toolkits.axisartist.floating_axes. refstd is
        the reference standard deviation to be compared to.

        :param refstd: reference value. Optional with default 1

        :param fig (optional) Default is None and a new figure will be created if not specified.
        :param rect (optional) sub-figure placement. Default is 111
        :param label (optional) label.
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF
        import matplotlib.pyplot as plt

        self.refstd = refstd  # Reference standard deviation

        if fig is None:
            fig = plt.gcf()
            # fig = PLT.figure()

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = NP.concatenate((NP.arange(10) / 10., [0.95, 0.99]))
        tlocs = NP.arccos(rlocs)  # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)  # Positions
        labels = dict(zip(tlocs, map(str, rlocs)))
        tf1 = GF.DictFormatter(labels)
        # Standard deviation axis extent
        gl2 = GF.MaxNLocator(6) # RMSE
        if theta_range is not None:
            self.theta_min =theta_range[0]
            self.theta_max = theta_range[1]
        else:
            self.theta_min =0.0
            self.theta_max = NP.pi/2

        if sd_range is not None:
            self.smin = sd_range[0]
            self.smax  = sd_range[1]
        else:
            self.smin = 0.0
            self.smax = 1.5 * self.refstd


        extremes = (self.theta_min, self.theta_max,  # 1st quadrant
                         self.smin, self.smax)

        ghelper = FA.GridHelperCurveLinear(tr,
                                           extremes=extremes,
                                           grid_locator1=gl1,
                                           grid_locator2=gl2,
                                           tick_formatter1=tf1,
                                           tick_formatter2=None
                                           )


        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")  # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Normalised Std. Dev.")
        # ax.axis['left'].xticklabel_format(style='plain')

        ax.axis["left"].toggle(ticklabels=True)

        ax.axis["right"].set_axis_direction("top")  # "Y axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["bottom"].set_visible(False)  # Useless
        #ax.axis["right"].set_visible(False)

        # Contours along standard deviations
        ax.grid(False)

        self._ax = ax  # Graphical axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates

        # Add reference point and stddev contour
        # print "Reference std:", self.refstd
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        if sdConts is None:
            sdC = [self.refstd]
        else:
            sdC = sdConts

        for sd in sdC:
            t = NP.linspace(self.theta_min, self.theta_max)
            r = NP.zeros_like(t) + sd
            self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """Add sample (stddev,corrcoeff) to the Taylor diagram. args
        and kwargs are directly propagated to the Figure.plot
        command."""

        l, = self.ax.plot(NP.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta,radius)
        self.samplePoints.append(l)

        return l

    def add_contours(self, levels=5, **kwargs):
        """Add constant centered RMS difference contours."""

        rs, ts = NP.meshgrid(NP.linspace(self.smin, self.smax),
                             NP.linspace(self.theta_min, self.theta_max))
        # Compute centered RMS difference
        rms = NP.sqrt(self.refstd ** 2 + rs ** 2 - 2 * self.refstd * rs * NP.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours
