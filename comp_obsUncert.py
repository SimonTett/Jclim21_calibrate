"""
Compute the uncertainty due to observations.
Does this very simply by setting all values in covariance but the target to value * 1e3
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import PaperLib
import StudyConfig
import pathlib
import scipy.stats


def scale_cov(obsCov, **obs):
    """
    Scale obs covariance for desired obs by scale value
    :param obsCov: Observed covariance
    :param obs: all remaining arguments are treated as kwargs for obs
    :return: Scaled observed covariance
    """

    scaled_cov = obsCov.copy()
    for k, v in obs.items():
        scaled_cov.loc[k, :] *= v
        scaled_cov.loc[:, k] *= v

    return scaled_cov


# stage 0 -- get in the various data we need.


TCRrename = {'TCR4': 'sat_TCR4', 'TCR': 'sat_TCR', 'ts_tppn_land4': 'precip_TCR4'}
# TCRNames =['sat_TCR4','sat_TCR','precip'] # names for TCR runs
TCRnames = TCRrename.keys()
TCRobs = list(TCRrename.values())
std = PaperLib.read_data().loc['Standard']
stdTCR = std.loc[['TCR4', 'TCR']].rename(index={'TCR4': 'T140'})
stdECS = std.loc[['ts_t15_4xCO2']].rename(index={'ts_t15_4xCO2': 'ECS4'})

studyPath = PaperLib.OptClimPath / 'data_files'
jacobianAtm = StudyConfig.readConfig(studyPath / 'jacobian' / 'jac14param_final.json')

obsNames = jacobianAtm.obsNames()
# get a covariance!
json_file = PaperLib.rootData / "7param" / "vierzehn" / "vierzehn_final.json"
config = StudyConfig.readConfig(json_file)
cov = config.Covariances(obsNames=obsNames, scale=True)
jacobianTCR = StudyConfig.readConfig(studyPath / 'coupJac' / 'coupJac10pTCR_final.json')
jacobianECS = StudyConfig.readConfig(studyPath / 'coupJac' / 'coupJac14p_try3_final.json')
paramNames = jacobianTCR.paramNames()  # param names
sevenParam = ['CT', 'EACF', 'ENTCOEF', 'RHCRIT', 'VF1', 'CW_LAND', 'ICE_SIZE']  # 7 param case.
jacAtm = jacobianAtm.runJacobian(normalise=True).Jacobian
jacAtm = jacAtm.sel(parameter=paramNames, Observation=obsNames).to_pandas()  # convert to DataFrame
jacTCR = jacobianTCR.runJacobian(normalise=True).Jacobian. \
    reindex(parameter=paramNames, Observation=['sat_TCR4', 'sat_TCR']).to_pandas(). \
    rename(columns={'sat_TCR4': 'T140', 'sat_TCR': 'TCR'})
jacECS = jacobianECS.runJacobian(normalise=True).Jacobian. \
    reindex(parameter=paramNames, Observation=['sat']).to_pandas(). \
    rename(columns={'sat': 'ECS4'})

stdParam = config.standardParam(paramNames=paramNames, scale=True)
stdParam = stdParam.reindex(jacTCR.index)

covObsErr = cov['CovObsErr'].copy(deep=True)
covIntVar = cov['CovIntVar']

# generate covIntVars for scaling
covObsErr_scale = dict()
scaleV = 1e4
covObsErr_scale['All'] = covObsErr
jacResp = pd.concat([jacTCR, jacECS], axis=1)
stdResp = pd.concat([stdTCR, stdECS])
npt = len(paramNames)
minSamp = 1000

## compute sens case when uncertainty is doubled = sqrt(2) and range increased to -0.5 1.5
# do for 7param and SigPR case.
cols = ['ECS4', 'TCR', 'T140']
for name, params in zip(['7PRx2', 'SigPRx2', 'IceRx2', 'NoIceRx2'],
                        [sevenParam, paramNames, paramNames, paramNames]):
    npts = len(params)
    plimit = np.zeros([2, npts])
    plimit[0, :] = -0.5
    plimit[1, :] = 1.5
    covParams = pd.DataFrame(np.diag(np.ones(npts)), index=params, columns=params)

    if name == 'IceRx2':
        indx = (np.array(params) == 'ALPHAM')
        plimit[0, ~indx] = 0
        plimit[1, ~indx] = 1  # set limits for params except ALPHAM back to 0,1
        covParams = scale_cov(covParams, ALPHAM=2)  # increase all alpha cov values by 2 (=4 on the diagonal)
    elif name == 'NoIceRx2': # restrict everything but ALPHAM
        indx = (np.array(params) == 'ALPHAM')
        plimit[0, indx] = 0
        plimit[1, indx] = 1  # set limits for ALPHAM back to 0,1
        pp = {param: 2 for param in params if param != 'ALPHAM'}
        covParams = scale_cov(covParams, **pp)  # increase all alpha cov values by 2 (=4 on the diagonal)
    else:
        covParams *= 4  # diagonal and want twice...
    paramPrior2 = scipy.stats.multivariate_normal(mean=np.ones(npts) * 0.5,
                                                  cov=covParams)  # prior on normalised parameters is broad.

    covErr = covObsErr + 2 * covIntVar
    paramCov = PaperLib.compParamCov(covErr, jacAtm.loc[params, :])
    # combine with prior on parameters or restricted calculation
    obsParam = scipy.stats.multivariate_normal(mean=stdParam.loc[params], cov=paramCov)  # obs param  covariance.

    prior2 = PaperLib.combCovar(paramPrior2,
                                obsParam)  # prior distribution on parameters used in restricted calculation
    posteriorResp2 = PaperLib.limitParamCov(prior2, stdParam.loc[params], stdResp, jacResp.loc[params, :],
                                            minSamp=minSamp, paramLimit=plimit)
    mn = pd.Series(posteriorResp2.mean, index=jacResp.columns)
    sd = pd.Series(np.sqrt(np.diag(posteriorResp2.cov)), index=jacResp.columns)
    cv = (sd / mn * 100 + 0.05).astype(int)
    print(f"{name:10s}", end="& - &")

    for c in cols:
        print(f" $ {mn.loc[c]:3.2g} \\pm {sd.loc[c]:2.1g} ({cv.loc[c]:2d}\\%)$", end='&')
    print("\\\\")
##
breakpoint()

vars_radn = ['olr', 'rsr', 'netflux']  # raadn variables
vars_trop = ['rh@500', 'temp@500']  # mid-trop variables
vars_sfc = ['lat', 'lprecip', 'mslp']  # all sfc variables
vars_best = ['lprecip', 'rsr', 'netflux']  # guess at best vars
for name, vars in zip(['Sfc', 'Radn', 'Trop', 'Best'], [vars_sfc, vars_radn, vars_trop, vars_best]):
    scale = {c: np.sqrt(scaleV) for c in covObsErr.columns if not (c.split("_")[0] in vars)}
    covObsErr_scale[name] = scale_cov(covObsErr, **scale)
    # complement...
    scale = {c: np.sqrt(scaleV) for c in covObsErr.columns if (c.split("_")[0] in vars)}
    covObsErr_scale['~' + name] = scale_cov(covObsErr, **scale)

for var in vars_radn + vars_trop + vars_sfc:
    scale = {c: np.sqrt(scaleV) for c in covObsErr.columns if not (c.split("_")[0] in var)}
    covObsErr_scale[var] = scale_cov(covObsErr, **scale)
    # complement
    scale = {c: np.sqrt(scaleV) for c in covObsErr.columns if (c.split("_")[0] in var)}
    covObsErr_scale['~' + var] = scale_cov(covObsErr, **scale)

covObsErr_scale['None'] = covObsErr * scaleV
rename = dict(olr='OLR', rsr='RSR', netflux='NET', lat='LAT', lprecip='LP', mslp='SLP')
rename['rh@500'] = 'q500'
rename['temp@500'] = 'T500'

var = dict()
mean = dict()
obsCovar = dict()

paramPrior = scipy.stats.multivariate_normal(mean=np.ones(npt) * 0.5, cov=1)  # prior on normalised parameters is broad.

# add keys to mean and var
for key in covObsErr_scale.keys():
    if key[0] != '~':
        mean[key] = pd.Series(dtype=float)
        var[key] = pd.Series(dtype=float)

for obsName, covObsErrS in covObsErr_scale.items():

    covErr = covObsErrS + 2 * covIntVar
    paramCov = PaperLib.compParamCov(covErr, jacAtm)
    # combine with prior on parameters or restricted calculation
    obsParam = scipy.stats.multivariate_normal(mean=stdParam, cov=paramCov,
                                               allow_singular=True)  # obs param  covariance.
    prior = PaperLib.combCovar(paramPrior, obsParam)  # prior distribution on parameters used in restricted calculation

    obsCovar[obsName] = prior
    # compute TCR &ECS posterior dists
    posteriorResp = PaperLib.compPredictionVar(jacResp, obsParam, stdParam, stdResp)
    posteriorRespR = PaperLib.limitParamCov(prior, stdParam, stdResp, jacResp, minSamp=minSamp)  # restricted

    # print(f"Posterior calc {obsName} {posteriorTCR.mean[0]:3.1f} {np.sqrt(np.diag(posteriorTCR.cov))[0]:3.1f}")
    names = jacResp.columns
    sd = np.sqrt(np.diag(posteriorRespR.cov))
    for indx, n in enumerate(names):
        print(f"R {n} {obsName} {posteriorRespR.mean[indx]:3.1f}  {sd[indx]:3.1f}")
    anti_rename = {n: 'A' + n for n in names}
    mnS = pd.Series(posteriorRespR.mean, index=names)
    varS = pd.Series(np.diag(posteriorRespR.cov), index=names)
    key = obsName
    if obsName[0] == '~':  # "anti"
        key = obsName[1:]
        varS = varS.rename(index=anti_rename)
        mnS = mnS.rename(index=anti_rename)

    var[key] = var[key].append(varS)
    mean[key] = mean[key].append(mnS)
## plot data
rot = 45
ms = 10
varDF = pd.DataFrame(var).rename(columns=rename)
meanDF = pd.DataFrame(mean).rename(columns=rename)

sd = np.sqrt(varDF)

# just show T140 Standard Deviation.
fig, axes = plt.subplots(nrows=1, ncols=1, num='Obs_cont', figsize=[4.3, 3], clear=True, sharex='all')
sd.loc['T140', :].plot.bar(rot=rot, ax=axes, color='grey')
sd.loc['AT140', :].plot.bar(rot=rot, ax=axes, color='red', align='edge')
for x in [0.7, 4.7, 7.7, 9.7, 12.7]:
    axes.axvline(x, linestyle='dashed', linewidth=1, color='black')

sd_limits = [0, 1.5]
axes.set_ylim(sd_limits)
axes.yaxis.set_ticks(np.linspace(sd_limits[0], sd_limits[1], 6, endpoint=True))
axes.axhline(sd.loc['T140', 'All'], linestyle='dashed', linewidth=2, color='grey')
axes.set_title('T140 Standard Deviation')
axes.set_ylabel("Std Dev (K)")
axes.set_xlabel("Variable")

fig.tight_layout()
fig.show()
PaperLib.saveFig(fig)

# code for 2x2 plots   
fig, axes = plt.subplots(nrows=2, ncols=2, num='Obs_cont_all', figsize=[8.5, 6], clear=True, sharex='all')
axes_TCR = axes[0]
axes_ECS = axes[1]

sd.loc['T140', :].plot.bar(rot=rot, ax=axes_TCR[1], color='grey')
sd.loc['AT140', :].plot.bar(rot=rot, ax=axes_TCR[1], color='red', align='edge')
meanDF.loc['T140', :].plot(rot=rot, ax=axes_TCR[0], linewidth=1, color='grey', marker='o', ms=ms)
meanDF.loc['AT140', :].plot(rot=rot, ax=axes_TCR[0], linewidth=1, color='red', marker='o', ms=ms)
sd.loc['ECS4', :].plot.bar(rot=rot, ax=axes_ECS[1], color='grey')
sd.loc['AECS4', :].plot.bar(rot=rot, ax=axes_ECS[1], color='red', align='edge')
meanDF.loc['ECS4', :].plot(rot=rot, ax=axes_ECS[0], linewidth=1, color='grey', marker='o', ms=ms)
meanDF.loc['AECS4', :].plot(rot=rot, ax=axes_ECS[0], linewidth=1, color='red', marker='o', ms=ms)
# decorate axis
label = PaperLib.plotLabel()
for axes, row_title, sd_limits in zip([axes_TCR, axes_ECS], ['T140', 'ECS4'], [[0, 1.5], [0, 3.]]):
    for ylab, title, ax in zip(['Mean (K)', 'Std Dev (K)'], ['Best Estimate', 'Standard Deviation'], axes):
        ax.set_ylabel(ylab)
        ax.set_title(title + " " + row_title)
        if title == 'Standard Deviation':
            ax.set_ylim(sd_limits)
            ax.yaxis.set_ticks(np.linspace(sd_limits[0], sd_limits[1], 6, endpoint=True))
            ax.axhline(sd.loc[row_title, 'All'], linestyle='dashed', linewidth=2, color='grey')
        else:
            ax.axhline(meanDF.loc[row_title, 'All'], linestyle='dashed', linewidth=2, color='grey')

        for x in [0.7, 4.7, 7.7, 9.7, 12.7]:
            ax.axvline(x, linestyle='dashed', linewidth=1, color='black')
        label.plot(ax)

fig.tight_layout()
fig.show()
PaperLib.saveFig(fig)
