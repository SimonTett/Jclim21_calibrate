import StudyConfig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray
import PaperLib
import os
import numpy.random
import seaborn as sns
import scipy.stats
import adjustText  # library to adjust text boxes so they don't overlap.  conda install -c conda-forge adjusttext


def extIndx(df, name):
    """

    :param df:
    :param name:
    :return:
    """

    L = df.index.str.match(name)
    result = df[L]
    indx = result.index.str.replace('^' + name, '')
    result.index = indx
    return result


def dfInv(df, rcond=1e-15):
    """
    Compute inverse (using pinv) wrapping things backup as a dataframe...
    :param df: dataframe
    :return: return inverse
    """

    inv = np.linalg.pinv(df, rcond=rcond)
    inv = pd.DataFrame(inv, index=df.index, columns=df.columns)
    return inv


def dfeig(df):
    """
    Compute inverse (using pinv) wrapping things backup as a dataframe...
    :param df: dataframe
    :return: return inverse
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
    :return: Diagonal
    """

    result = pd.Series(np.diag(df), index=df.columns)
    return result


def coupUncert(covErr, deltaParam, paramSamp, nSampRef=1, paramZero=None):
    """
    Compute uncertainty due, to random error in the Jacobian estimate.
    :param covErr: error covariance. -- a pandada dataframe with the same index and column values. Size MxM
    :param deltaParam: Param delta -- a pandas series Size P
    :param paramSamp: Sampled parameters. -- index is sample number, column parameter size NxP
    :return: sampled set. Size N*M
    """
    # step 1 work out how many samples we have and generate them.
    nsamp = paramSamp.shape[0]  # number of samples needed.
    nparam = len(deltaParam)  # number of parameters
    npredict = covErr.shape[0]
    # generate the data.
    numpy.random.seed(123456789)  # always make sure have the same RNG seed -- note different from other RNG use.
    noiseRef = numpy.random.multivariate_normal(np.zeros(npredict), covErr / nSampRef,
                                                (nsamp, 1))  # noise for reference case
    noise = numpy.random.multivariate_normal(np.zeros(npredict), covErr,
                                             (nsamp, nparam))  # generate noise for perturbed cases
    if paramZero is not None:
        # set noise where paramZero is T to 0.0 -- no change.
        noiseRef[:, :, :] = 0.0
        noise[:, paramZero, :] = 0.0

    # now compute noisy Jacobian...
    noiseJac = (noise - noiseRef) / deltaParam.values.reshape(1, -1, 1)  # should be size MxNxP
    # and then apply it to compute noisy climate sensitivity (or what ever)
    # import pdb; pdb.set_trace()
    # can't see how to do the matrix multiplications in one go... :-(
    sampUncert = np.zeros((npredict, nsamp))
    for indx in range(0, nsamp):
        sampUncert[:, indx] = noiseJac[indx, :, :].transpose() @ paramSamp[indx, :]
    # compute mean and covariance and bundle up into pandas series and dataframes.
    mean = pd.Series(sampUncert.mean(1), index=covErr.index)
    cov = pd.DataFrame(np.cov(sampUncert), index=covErr.index, columns=covErr.columns)
    # hack to turn off
    # mean.iloc[:] =0.0
    # cov.iloc[:,:]=0.0
    return sampUncert, (mean, cov)


def compParamCov(obsCov, jacAtm):
    """
    Compute the parameter covariance error
    :param obsCov:
    :param jacAtm:
    :return: parameter covariance
    """

    invCov = dfInv(obsCov)
    invCov = jacAtm @ invCov
    Hessian = invCov @ jacAtm.transpose()  # matrix to be inverted
    invM = dfInv(Hessian)
    transMat = invM @ invCov  # transformation matrix
    paramCov = (transMat.dot(covErr)).dot(transMat.T)  # covariance of parameters

    return paramCov


def compPredictionVar2(jacCoup, paramNorm, stdParam, stdCoup):
    """
    Compute the covariance of predictions
    :param jacCoup: Coupled Jacobian
    :param paramNorm: frozen scipy.stats.multivariate_normal object
    :param stdMn: standard parameter setting.
    :return: a scipy.stats.multivariate_normal object with the mean and std dev.
    """
    mn = stdCoup + jacCoup.transpose() @ (stdParam - paramNorm.mean)
    cov = (jacCoup.values.T @ paramNorm.cov) @ jacCoup.values
    return scipy.stats.multivariate_normal(mean=mn, cov=cov, allow_singular=True)


def limitParamCov2(paramDist, stdParam, stdCoup, jacCoup, minSamp=1000, verbose=False):
    """
    Compute the covariance of the coupled model when parameters are limited to 0..1
    :param paramDist: scipy.stats.multivariate_normal object -- will be used to generate samples
    :param stdParam: standard parameters
    :param stdCoup: standard model coupled response
    :param minSamp: Minimum number of samples to generate.
    :param verbose: Optional -- default False. If True be a bit verbose.
    :return: a multivariate normal dist.
    """

    numpy.random.seed(123456)  # always make sure have the same RNG seed

    sampCount = 0  # no of samples
    nSamp = int(minSamp)

    samp = np.zeros((0, len(stdParam)))
    cnt = np.zeros(len(stdParam))  # total number of OK samples for each parameter
    nGen = 0  # no of samples generated

    while sampCount < minSamp:
        lsamp = paramDist.rvs(nSamp)
        nGen += nSamp  # running count of how many samples have been generated.
        L = ((lsamp >= 0.0) & (lsamp <= 1.0))  # where those are OK
        cnt += L.sum(0)  # increment count of OK for individual parameters
        OK = L.all(axis=1)  # good cases
        sampCount += OK.sum()  # total no of samples we've generated.
        samp = np.append(samp, lsamp[OK, :], 0)  # list of OK samples!
        nSamp *= 10  # next time generate 10 times as many.

    fractParamOK = cnt / float(nGen)  # what fraction of parameters are good..
    fractOK = sampCount / float(nGen)  # what fraction of samples are good
    if verbose:
        print(f"Sample Fract. {fractOK} Paramfract: {fractParamOK} ")
    # compute the (linear) climate change for the good cases...

    samp = pd.DataFrame(samp, columns=stdParam.index)  # make it a dataFrame
    change = (samp - stdParam) @ jacCoup  # compute the linear climate sensitivity
    covDelta = np.cov(change.transpose())  # compute covariance
    result = scipy.stats.multivariate_normal(mean=stdCoup + change.mean(), cov=covDelta, allow_singular=True)

    return result


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
    return scipy.stats.multivariate_normal(mean=mn, cov=covar)


## start of code.

corrCov = False  # if True produce a correlated covariance.
scaleNonRad = None  # How much to scale each element of the non radiation obs error covariance elements. None means no scaling
JMGensData = pd.read_csv(os.path.join('data', 'JMG_ens_data.csv'), index_col=0)  # read the ensemble data.
# it is quite a small ensemble and missing the land precip...
ECSrename = {'ECS_4xCO2': 'sat', 'ts_tppn_4xCO2': 'precip'}
ECSnames = ECSrename.keys()
ECSobs = list(ECSrename.values())
# Stdnames = ['ECS_4xCO2','ts_tppn_4xCO2']
ECScov = np.cov(JMGensData.loc[:, ECSnames].rename(index=ECSrename).values.T)
ECScov = pd.DataFrame(ECScov, index=ECSnames, columns=ECSnames)
StdECS = JMGensData.loc[:, ECSnames].mean().rename(ECSrename)
# ECSnames=['sat', 'precip'] # names from ECS runs

TCRrename = {'TCR4': 'sat_TCR4', 'TCR': 'sat_TCR', 'ts_tppn_land4': 'precip_TCR4'}
# TCRNames =['sat_TCR4','sat_TCR','precip'] # names for TCR runs
TCRnames = TCRrename.keys()
TCRobs = list(TCRrename.values())
stdTCR = PaperLib.read_data().loc['Standard', TCRnames]
stdTCR = stdTCR.rename(index=TCRrename)

studyPath = os.path.join(PaperLib.OptClimPath, 'data_files')
jacobianAtm = StudyConfig.readConfig(os.path.join(studyPath, 'jacobian', 'jac14param_final.json'))

obsNames = jacobianAtm.obsNames()
# get a covariance!
json_file = os.path.join(PaperLib.rootData, "7param", "vierzehn", "vierzehn_final.json")
config = StudyConfig.readConfig(json_file)
cov = config.Covariances(obsNames=obsNames, scale=True)
jacobianECS = StudyConfig.readConfig(os.path.join(studyPath, 'coupJac', 'coupJac14p_try3_final.json'))
jacobianTCR = StudyConfig.readConfig(os.path.join(studyPath, 'coupJac', 'coupJac10pTCR_final.json'))

## compute correction to Jacobian for ECS (going from 1 ref sim to 7 sims)
correction = StdECS - jacobianECS.simObs().loc['jc001', ECSrename.values()]
deltaParam = jacobianECS.steps(normalise=True)  # scale by deltaParam
# compute correction -- unpacking from dataframe to get numpy broadcasting then converting back to dataframe..
correction = pd.DataFrame(correction.values.reshape(-1, 1) / deltaParam.values.reshape(1, -1), index=StdECS.index,
                          columns=deltaParam.index)
correction = correction.transpose()

## work out a new covariance by hacking the error covariance.
import copy

covObsErr = cov['CovObsErr'].copy(deep=True)
covErr = cov['CovTotal']
covIntVar = cov['CovIntVar']

figName = 'linearECSTCR'
if corrCov:
    figName += '_corr'
    sd = np.sqrt(dfDiag(covObsErr))
    for obs in ['olr', 'rsr', 'lat', 'lprecip', 'temp@500', 'rh@500', 'mslp', 'netflux']:
        L = sd.index.str.match(obs)
        corrSubCov = np.outer(sd[L], sd[L])
        covObsErr.loc[L, L] = corrSubCov

if scaleNonRad is not None:
    figName += '_scale'
    for obs in ['lat', 'lprecip', 'temp@500', 'rh@500', 'mslp', 'netflux']:
        L = covObsErr.index.str.match(obs)
        covObsErr.loc[L, :] *= np.sqrt(scaleNonRad)  # scale columns
        covObsErr.loc[:, L] *= np.sqrt(scaleNonRad)  # and rows

covErr = covObsErr + 2 * covIntVar

sdIntVar = np.sqrt(np.diag(covIntVar))
pp = jacobianAtm.paramNames()

paramScale = []
paramMn = []
obsScale = []
obsMn = []

# compute parameters that appear to be sufficiently sensitive to temperature change...
jacECS = jacobianECS.runJacobian(normalise=True).Jacobian.to_pandas()
jacECS += correction
# sort everything so in order..
ecs = np.abs(jacECS.loc[:, 'sat'])
ECSindx = ecs.sort_values(ascending=False).index

jacTCR = jacobianTCR.runJacobian(normalise=True).Jacobian.reindex(parameter=pp, Observation=TCRobs).to_pandas()
jacTCR = jacTCR.fillna(0.0)  # set all nan to 0

delta = (jacECS.transpose() * deltaParam).transpose() / np.sqrt(dfDiag(ECScov))  # 1 member ensemble for clim-sens
Lsmall = abs(delta.loc[:, 'sat']) < 1  # (10-90%)
paramSig = delta[Lsmall].index.values

mnSens = {}  # the mean sens
covSens = {}  # the covariance of the sens
covNoise = {}  # estimated covariance of the noise.
postDist = {}  # estimated posterior distributions.
postDistTCR4 = {}  # estimated posterior distributions for TCR4
sevenParams = pp[0:7]
for paramNames, paramTitle in zip([sevenParams, pp[0:13], pp, pp], ['7P', 'NoIce', '14P', 'SigP']):
    jacAtm = jacobianAtm.runJacobian(normalise=True).Jacobian
    # jacAtmVar = jacobianAtm.runJacobian(normalise=True).Jacobian_var
    jacAtm = jacAtm.sel(parameter=paramNames, Observation=obsNames).to_pandas()  # convert to DataFrame
    # get coupled values
    jacECS = jacobianECS.runJacobian(normalise=True).Jacobian.sel(parameter=paramNames, Observation=ECSobs).to_pandas()
    jacECS += correction.loc[paramNames, :]
    # sortby descending order of ECS
    jacECS = jacECS.reindex(index=ECSindx).dropna()
    jacAtm = jacAtm.reindex(index=jacECS.index)

    jacTCR = jacobianTCR.runJacobian(normalise=True).Jacobian.reindex(parameter=paramNames,
                                                                      Observation=TCRobs).to_pandas()
    jacTCR = jacTCR.reindex(jacECS.index)
    jacTCR = jacTCR.fillna(0.0)  # set all nan to 0
    # jacTCR = jacTCR.rename({'sat_TCR4':'sat'},axis='columns')
    if paramTitle == 'SigP':
        jacECS.loc[Lsmall, :] = .0  # set small sens to zero.
        jacTCR.loc[Lsmall, :] = .0  # set small sens to zero.
        paramZero = Lsmall  # got small values
        print("Setting small Jac values to zero")
    else:
        paramZero = None

    stdParam = config.standardParam(paramNames=paramNames, scale=True)
    stdParam = stdParam.reindex(jacECS.index)
    # deltaParam = jacobianECS.steps(paramNames=paramNames,normalise=True)
    paramCov = compParamCov(covErr, jacAtm)
    # combine with prior for restricted calculation
    npt = paramCov.shape[0]
    prior = scipy.stats.multivariate_normal(mean=np.ones(npt) * 0.5, cov=1)  # prior on normalised parameters is broad.
    obsParam = scipy.stats.multivariate_normal(mean=stdParam, cov=paramCov)  # obs param  covariance.
    prior = combCovar(prior, obsParam)  # prior distribution on parameters used in restricted calculation

    # compute TCR posterior dist
    posteriorTCR = compPredictionVar2(jacTCR, obsParam, stdParam, stdTCR)
    posteriorTCRR = limitParamCov2(prior, stdParam, stdTCR, jacTCR)

    # and for 4xCO2
    posteriorCoup = compPredictionVar2(jacECS, obsParam, stdParam, StdECS)
    posteriorCoupR = limitParamCov2(prior, stdParam, StdECS, jacECS)

    postDist[paramTitle] = (obsParam, posteriorCoup, jacECS, posteriorTCR, jacTCR)
    postDist[paramTitle + "R"] = (prior, posteriorCoupR, jacECS, posteriorTCRR, jacTCR)
    print("Posterior calc ", posteriorCoup.mean, np.sqrt(np.diag(posteriorCoup.cov)))

# make dataFrames
paramScale = pd.DataFrame(paramScale)
paramMn = pd.DataFrame(paramMn)
obsScale = pd.DataFrame(obsScale)
obsMn = pd.DataFrame(obsMn)
## now make some plots.
width = 0.8
rot = 40
fontsize = 'x-small'

## plot mean and uncertainty

error_kw = dict(ecolor='black', capsize=10, capthick=2)
# sd = np.sqrt(pd.DataFrame({k:dfDiag(v) for k,v in covSens.items()})).transpose()
sd = []
mn = []
for k, v in postDist.items():
    mn_series = pd.Series(v[1].mean, index=v[2].columns)
    mn_series = mn_series.append(pd.Series(v[3].mean, index=v[4].columns)).rename(k)
    # and the SD
    sd_series = pd.Series(np.sqrt(np.diag(v[1].cov)), index=v[2].columns)
    sd_series = sd_series.append(pd.Series(np.sqrt(np.diag(v[3].cov)), index=v[4].columns)).rename(k)
    mn.append(mn_series)
    sd.append(sd_series)

mn = pd.DataFrame(mn)
# set TCR cases where not enough params to missing

cols = [c for c in mn.columns if 'TCR' in c]

rows = ['NoIce', 'NoIceR', '14P', '14PR']  # no TCR values defined

mn.loc[rows, cols] = np.nan
sd = pd.DataFrame(sd)

## save the data for subsequent use
mn.to_csv(PaperLib.dataPath/'mn_change.csv')
sd.to_csv(PaperLib.dataPath/'sd_change.csv')
## What to plot for final calculations.
keys = ['7P', '14P', '7PR', '14PR', 'NoIce', 'NoIceR', 'SigP', 'SigPR']
# keysPlot = [keys[k] for k in [0,2,4,6,8,10,5,11]]

cv = (100 * sd / mn + 0.5).fillna(-10000).astype(int)  # coeft of variation.
# print out the values
print("-------------------------------------------")
print("xxxxxxxxxxxxx %s xxxxxxxxxxxx" % figName)
for indx, k in enumerate(keys):  # formatted for inclusion in a latex table
    print(f"{k:10.10s} & -  ", end='&')
    for var in ['sat', 'sat_TCR', 'sat_TCR4']:
        print(f" $ {mn.loc[k, var]:3.2g} \\pm {sd.loc[k, var]:2.1g} ({cv.loc[k, var]:2d}\\%)$", end='&')
    print("\n")
print("=" * 80)

## plot things related to the Jacobian...
label = PaperLib.plotLabel()
annot_kws = {'fontsize': 'x-small'}  # annotation control in sns.heatmap

rotation = 45
fig, ax = plt.subplots(2, 2, num=figName, figsize=[8.5, 6], clear=True)
(axJacATm, axParamCov, axJac, axSens) = ax.flatten()
axis = (axJacATm, axParamCov, axJac, axSens)
# plot titles & add labels.
for a, title in zip(axis, ['Normalised Atmospheric Jacobian', r'Parameter Covariance $\times 10$',
                           'Normalised Coupled Jacobians', 'Linear Responses']):
    a.set_title(title)
    label.plot(a)

# plot a heatmaps for the atm Jacobian and the param covariance matrix.
#sns.set(font_scale=1)
normJacAtm = jacAtm / np.sqrt(dfDiag(covErr))  # normalise by error.
renameDict = PaperLib.shortNames(paramNames=normJacAtm.index, obsNames=normJacAtm.columns, mark=sevenParams)
normJacAtm = normJacAtm.rename(index=renameDict, columns=renameDict)
levels = [-100, -50, -20, -10, -5, -2, 2, 5, 10, 20, 50, 100]
import matplotlib.colors as mplc

norm = mplc.BoundaryNorm(levels, 12)
cmap = sns.color_palette("RdBu_r", 12)
htmap = sns.heatmap(normJacAtm, center=0, ax=axJacATm, linewidths=1, cmap=cmap, annot=True,
                    fmt='2.0f', cbar=False, annot_kws=annot_kws,  # norm=norm,
                    xticklabels=1, yticklabels=1, vmin=-50, vmax=50, cbar_kws={'extend': 'both'})
# split into regions...
for x, rgn in zip([7, 14, 20], ('NHX', 'Tropics', 'SHX')):
    axJacATm.axvline(x, linewidth=2, linestyle='dashed', color='black')
    axJacATm.annotate(rgn, (x - 3.5, 12.0), fontsize='xx-large', alpha=0.5, color='black', ha='center')
# axJacATm.axhline(7.0,linewidth=2,linestyle='dashed',color='black') # as we sort the parameters then seven doesn't make sense.
htmap.set_xticklabels(htmap.get_xticklabels(), rotation=rotation, fontsize='small')
axJacATm.set_xlabel(None)
axJacATm.set_ylabel(None)
# hack

case = '14P'
indx = postDist[case][2].index
paramCov = pd.DataFrame(postDist[case][0].cov, index=indx, columns=indx)
renameDict = PaperLib.shortNames(paramCov.index, mark=sevenParams)
paramCov = paramCov.rename(index=renameDict, columns=renameDict)

htmap = sns.heatmap(paramCov * 10, center=0, ax=axParamCov, cmap='RdBu',
                    annot_kws=annot_kws,
                    annot=True, fmt='3.1f', cbar=False, xticklabels=1, yticklabels=1)

htmap.set_xticklabels(htmap.get_xticklabels(), rotation=rotation, fontsize='small')
xlim = htmap.get_xlim()
ylim = htmap.get_ylim()
axParamCov.plot(xlim, ylim[::-1], linewidth=2, color='gray', alpha=0.5)
axParamCov.set_xlabel(None)
axParamCov.set_ylabel(None)
# plot ECS4 & TCR4
case = '14P'
jj = [postDist[case][2].loc[:, 'sat'].rename('ECS4')]  # extract ECS
jj.append(postDist[case][4].loc[:, 'sat_TCR4'].rename('T140'))  # extract TCR
jj = pd.DataFrame(jj)  # convert list of series into dataframe.
jj = jj.rename(columns=renameDict)
jj.T.plot.bar(ax=axJac, rot=rotation, color=['red', 'blue'], legend=True)

axJac.set_ylabel("4xCO$_2$ warming")
axJac.set_xlabel(None)
# axJac.tick_params('x',bottom=False,labelbottom=False) # remove xticks

# plot mean and uncertainty

width = 0.4
error_kw = dict(ecolor='black', capsize=6, capthick=2)
mn.loc[keys, 'sat'].plot.bar(ax=axSens, yerr=2 * sd.loc[keys, 'sat'], rot=rotation, color='red', error_kw=error_kw,
                             position=0.5, width=width)
mn.loc[keys, 'sat_TCR4'].plot.bar(ax=axSens, yerr=2 * sd.loc[keys, 'sat_TCR4'], rot=rotation, color='blue',
                                  error_kw=error_kw, position=0.0, width=width)
axSens.set_ylabel(None)
axSens.set_ylim(4.0, 9.0)
axSens.set_xlabel(None)

fig.tight_layout()
fig.show()
PaperLib.saveFig(fig)

## Make some Supp Figures...

figScatter, axis = plt.subplots(1, 2, num='ScatterTCR', clear=True, figsize=PaperLib.fsize)
(axSim, axJac) = axis

j = jacobianTCR.runJacobian(normalise=True).Jacobian.to_pandas().loc[:, ['sat_CTL', 'sat_TCR4']]
j = j.rename(index=renameDict)
steps = jacobianTCR.steps(normalise=True).rename(renameDict)
jj = (j.T * steps).T
err = 1.0 / steps * np.sqrt(2) * 0.06  # error

j.plot.scatter('sat_CTL', 'sat_TCR4', ax=axJac, c='black', s=40, yerr=2 * err)
axJac.set_xlabel(r'Normalised Control $\Delta$ GMSAT (K)')
axJac.set_ylabel(r'Normalised $\Delta$ TCR4 (K)')
axJac.set_title("Control GMSAT vs TCR4 Normalised Jacobian")
axJac.set_ylim(-5.5, 5.5)
axJac.set_xlim(-18, 18)

textList = []
for indx, (n, r) in enumerate(j.iterrows()):
    xy = np.array((r.sat_CTL, r.sat_TCR4))
    textList.append(
        axJac.text(xy[0], xy[1], r.name, ha='center', fontsize='small'))  ##,xy,xytext=xytext,arrowprops={'width':4}))

fit = np.polyfit(j.sat_CTL, j.sat_TCR4, 1)
xlim = axJac.get_xlim()
x = np.linspace(xlim[0], xlim[1])
y = fit[0] + fit[1] * x
axJac.plot(x, y, linewidth=2, color='grey', alpha=0.5)
# now adjust the text labels
adjustText.adjust_text(textList, ax=axJac, autoalign='x')

# now plot the actual changes -- just multiply by actual change.


jj.plot.scatter('sat_CTL', 'sat_TCR4', ax=axSim, c='black', s=20, yerr=0.16)
axSim.set_xlabel(r'Control $\Delta$ GMSAT (K)')
axSim.set_ylabel(r'$\Delta$ TCR4 (K)')
axSim.set_title("Simulated change in Control SAT and TCR4")

axSim.set_xlim(-3, 3)
axSim.set_ylim(-0.75, 0.75)
textList = []
for indx, (n, r) in enumerate(jj.iterrows()):
    xy = np.array((r.sat_CTL, r.sat_TCR4))
    textList.append(
        axSim.text(xy[0], xy[1], r.name, ha='center', fontsize='small'))  ##,xy,xytext=xytext,arrowprops={'width':4}))

xlim = axJac.get_xlim()
x = np.linspace(xlim[0], xlim[1])
axSim.fill_between(x, -0.17, 0.17, color='grey', alpha=0.5)
# now adjust the text labels
adjustText.adjust_text(textList, ax=axSim, autoalign='x')
lab = PaperLib.label()
for a in axis:
    PaperLib.plotLabel(a, lab)

figScatter.show()
PaperLib.saveFig(figScatter)
##
