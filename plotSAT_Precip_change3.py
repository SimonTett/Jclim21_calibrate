"""
Plot SAT change & Zonal-Mean changes.
"""
import pandas as pd
import PaperLib
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import iris
import iris.plot
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import pathlib

gmConstraint = iris.Constraint(site_number=3.0)  # constraint for g-m


def compDelta(ctl, force, file, realization=None):
    """
    Compute the difference between forced and ctl values
    :param ctl: directory where ctl data is
    :param force: directory where forced data is
    :param file: filename to use
    :param
    :return: data

    """
    if ctl is None or force is None:
        print("Ctl is ", ctl)
        print("force is ", force)
        return None
    try:
        if file == 'NetFlux':  # special code for net flux
            delta = compDeltaFlux(ctl, force)
        else:
            ctlData = PaperLib.read_pp(os.path.join(ctl, file), realization=realization)
            forceData = PaperLib.read_pp(os.path.join(force, file), realization=realization)
            delta = PaperLib.comp_delta(forceData, ctlData)
            if delta.shape[1] == 3: delta = delta.extract(gmConstraint)
            if delta is None:
                raise Exception("woops")
    except IOError:
        print("IOError returning None for ctl: %s Force: %s" % (ctl, force))
        delta = None

    return delta


def compDeltaFlux(ctl, force):
    """
    Compute difference in netflux
    :param ctl: directory of control
    :param force: directory of forcing
    :return: difference in net flux between forced and ctl simultions
    """
    ctlNF = PaperLib.comp_net(ctl)
    forceNF = PaperLib.comp_net(force)
    deltaNF = PaperLib.comp_delta(forceNF, ctlNF)
    return deltaNF


def comp_multi_mn(variable, allTsData, control):
    """
    Compute mean from ctl  for given variable
    :param variable -- name of variable to process
    :param allTsData:  Pandas object containing TS info -- only used for keys and TCR/TCR4
    :param deltaZM: dict containing iris objects of actual data...
    :return:  tupple of estimate value at 2XCO2 & 4xCO2 scaled by TCR & TCR respectively.
    """
    mn = []
    for k in allTsData.index:
        model = control[k][variable]
        import pdb
        # pdb.set_trace()
        if model is None:
            continue  # skip rest of loop and go onto next case

        m = model.data.mean(axis=0)
        mn.append(m)

    mn = np.ma.array(mn)  # "compress" to masked array
    return mn


def comp_multi_fit(variable, delta):
    """
    Compute multi fits for given variable
    :param variable -- name of variable to process
    :param delta: dict containing iris objects of actual data...
    :return:  tuple of estimate value at 2XCO2 & 4xCO2 scaled by TCR & TCR respectively.
    """

    fit = []
    fit4 = []
    for k,v in delta.items():
        model = v[variable]
        if model is None:
            continue  # go onto next case

        f = PaperLib.comp_fit(model) # fit trends.
        fit.append(f[0, :])
        fit4.append(f[1, :])
        y = model.coord('latitude').points  # compute co-ords

    fit = np.ma.masked_invalid(np.ma.array(fit))  # "compress" to masked array
    fit4 = np.ma.masked_invalid(np.ma.array(fit4))  # ditto

    return (fit, fit4, y)


def dirs(series, extn='.000100', perDictTimeCache=None):
    """

    :param series: series with information on model
    :param perDictTimeCache:  if not none use this as root path and assume cache dir is set on a per dict basis.
    :return: ctl and onePercent dirs
    """
    if perDictTimeCache is not None:
        ctlDir = pathlib.Path(series.Control)
        ctlDir = (perDictTimeCache / ctlDir / "A") / (ctlDir.name + extn)
        onePerDir = pathlib.Path(series.OnePercent)
        onePerDir = (perDictTimeCache / onePerDir / "A") / (onePerDir.name + extn)
    else:
        ctlDir = pathlib.Path(os.path.join(PaperLib.time_cache, os.path.basename(series.Control) + '.000100'))
        onePerDir = pathlib.Path(os.path.join(PaperLib.time_cache, os.path.basename(series.OnePercent) + '.000100'))

    return ctlDir, onePerDir


def compTrends(tsData, LSmask=None, perDictTimeCache=None):
    """
    Compute trends
    :param tsData:
    :param LSmask -- land/seas mask field
    :param perDictTimeCache -- if non Note cached ummonitor files are in model directory structure with perDictTimeCache being the root directory
    :return:
    """

    landRatio = LSmask.collapsed('longitude', iris.analysis.MEAN).data  # ls mask is 1 for land, 0 for ocean
    # Forget iris at this point...
    ocnRatio = 1.0 - landRatio.data  # then do 1- it!
    # Work around because iris sucks (or because I don't understand it..)

    deltaZM = dict()
    control = dict()
    count = 0
    for name, series in tsData.iterrows():
        deltaZM[name] = dict()
        control[name] = dict()
        # make sure ocean gets processed last...
        varsToProcess = ('t15', 't15_land', 't15_max_land', 't15_min_land', 't15_ocean',
                         'slp', 'slp_land', 'slp_ocean', 'pstar', 'pstar_land', 'pstar_ocean')

        for var in varsToProcess:
            timeCacheDir = PaperLib.time_cache  # default cache dir
            ctlDir, onePerDir = dirs(series, perDictTimeCache=perDictTimeCache)
            # all these files have been produced by gen_ummonitor and are in the time-cache dir.
            # hacking in ocean case...
            if '_ocean' in var:

                base = var.split('_')[0]
                # work out ocean from full and land only zm. ZM = fL+(1-f) O => O = (ZM-f L)/(1-f)
                # where ZM -- full zonal mean, f = fraction ocm, L = land ZM, O = ocean ZM
                # One complication is that 0*Nan = Nan...
                for delta in (deltaZM, control):
                    delta[name][var] = delta[name][base].copy()
                    lnd = np.nan_to_num(landRatio * delta[name][base + '_land'].data.filled(0.0))
                    # set missing to zero

                    delta[name][var].data = (delta[name][base].data - lnd) / ocnRatio
                # control[name][var] = control[name][base].copy()
                # control[name][var] = (control[name][base].data-landRatio*control[name][base+'_land'].data)/ocnRatio
            else:
                ctl = PaperLib.readFile(ctlDir, var + '.pp', realization=count)
                diff = compDelta(ctlDir, onePerDir, var + '.pp', realization=count)
                deltaZM[name][var] = diff
                control[name][var] = ctl

        # compute the % change in mean and annual max precip. These are all full fields.
        for var, crit in zip(('precip', 'precip_ocean', 'precip_land', 'precip_max_land'), (1e-5, 1e-5, 1e-5, 1e-6)):
            ctlDir, onePerDir = dirs(series, perDictTimeCache=perDictTimeCache)


            if '_ocean' not in var:  # ocean gets computed from full field.
                ctlPrecip = PaperLib.readPP(ctlDir, var + '.pp', realization=count)
                onePerPrecip = PaperLib.readPP(onePerDir, var + '.pp', realization=count)
            else:  # ocean field so read it raw data.
                var2 = var.replace('_ocean', '')
                ctlPrecip = PaperLib.readPP(ctlDir, var2 + '.pp', realization=count)
                onePerPrecip = PaperLib.readPP(onePerDir, var2 + '.pp', realization=count)

            # work out trend
            ctlTrend = PaperLib.comp_fit(ctlPrecip, order=2, makeCube=True, year=179)
            onePerTrend = PaperLib.comp_fit(onePerPrecip, order=2, makeCube=True, year=179)
            # mask data
            ctlTrend.data = np.ma.masked_less(ctlTrend.data, crit, copy=True)
            onePerTrend.data = np.ma.masked_less(onePerTrend.data, crit, copy=True)
            doMask = (onePerTrend.data.ndim == 2)
            if ('_ocean' in var) and (doMask):  # mask data
                ctlTrend.data = np.ma.masked_array(ctlTrend.data, mask=(LSmask.data == 1))
                onePerTrend.data = np.ma.masked_array(onePerTrend.data, mask=(LSmask.data == 1))
            elif ('_land' in var) and (doMask):  # mask data
                ctlTrend.data = np.ma.masked_array(ctlTrend.data, mask=(LSmask.data == 0))
                onePerTrend.data = np.ma.masked_array(onePerTrend.data, mask=(LSmask.data == 0))

            ratio = onePerTrend / ctlTrend
            if doMask:
                ratio = ratio.collapsed('longitude', iris.analysis.MEAN)
                ctlTend = ctlTrend.collapsed('longitude', iris.analysis.MEAN)
            deltaZM[name][var] = ratio
            control[name][var] = ctlTrend
        print(f"Name is {name}")
        count += 1

    return (deltaZM, control)


def comp_delta(allTs, extn=".000100", fileName='sat.pp', perDictTimeCache=None):
    """
    Compute change in sfc temp
    :param allTs:
    :return:
    """

    delta = iris.cube.CubeList()
    print("Starting to process data")
    count = 0
    for k, series in allTs.iterrows():
        ctlDir, onePerDir = dirs(series, extn=extn, perDictTimeCache=perDictTimeCache)
        ctl = ctlDir / fileName
        CTLtas = iris.load_cube(str(ctl))
        onePer = onePerDir / fileName
        onePERtas = iris.load_cube(str(onePer))
        for c in [CTLtas, onePERtas]:
            PaperLib.addAuxCoord(c, realization=count)
        count += 1
        tas1percent = PaperLib.comp_fit(onePERtas, order=2, makeCube=True, year=179)
        tasCtl = PaperLib.comp_fit(CTLtas, order=2, makeCube=True, year=179)
        change = tas1percent - tasCtl
        delta.append(change)

        print("Read data for ", k)

    # convert cubelists to cube.
    delta = delta.merge_cube()
    delta.units = '1'
    delta.long_name = 'Temperature Change'

    mnDelta = delta.collapsed('realization', iris.analysis.MEAN)  # compute the std-dev
    stdDelta = delta.collapsed('realization', iris.analysis.STD_DEV)  # compute the std-dev
    cvDelta = stdDelta * 100. / mnDelta
    print("All data processed")
    return delta, mnDelta, cvDelta, stdDelta


LSmask = iris.load_cube(os.path.join(PaperLib.OptClimPath, 'data_files', 'HadCM3', 'qrparm.mask'), 'land_binary_mask')
msk = (LSmask.data == 0)

allTsData = PaperLib.read_data(sevenParam=True)
allTs14Data = PaperLib.read_data(fourteenParam=True)
allDFOLSdata = PaperLib.read_data(dfols=True)

deltaZM, control = compTrends(allTsData, LSmask=LSmask)
deltaZMDF14, controlDF14 = compTrends(allDFOLSdata, LSmask=LSmask, perDictTimeCache=PaperLib.DFOLSpath)
delta, mnDelta, cvDelta, stdDelta = comp_delta(allTsData)
deltaDF14, mnDeltaDF14, cvDeltaDF14, stdDeltaDF14 = comp_delta(allDFOLSdata, perDictTimeCache=PaperLib.DFOLSpath)

## now to plot maps and zonal-means
levSD = [0, 2, 5, 10, 20, 50]
levMN = np.array([0, 50, 75, 100, 125, 150, 200, 400])
cmapMN = mpl_cm.get_cmap('YlOrRd')
cmapMN = mpl_cm.get_cmap('viridis', len(levMN))
cmapMN = mpl_cm.get_cmap('RdBu_r')  # cool might be better
cmapSD = mpl_cm.get_cmap('viridis')
norm_lst = [mpl_colors.BoundaryNorm(lev, 256) for lev in [levMN, levSD]]
cmap_lst = [mpl_cm.get_cmap(name) for name in ['RdBu_r', 'viridis']]
proj = ccrs.PlateCarree(central_longitude=0.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

## plot maps of CV
subplot_kw = dict(projection=proj)
fig, axes = plt.subplots(nrows=2, ncols=1, num='SAT_var', clear=True,
                         figsize=[8, 9], subplot_kw=subplot_kw)
fig.subplots_adjust(hspace=0.22)
lab = PaperLib.plotLabel()
cmap = 'YlOrRd_r'
norm = mpl_colors.BoundaryNorm(levSD, 256)
for sd, ax, title in zip((cvDelta, cvDeltaDF14), axes, ('CE7', 'DF14')):
    cf_sd = iris.plot.pcolormesh(sd, axes=ax, cmap=cmap, norm=norm, vmin=levSD[0], vmax=levSD[-1])
    cs_sd = iris.plot.contour(sd, axes=ax, levels=levSD, colors='black', linewidths=2., vmin=levSD[0], vmax=levSD[-1])
    labels = cs_sd.clabel(fmt='%2d', inline=False, colors='white')  # fontweight removed

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_title(f'{title} $\Delta$Temp. CV(%)')

    coord_proj = ccrs.PlateCarree()
    ax.set_xticks(np.arange(-180, 181, 60), crs=coord_proj)
    ax.set_yticks(np.arange(-60, 61, 30), crs=coord_proj)
    ax.coastlines(color='black', linewidth=2)

    lab.plot(ax=ax)  # label axis

# add a colour bar.
cbar = fig.colorbar(cf_sd, ax=axes, ticks=levSD, orientation='horizontal',
                    extend='both', fraction=0.03, drawedges=True)
fig.tight_layout()
fig.show()
PaperLib.saveFig(fig)


## plot zm of changes in precip & temp.
fig, axes = plt.subplots(nrows=2, ncols=2, num='ZM_var', sharey=True, clear=True,
                         figsize=[8, 6],constrained_layout=True)
label = PaperLib.plotLabel()
slicer = slice(0, -1, 3)  # how often to plot symbols

linewidth = 2
marker = None # 'o'
colors = ('gray', 'green', 'red', 'blue')  # colors to plot things
title = ['Ocean', 'Land', 'Land Txx', 'Land Tnn']
# plot temperatures
for var, t, col in zip(('t15_ocean', 't15_land', 't15_max_land', 't15_min_land'), title, colors):
    fit, fit4, y = comp_multi_fit(var,  deltaZM)

    fitDF, fitDF_4, y = comp_multi_fit(var,  deltaZMDF14)

    for ax, data in zip(axes[0, :], [fit4, fitDF_4]):
        mn = data.mean(0)
        sd = data.std(0)
        cv = 100 * sd / mn  # CV
        ax.plot(cv, y, color=col, marker=marker, linewidth=linewidth, label=t)


# Plot ZM Precip

colors = ('gray', 'green', 'royalblue')  # colors to plot things
title = ['Ocean', 'Land', 'Land max']
for var, t, col in zip(['precip_ocean', 'precip_land', 'precip_max_land'], title, colors):
    lst = [f[var]  for k, f in deltaZM.items()]
    lstDF14 = [f[var] for k, f in deltaZMDF14.items()]
    data = iris.cube.CubeList(lst).merge_cube()  # merge the values into one variable
    dataDF14 = iris.cube.CubeList(lstDF14).merge_cube()
    y = data.coord('latitude').core_points()
    for ax, d in zip(axes[1, :], [data, dataDF14]):
        mean = d.collapsed('realization', iris.analysis.MEAN)
        sd = d.collapsed('realization', iris.analysis.STD_DEV)
        cv = sd*100/mean
        ax.plot(cv.data, y, color=col, marker=marker, linewidth=linewidth, label=t)


# add titles
values = [-90, -60, -30, 0, 30, 60, 90]
titles = [r"CE7 $\Delta$Temp.", r"DF14 $\Delta$Temp.",
          r" CE7 %Precip.  ", r"DF14 %Precip."]
sd_lims = [(0, 25), (0, 25), (0, 20), (0, 20)]

for ax, title,  loc, sd_lim in zip(axes.flatten(), titles,
                                          ('upper right', None, 'upper right', None), sd_lims):
    ax.set_yticks(values)
    ax.yaxis.set_major_formatter(FuncFormatter(PaperLib.latitudeLabel))
    ax.set_ylim(-90, 90)
    ax.set_xlim(sd_lim)
    ax.set_xlabel('%')
    ax.set_title(title+' CV (%)')
    ax.tick_params(axis='both', which='major', labelsize='small')
    if loc is not None:
        ax.legend(loc=loc, frameon=True, prop=dict(size='small'))

    label.plot(ax=ax, where=(-0.03, 1.075))

fig.show()
PaperLib.saveFig(fig)