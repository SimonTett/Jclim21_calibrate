"""
Compute variables for taylor diagram.
These are (for each variable) the std dev of the simulated - obs field scaled by the obs std dev.
And the correlation between the simulated and obs fields.
In both cases the area-average mean is removed and everything is area weighted.
"""
import numpy as np
import pandas as pd

verbose = True
test = False
skipRead = False
import xarray
import PaperLib
import collections

xarray.set_options(keep_attrs=True)


def taylor(sim_ds, obs_ds):
    """
    Compute area mean for each var in a dataset
    :param ds: dataset.
    :return: named tuple with scaled std_dev
    """

    wt = PaperLib.area_wt(obs_ds)

    # delta = delta - mn_delta

    sd_obs = obs_ds.weighted(wt).std()

    sim_anom = sim_ds - sim_ds.weighted(wt).mean()
    sd = sim_anom.weighted(wt).std()
    norm_sd = sd / sd_obs
    obs_amom = obs_ds - obs_ds.weighted(wt).mean()
    corr = (sim_anom * obs_amom).weighted(wt).mean()
    corr = corr / (sim_anom.weighted(wt).std() * obs_amom.weighted(wt).std())

    result_dict = dict()
    for var in norm_sd.data_vars:
        result_dict[var + '_sd'] = float(norm_sd[var].data)
        result_dict[var + '_corr'] = float(corr[var].data)

    result = pd.Series(result_dict)  # convert to a pandas series.
    return result


def msk_sfc(ds, ls_msk, variables=None):
    """
    Apply a mask to surface variables in a ds
    :param ds: dataset
    :param ls_msk: ls_msk 1 = land; 0 = sea
    :param variables: list of variables to mask. Default is air_temperature & precipitation_flux
    :return: masked ds
    """

    if variables is None:
        variables = ['air_temperature_tas', 'precipitation_flux']

    result = ds.copy()
    for var in variables:
        try:
            result[var] = xarray.where(ls_msk == 1, ds[var], np.NAN)
        except KeyError:
            pass

    return result


def model_name(ds, default=None):
    """
    Work out model name for dataset. Uses the global attrs "model_id"
    :param ds: dataset
    :param default what to call the model if we don't find the name. If nto given then Unknown is used
    :return: name of the model
    """
    model_name = 'Unknown'
    if default is not None:
        model_name = default

    for name in ['source_id', 'model_id']:  # CMIP5 & CMIP6 use different names for the realization
        try:
            model_name = ds.attrs[name]
            break  # found something so exit the loop.
        except KeyError:  # failure so pass and continue in the loop of
            pass

    return model_name


def add_realization(ds):
    # add realization to the dataset
    realization = None
    for name in ['realization_index', 'realization',
                 'variant_id']:  # CMIP5 & CMIP6 use different names for the realization
        try:
            realization = int(ds.attrs[name])
            break  # found something so exit the loop.
        except KeyError:  # failure so pass and continue in loop of
            pass
    if realization is None:
        print('------------------------------------------')
        print(f"For model {model_name(ds)} no realization found. Setting to 0")
        print(ds.attrs)
        print("------------------------------------")
        realization = 0

    ds2 = ds.copy()
    for v in ds.data_vars:
        ds2[v] = ds[v].expand_dims(realization=[realization])

    return ds2


def read_cmip(files, ls_msk, default=None, verbose=False):
    """
    Read and process data from CMIP archives
    :param files: files to read
    :param verbose: If True bprint out models as read them in. Default is False
    :return: dict indexed by model name. Each value dataset of the ensemble average for each model.
    """
    CMIP = dict()

    for file in files:
        ds = xarray.open_dataset(file).squeeze(drop=True)
        ds = add_realization(ds)

        # check have all the variables we want...
        name = model_name(ds, default=default)
        ds = ds.expand_dims(model=[name])
        lst = CMIP.get(name, [])
        lst.append(ds)
        CMIP[name] = lst
        if verbose: print(name)

    data_vars = set()
    # now merge the data and then mean over realization. That gives for each case the ensemble mean.
    # Also work out if are missing variables..
    for key in CMIP.keys():
        if verbose:
            print("combining ", key)

        combined_ds = xarray.combine_nested(CMIP[key], None, compat='override', combine_attrs='drop_conflicts')
        combined_ds = combined_ds.mean('realization')
        # rename variables to "std" names
        # work out rename dict
        rename = dict()
        for var in combined_ds.data_vars:
            try:
                new_name = combined_ds[var].attrs['standard_name']
                if var == 'ta':
                    new_name = 'air_temperature_500'
                elif var == 'tas':
                    new_name = 'air_temperature_tas'
                rename[var] = new_name
            except KeyError:
                pass

        combined_ds = combined_ds.rename(rename)
        bad_std_rename = dict(air_pressure_at_sea_level='air_pressure_at_mean_sea_level')
        # some models have bad standard names. This will fix em!
        try:
            combined_ds = combined_ds.rename(bad_std_rename)
        except ValueError:
            pass

        CMIP[key] = msk_sfc(combined_ds, ls_msk)





    return CMIP


taylor_diag = PaperLib.OptClimPath / 'grl17_coupled/taylor_diag'
# step 0 get in the l/s mask and modify it
ls_mask = xarray.load_dataset(taylor_diag / 'HadAM3_N48_land.nc')['lsm'].squeeze(drop=True)
ls_mask = xarray.where(ls_mask.latitude >= -60, ls_mask, 0)  # data sth of 60 to be masked
# step 1 read in the obs data

obs_path = taylor_diag / 'obsData/N48'
obs_ds = xarray.open_mfdataset(obs_path.glob('*2000*.nc'))
# select out the 500 hPa data for RH & Temp
for v in ['r', 't']:
    obs_ds[v] = obs_ds[v].sel(level=500)

# convert precip to kg/m^2/second from mm/month
obs_ds['pre'] /= (365 / 12 * 24 * 60 * 60)
obs_ds['tmp'] += 273.16  # convert to K
# and apply l/s mask

for v in ['tmp', 'pre']:
    obs_ds[v] = xarray.where(ls_mask == 1, obs_ds[v], np.NAN)

rename = dict(r='relative_humidity', msl='air_pressure_at_mean_sea_level', t='air_temperature_500',
              tmp='air_temperature_tas',
              toa_lw_all_mon='toa_outgoing_longwave_flux', toa_sw_all_mon='toa_outgoing_shortwave_flux',
              pre='precipitation_flux')

obs_ds = obs_ds[rename.keys()].rename(rename)  # extract what we want and rename it.
obs_ds = obs_ds.squeeze().load()

CMIP5_path = taylor_diag / 'CMIP5/N48'
CMIP6_path = taylor_diag / 'CMIP6/N48'
# going to read in (and avg data if nesc for each model)
CMIP5_files = CMIP5_path.glob('*.nc')
CMIP6_files = CMIP6_path.glob('*.nc')
if test:
    CMIP5_files = [f for f in CMIP5_files if f.name.startswith('Model1')]
    CMIP6_files = [f for f in CMIP6_files if f.name.startswith('Model2')]

if not skipRead:
    CMIP5 = read_cmip(CMIP5_files, ls_mask, default='Unknown CMIP5', verbose=verbose)
    CMIP5['CMIP5-MM'] = xarray.combine_nested([ds.drop_vars('average_DT', errors='ignore') for ds in CMIP5.values()],'model').mean('model')
    CMIP6 = read_cmip(CMIP6_files, ls_mask, default='Unknown CMIP6', verbose=verbose)
    # dealing with missing variables for MM mean...
    mm=dict()
    for var in list(CMIP6.values())[0].data_vars:
        lst = [ds[var] for ds in CMIP6.values() if ds.get(var) is not None]
        mm[var] = xarray.concat(lst,'model',coords='minimal',compat='override').mean('model')

    CMIP6['CMIP6-MM'] = xarray.Dataset(mm)

## now to get in the HadAM3 simulations
# first get in the simulation info
runInfo = pd.read_excel('OptClim_lookup.xlsx', index_col=0)  # read in meta data on runs
worked = runInfo.Status == 'Succeeded'  # ones that worked
runInfo = runInfo[worked]
# DFOLS first
DFOLS = dict()
mn_file = '_2000_2005_mn.nc'
rename = dict(air_pressure_at_sea_level='air_pressure_at_mean_sea_level',
              air_temperature='air_temperature_tas',
              air_temperature_2='air_temperature_500',
              relative_humidity_2='relative_humidity')


def comp_mn(row):
    p1 = taylor_diag / row['Atmosphere Run#1']
    p2 = taylor_diag / row['Atmosphere Run#2']
    p1 = p1 / (p1.name + mn_file)
    p2 = p2 / (p2.name + mn_file)
    mn_data = ((xarray.load_dataset(p1) + xarray.load_dataset(p2)) / 2).rename(rename)
    # and fix the surface temperature & precip
    return msk_sfc(mn_data, ls_mask)


for s, row in runInfo.query('Ensemble=="DF14"').iterrows():
    DFOLS[s] = comp_mn(row)
DFOLS['DFOLS-MM'] = xarray.concat(DFOLS.values(),'model').mean('model')


# and do the standard case
standard = comp_mn(runInfo.loc['Standard', :])

## now do compute the values we want.
# CMIP5 first
series = []
for k, v in CMIP5.items():
    sim = v.drop_vars('average_DT', errors='ignore')
    series.append(taylor(sim, obs_ds).rename(k))
CMIP5_df = pd.DataFrame(series)

# then CMIP6
series = []
for k, v in CMIP6.items():
    sim = v.drop_vars('average_DT', errors='ignore')
    series.append(taylor(v, obs_ds).rename(k))
CMIP6_df = pd.DataFrame(series)

# then DFOLS from HadAM3
series = []
for k, v in DFOLS.items():
    sim = v.drop_vars('average_DT', errors='ignore')
    series.append(taylor(v, obs_ds).rename(k))
# wrap in the standard run.

DFOLS_df = pd.DataFrame(series)
standard_series = taylor(standard, obs_ds).rename('Standard')

# write things out.

CMIP6_df.to_csv(PaperLib.dataPath / 'CMIP6_taylor.csv')
CMIP5_df.to_csv(PaperLib.dataPath / 'CMIP5_taylor.csv')
DFOLS_df.to_csv(PaperLib.dataPath / 'DFOLS_taylor.csv')
standard_series.to_csv(PaperLib.dataPath / 'Standard_taylor.csv')
