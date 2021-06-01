"""
Little module to provide common tools to read data. Eventually gets integrated into PaperLib
"""
import functools
import PaperLib
import os
import iris
import numpy as np


gmConstraint = iris.Constraint(site_number=3.0) # constraint for g-m
@functools.lru_cache(maxsize=5012)
def compDeltaFlux(ctl,force,clear=False):
    """
    Compute difference in netflux between forced and ctl case.plt.figure(
    :param ctl:
    :param force:
    :param clear: if True (default is False) compute clear sky flux.
    :return: difference in net flux between forced and ctl simultions
    """

    ctlNF = PaperLib.comp_net(ctl,clear=clear)
    forceNF = PaperLib.comp_net(force,clear=clear)
    deltaNF = PaperLib.comp_delta(forceNF, ctlNF)
    return deltaNF

@functools.lru_cache(maxsize=5012)
def compCtl(dir,file, scale=1.0, order=2,verbose=False):
    """
    Compute the nth order fit
    :param dir: ctl directory
    :param file: filename to use
    :param scale (optional default 1): How to scale result by
    :param order (optional default is 2): Order of fit -- see PaperLib.comp_fit
    :param verbose (optional defualt False): produce some output.
    :return: fitted values

    """
    if dir is None:
        print("dir not specified")
        return None

    if verbose:
        print("compCtl: dir: %s file:%s"%(dir,file))
    try:
        if file is 'NetFlux': # special code for net flux
            data = PaperLib.comp_net(dir)
        elif file is 'ClearNetFlux':
            data = PaperLib.comp_net(dir,clear=True)
        elif file is 'CRF':
            data = PaperLib.comp_crf(dir)
        else:
            path= os.path.join(dir, file)
            data = PaperLib.read_pp(path,verbose=True)
            if data.shape[1] == 3: data = data.extract(gmConstraint)
            if data is None:
                raise Exception("woops")
        result = PaperLib.comp_fit(data, order=order)*scale
        result = result.squeeze()
    except IOError:
        print("IOerror when reading %s %s"%(dir,file))
        result = [None]*order


    return result

#@functools.lru_cache(maxsize=5012)
def compDelta(ctl, force, file,verbose=False):
    """
    Compute the difference between forced and ctl values
    :param dir: directory
    :param file: filename to use
    :param (optional): scale -- how much to scale data by.
    :return: data

    """
    if ctl is None or force is None:
        return None
    try:
        if file is 'NetFlux': # special code for net flux
            delta = compDeltaFlux(ctl, force)
        elif file is 'ClrNetFlux': # special code for net flux
            delta = compDeltaFlux(ctl, force,clear=True)
        elif file is 'CRF': #special code for CRF
            ctlData = PaperLib.comp_crf(ctl)
            forceData  = PaperLib.comp_crf(force)
            delta = PaperLib.comp_delta(forceData, ctlData)
        else:
            ctlData = PaperLib.read_pp(os.path.join(ctl, file),verbose=verbose)
            forceData = PaperLib.read_pp(os.path.join(force, file),verbose=verbose)
            #import pdb; pdb.set_trace()
            if verbose:
                print("Ctl",ctlData,"Force",forceData)
            delta = PaperLib.comp_delta(forceData, ctlData)
            if delta.shape[1] == 3: delta = delta.extract(gmConstraint)
            if delta is None:
                raise Exception("woops")
    except IOError:
        delta = None


    return delta

def compEquil(ctl,force,file, scale=1, constraint=None):
    """
    Compute equilibrium value. Does it by regressing netflux on variable and then computing value of variable when
      netflux is zero by solving the polynomail. Does it in this round about way for consistency with ECS computation.
      Differences is because there is noise in both the surface temperature and netflux.
    :param ctl: dir path to control simulation
    :param force: dir path to forced simulation
    :param file: name of time_Cache file
    :param simName: name of simulation.
    :return: estimated equilibrium value
    """
    delta = compDelta(ctl,force,file)

    if delta is None:
        return None # no data so return None

    deltaNF = compDeltaFlux(ctl,force)
    if deltaNF is None:
        return None

    deltaNF = np.squeeze(deltaNF.data)
    if constraint is not None:
        delta = delta.extract(constraint)
    delta = np.squeeze(delta.data)*scale

    if delta.ndim == 1: # one-d -- make it two d.
        delta = np.reshape(delta,(-1,1))
    npts = delta.shape[1]
    eqValue = np.zeros(npts)
    for k in range(0, npts):
        reg = np.polyfit(delta[:,k], deltaNF, 1)
        roots = np.roots(reg)
        eqValue[k] = roots[np.isreal(roots)][0]  # find first **real** root.


    try:
        eqValue = np.asscalar(eqValue) # if a single value want to return as a scalar value.
    except ValueError: # not a single value so just squeeze it.
        eqValue = eqValue.squeeze()

    return eqValue