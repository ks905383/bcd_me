import os
import re
import numpy as np
import pooch
import fsspec
import dask
import zarr
import xarray as xr
from fsspec.implementations import local
import warnings
import shutil

from funcs_support import get_params,get_filepaths
dir_list = get_params()

modlists = {'qdm':['ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR',
                     'BCC-ESM1','CAMS-CSM1-0','CESM2-WACCM',
                     'CMCC-CM2-SR5','CMCC-ESM2','CNRM-CM6-1',
                     'CNRM-ESM2-1','CanESM5','EC-Earth3',
                     'EC-Earth3-Veg','FGOALS-g3','IITM-ESM',
                     'IPSL-CM6A-LR','MIROC-ES2L','MIROC6',
                     'MPI-ESM-1-2-HAM','MPI-ESM1-2-LR',
                     'MRI-ESM2-0','NESM3'],
             'qdm-byyr':['CESM2-WACCM','CNRM-CM6-1','CNRM-ESM2-1',
                         'CanESM5','EC-Earth3','EC-Earth3-Veg','FGOALS-g3',
                         'IPSL-CM6A-LR','MIROC6','MPI-ESM1-2-LR',
                         'MRI-ESM2-0','NESM3'],
             'qdm-qplad':['ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR',
                     'BCC-ESM1','CAMS-CSM1-0','CESM2-WACCM',
                     'CMCC-CM2-SR5','CMCC-ESM2','CNRM-CM6-1',
                     'CNRM-ESM2-1','CanESM5','EC-Earth3',
                     'EC-Earth3-Veg','FGOALS-g3','IITM-ESM',
                     'IPSL-CM6A-LR','MIROC-ES2L','MIROC6',
                     'MPI-ESM-1-2-HAM','MPI-ESM1-2-LR',
                     'MRI-ESM2-0','NESM3']}




#flagged_members = xr.open_dataset(f'{dir_list['aux']}problem_member_flags.nc')
def filter_problems(dss,flagged_members,flag_var = None,return_flags = False,verbose=False):
    ''' Filter for flagged problem ensemble mmebers

    Given `problem_members_all` and input data `dss`, returns 
    `dss` with members flagged for technical or quality control
    issues set to all nans. 

    Parameters
    -----------------
    dss : xr.Dataset
        a BCD-ME store or subset or concatenated set of BCD-ME stores.
        Must have the `idv` dimension with model, experiment, run info.

    flagged_members : None (default) or xr.Datatset or str
        The dataset of flagged "problem" BCD-ME members. If a string, 
        assumes string is the filename of this file and loads it. 

        If None (default), then the most recent set of problem_member 
        details will be accessed and cached (or reloaded from cache) 
        from the BCD-ME GitHub, with the BCD-ME "set" (i.e., qdm, qdm-qplad`
        etc...) inferred from `dss` characteristics.  

    flag_var : str, by default None
        If not None, `tas` or `tasmax`, corresponding to the flags from
        flagged_members to use to filter the variables in `dss`. Use if
        the `dss` variables are not "raw" BCD-ME variables (i.e., if they
        are just `tas` and/or `tasmax`, or the statistics `tasmaxbinF`, etc., 
        they are automatically matched to the flags)

    return_flags : bool, by default False
        If True, returns matched flags as well

    verbose : bool, by default False
        If True, returns information about matched variables and flags

    Returns
    -----------------
    dss : xr.Dataset
        the input dss, with the flagged runs turned to nans

    flags : xr.Dataset
        if `return_flags==True`, the flags used to filter the dss
    '''

    if flagged_members is None:
        pm_source_url = 'https://github.com/ks905383/bcd_me/blob/main/bcd_me_helper_pack.zip?raw=True'
        def fetch_problem_members(url = pm_source_url):
            files = pooch.retrieve(
                url=url,
                known_hash=None,             
                path=pooch.os_cache('problem_members'),
                processor=pooch.Unzip(),
            )
            return files
        
        files = fetch_problem_members()
        pm_fns = [fn for fn in files if 'problem_member' in fn]
        
        qdm_dims = ['dayofyear']
        qplad_dims = ['binF','binC','degree']
        if np.any([k in dss.gwl.values for k in [2010,2040,2070]]):
            flagged_members = [fn for fn in pm_fns if re.search(r'qdm\-byyr',fn)][0]
            print('Assuming target data is from the `qdm-byyr` BCD-ME set...')
        elif np.any([k in dss.sizes for k in qdm_dims]) or np.all([k not in dss.sizes for k in qplad_dims]):
            flagged_members = [fn for fn in pm_fns if re.search(r'qdm\.',fn)][0]
            print('Assuming target data is from the `qdm` BCD-ME set...')
        else:
            flagged_members = [fn for fn in pm_fns if re.search(r'qdm\-qplad',fn)][0]
            print('Assuming target data is from the `qdm-qplad` BCD-ME set...')


    if type(flagged_members) == str:
        flagged_members = xr.open_dataset(flagged_members)

    dss_index = dss.set_index(idv = ['model','experiment','run']).idv
    flagged_members = flagged_members.set_index(idv = ['model','experiment','run'])

    # Get the problem members that are present in the input data
    join_idvs = [idv for idv in flagged_members.idfv.values if idv in list(dss_index.values)]
    
    flags = []
    for var in flagged_members:
        # Make an initial problem = False boolean map for all members
        # in the input data
        idv_tmp = xr.zeros_like(dss_index).astype(bool)
        # Make the boolean true for the problem members
        idv_tmp.loc[join_idvs] = flagged_members[var].loc[join_idvs]
        # Add to structure
        idv_tmp = idv_tmp.to_dataset(name=var)
        flags.append(idv_tmp)
    flags = xr.merge(flags)
    # Switch from True = problem to True = keep
    flags = ~flags

    if flag_var is None:
        matched_vars = []
        if set([k for k in flags]) == set([k for k in dss]):
            dss = dss.where(flags)
            matched_vars = [k for k in dss]
        else:
            varconnects = {'tasmaxbinF':'tasmax',
                           'tasmaxbinC':'tasmax',
                           'tasmaxsumpoly':'tasmax',
                           'tasbinF':'tas',
                           'tasbinC':'tas',
                           'tassumpoly':'tas'}
            if np.all([v in varconnects for v in dss]):
                for v in dss:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        # Deleting level of MultiIndex warning. Not sure what it's warning about
                        # since the multiindex is unchanged here
                        flags[v] = flags[varconnects[v]]
                    matched_vars.append(v)
                dss = dss.where(flags)
            else:
                raise KeyError(('Cannot match flag variables (tas, tasmax) to input data variables (either tas, tasmax or '+
                                'tas*binF, *binC, *sumpoly). Please manually choose a `flag_var`, which will set the variable '+
                                'whose problem flags are used to filter the input data.'))
    else:
        if flag_var not in ['tas','tasmax']:
            raise KeyError(f'`flag_var` must be one of "tas" or "tasmax", those are the variables with a problem filter attached.')
        dss = dss.where(flags[flag_var])

    if verbose:
        print((f'{len(join_idvs)} out of {flagged_members.sizes['idv']} total problem ensemble members identified in '+
                   f'the input dataset.')) 
        if flag_var is None:
            print((f'Filtering matched variables {', '.join(matched_vars)}.'))
        else:
            print((f'Filtering variables {', '.join([k for k in dss])} using flag from variable {flag_var}.'))

    dss = dss.reset_index('idv')
    if return_flags:
        return dss,flags
    else:
        return dss