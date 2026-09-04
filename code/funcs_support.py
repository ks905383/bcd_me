import xarray as xr
import xagg as xa
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
import os
import shutil
import glob
import warnings
import datetime

class NotUniqueFile(Exception):
    """ Exception for when one file needs to be loaded, but the search returned multiple files """
    pass

# Function to convert integer to Roman values
def printRoman(number):
    # from https://www.geeksforgeeks.org/python-program-to-convert-integer-to-roman/
    num = [1, 4, 5, 9, 10, 40, 50, 90,
        100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL",
        "L", "XC", "C", "CD", "D", "CM", "M"]
    i = 12

    output = ''
    while number:
        div = number // num[i]
        number %= num[i]
 
        while div:
            output = output + sym[i]
            div -= 1
        i -= 1

    return output

def get_params():
    ''' Get parameters 
    
    Outputs necessary general parameters. 
    
    Parameters:
    ----------------------
    (none)
    
    
    Returns:
    ----------------------
    dir_list : dict()
        a dictionary of directory names for file system 
        managing purposes: 
            - 'raw':   where raw climate files are stored, in 
                        subdirectories by model/product name
            - 'proc':  where processed climate files are stored,
                        in subdirectories by model/product name
            - 'aux':   where aux files (e.g. those that transcend
                        a single data product/model) are stored
    '''

    # Dir_list
    dir_list = pd.read_csv('dir_list.csv')
    dir_list = {d:dir_list.set_index('dir_name').loc[d,'dir_path'] for d in dir_list['dir_name']}


    # Return
    return dir_list

dir_list = get_params()

def get_filepaths(source_dir = 'raw',
                  mod = None,
                  dir_list = dir_list,
                  col_namer = {'(hadley$)|(CMIP[0-9]$)':'forcing_dataset',
                                 'PDO$':'pdo_state',
                                  'AMO$':'amo_state',}):
    ''' Get filepaths of climate data, split up by CMIP filename component
    
    
    Uses modified CMIP5/6 filename standards used by Kevin Schwarzwald's 
    filesystem - in other words, with the additional optional "suffix" 
    between the daterange and the filetype extension. 
    
    Returns
    ------------
    df : pd.DataFrame
        A dataframe containing information for all files in 
        `dir_list[source_dir]/mod/*.nc/.zarr`, with the full filepath in the
        column `path`, and filename components `varname`, `freq`, 
        `model`, `exp`, and optionally `run`, `grid`, `time`, 
        'gwl', 'proj_method'/'proj_target' for  method',`suffix`, in their own
        columns. `grid` may be Nones if files use CMIP5 conventions, 
        `suffix` may be Nones if no suffixes are found. Last time modified
        (from :py:meth:`os.path.getmtime()`) is listed as 'mtime'. 
        
        If `exp` has a match for the regex r"-", then additionally
        extra columns for each experiment name component will be 
        created, if possible, using the `col_namer` input.
    
    
    '''
    
    def id_fncomps(comps,col_namer=col_namer):
        # Make sure there are enough components 
        if len(comps)<6:
            # For now - but there has to be a better way to 
            # flag this
            slots = {'varname':None}
        else:
            try:
                # Figure out filetype
                filetype = re.split(r'\.',comps[-1])[-1]                    
                
                # Prepopulate set components
                slots = {s:comps[n] for n,s in zip(np.arange(0,4),['varname','freq','model','exp'])}
                proc_comps = list(np.arange(0,4))
                
                # Find slot for run, which will be of the form r##i## or "reanalysis" or "ALLRUNS"
                run_match = [re.search(r'(r[0-9]{1,3}([ipf][0-9]{1,2}){1,3})|(ALLRUNS)|(allruns)|(reanalysis)|(obs)|(stations)|(run[0-9]{1,3})|(RUNS[0-9]{1,3})|(ens[0-9]{1,3})',comp) 
                                   for comp in comps]
                if np.any(run_match):
                    slots['run'] = [k.group() for k in run_match
                                     if k is not None][0]
                    proc_comps.append(np.where(run_match)[0][0])
                
                # Get which slot is the timeframe (fx have "na" as timeframe)
                time_match = [re.search('([0-9]{4,8}'+r'-'+'[0-9]{4,8})|(^na($|'+r'.'+'))|(ALLPERIODS)',comp) 
                                   for comp in comps]
                if np.any(time_match):
                    slots['time'] = [k.group() for k in time_match
                                     if k is not None][0]
                    proc_comps.append(np.where(time_match)[0][0])
                
                # Determine whether there's a grid slot 
                # (assuming of the form 'gX(X)')
                grid_match = [re.search('^g[a-z0-9]{1,2}$',comp) for comp in comps]
                if np.any(grid_match): 
                    slots['grid'] = [k.group() for k in grid_match
                                     if k is not None][0]
                    proc_comps.append(np.where(grid_match)[0][0])

                # GWL slot
                gwl_match = [re.search('(^GWL)|(ALLGWLs)|(ALLGWLS)',comp) 
                           for comp in comps]
                if np.any(gwl_match):
                    slots['gwl'] = np.where(gwl_match)[0][0]
                    proc_comps.append(slots['gwl'])

                    if (comps[slots['gwl']] == 'ALLGWLs') or (comps[slots['gwl']] == 'ALLGWLS'):
                        slots['gwl'] = 'ALLGWLs'
                    else:
                        slots['gwl'] = re.sub(r'\-', '.', comps[slots['gwl']][3:None])

                # Projection slot
                proj_match = [re.search(r'^proj[a-zA-Z0-9]*\-base[a-zA-Z0-9\-]*',comp) 
                                   for comp in comps]
                if np.any(proj_match):
                    proj_info = [k.group() for k in proj_match
                                 if k is not None][0]
                    slots['proj_method'] = re.split(r'\-',re.split('proj',proj_info)[1])[0]
                    slots['proj_base'] = re.split('base',proj_info)[1]
                    proc_comps.append(np.where(proj_match)[0][0])

                # Downscaling slot
                dwscl_match = [re.search(r'^dwnscl[a-zA-Z0-9]*\-target[a-zA-z0-9\-]*',comp) 
                                   for comp in comps]
                if np.any(dwscl_match):
                    dwnscl_info = [k.group() for k in dwscl_match
                                 if k is not None][0]
                    slots['dwnscl_method'] = re.split(r'\-',re.split('dwnscl',dwnscl_info)[1])[0]
                    slots['dwnscl_target'] = re.split('target',dwnscl_info)[1]
                    proc_comps.append(np.where(dwscl_match)[0][0])

                # Seasstats slot
                seasstats_match = [re.search(r'seasstats',comp)
                                   for comp in comps]
                if np.any(seasstats_match):
                    seasstats_match = np.where(seasstats_match)[0][0]
                    slots['seasstats'] = comps[seasstats_match+1]
                    proc_comps.append(seasstats_match)
                    proc_comps.append(seasstats_match+1)


                # Get remaining, otherwise unlabeled components
                unlabeled_idxs = np.array([idx for idx in range(len(comps)) if idx not in np.array(proc_comps)])
                if len(unlabeled_idxs) > 0:
                    comps = np.array(comps)[unlabeled_idxs]
                else:
                    comps = []

                # If a remaining, unlabeled slot has the file ending in it, it's the "suffix"
                if np.any([re.search(r'[a-zA-z0-9]*\.'+filetype+r'$',comp) for comp in comps]):
                    slots['suffix'] = [k.group() for k in [re.search(r'[a-zA-z0-9]*\.'+filetype+r'$',comp) for comp in comps]
                                       if k is not None][0]
                    slots['suffix'] = re.split(r'\.',slots['suffix'])[0]
                    # Remove it from the list
                    comps = [comp for comp in comps if not re.search(r'[a-zA-z0-9]*\.'+filetype+r'$',comp)]

                # Add remaining unlabeled slots to just a generic catch-all 
                if len(comps)>0:
                    slots['unlabeled'] = str(comps)

                # If the experiment slot has multiple sub-experiments,
                # save them seperately using the column namer dict
                exp_comps = re.split(r'-',slots['exp'])
                if len(exp_comps)>1:
                    for exp_comp in exp_comps:
                        if np.any([re.search(k,exp_comp) for k in col_namer]):
                            match_type = [v for k,v in col_namer.items() if re.search(k,exp_comp)]
                            if len(match_type) > 1:
                                warnings.warn('More than one column match found for '+exp_comp+
                                              '. Check col_namer, no exp has been split.')
                            else:
                                slots[match_type[0]] = exp_comp

                # Add filetype as a slot
                slots['filetype'] = filetype
            except:
                # Assuming that if there's an error it's because the
                # file in question isn't in a standard respected form
                # For now - but there has to be a better way to 
                # flag this
                slots = {'varname':None}

        return slots

    #---------- Get list of files ----------
    if mod is None:
        # Get all mods
        mods = [re.split('/',mod)[-1] for mod in glob.glob(dir_list[source_dir]+'*')]
    else:
        mods = [mod]
        
    fns_all = [None]*len(mods)
    for mod,mod_idx in zip(mods,np.arange(0,len(mods))):
        # Get list of subdirectories (nc and zarr)
        fns = [*glob.glob(dir_list[source_dir]+mod+'/*.nc'),
               *glob.glob(dir_list[source_dir]+mod+'/*.zarr')]

        # Split up filename by components
        fn_comps = [re.split(r'_',re.split(r'/',fn)[-1]) for fn in fns]
        # Identify components, concatenate with path
        fns_all[mod_idx] = pd.DataFrame([id_fncomps(comps) for comps in fn_comps])
        fns_all[mod_idx] = pd.concat([fns_all[mod_idx],pd.DataFrame([{'path':fn} for fn in fns])],axis=1)

    # Concatentate into single df
    df = pd.concat(fns_all)

    # Add last time modified
    df.loc[:,'mtime'] = [datetime.datetime.fromtimestamp(os.path.getmtime(fn),
                            datetime.UTC)
                         for fn in df.path.values]

    # Sort by index
    df = df.reset_index().drop('index',axis=1).sort_index()

    #---------- Return ----------
    return df


# The next two are from https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)

    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5)
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5
        )

    return r

def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos
    from xarray import DataArray

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    xda = DataArray(
        area,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda

def area_mean(ds,assume_rectangular=True):
    """ Calculate area-weighted mean of all variables in a  dataset
    
    Mean over lat / lon, weighted by the relative size of each
    pixel, dependent on latitude. Only weights by latitude, does
    not take into account lat/lon bounds, if present. 
    
    Parameters
    ------------------
    ds : xr.Dataset
    
    Returns
    ------------------
    dsm : xr.Dataset
        The input dataset, `ds`, averaged.
    
    """
    
    if (ds.sizes['lat'] == 1) and (ds.sizes['lon'] == 1):
        # If just one pixel, return that one pixel
        ds = ds.isel(lat=0,lon=0).drop(['lat','lon'])
        
    elif (ds.sizes['lat'] == 1) and (assume_rectangular):
        # If only one lat row, but multiple long rows, 
        # just get the cartesian mean, if assuming rectangular
        # grids. 
        ds = ds.mean(('lat','lon'))
        
    else:
        # Calculate area in each pixel
        weights = area_grid(ds.lat,ds.lon)

        # Remove nans, to make weight sum have the right magnitude
        weights = weights.where(~np.isnan(ds))

        # Calculate mean
        ds = ((ds*weights).sum(('lat','lon'))/weights.sum(('lat','lon')))

    # Return 
    return ds


def utility_save(ds,output_fn,dir_list=None,raw_overwrite_flag=False,create_dir=True,
                 keep_chunk_encoding = True, save_kwargs = {},
                 zarr_mode = 'w-',
                 add_done_flag = True):
    ''' Save xarray dataset as netcdf or zarr file, with safeguards
    By default overwrites `output_fn`, *unless* `output_fn` is in the 
    raw data directory as defined by `dir_list`. Creates the implied
    directory if it does not already exist.

    Parameters
    ---------------
    ds : :py:class:`xr.Dataset`

    output_fn : :py:class:`str`

    dir_list : :py:class:`dict` or `None`, default `None`
        If None, then directories are grabbed using `get_params()`.
        Otherwise, put in a manual `dir_list` - which only requires
        `['raw']` as a field (to test whether anything in the 
        `raw` directory is being touched

    raw_overwrite_flag : :py:class:`bool`, default False
        If False, then if `output_fn` already exists in the `dir_list['raw']` 
        directory, an error is raised instead of overwriting the file

    create_dir : :py:class:`bool`, default True
        If True, then creates the implied directory (using 
        `os.path.dirname(output_fn)`) if it does not yet exist. 

    zarr_mode : :py:class:`str`, default "w-"
        If saving to zarr, the mode. From the zarr docs: 
            - "w-" : create, fail if exists
            - "w" : create, overwrite if exists
            - "a” : override all existing variables including dimension 
                    coordinates (create if does not exist)
            - "a-" : only append those variables that have append_dim 
                     (which can be set using save_kwargs)
            - "r+" : modify existing array values only (raise an error 
                     if any metadata or shapes would change)
        Note also if zarr_mode = 'w' and the file path exists, then 
        if overwriting, the zarr store will be deleted first (following
        the rules of `raw_overwrite_flag`). 

    add_done_flag : :py:class:`bool`, default True
        If True and saving as zarr, then adds an empty '.done' file in 
        the zarr store after saving is complete
    '''
    
    if dir_list is None:
        dir_list = get_params()

    if re.search(r'\.zarr$',os.path.basename(output_fn)):
        filetype = 'zarr'
    elif re.search(r'\.nc$',os.path.basename(output_fn)):    
        filetype = 'nc'

    if not os.path.exists(os.path.dirname(output_fn)):
        os.mkdir(os.path.dirname(output_fn))
        print(os.path.dirname(output_fn)+' created!')

    if (os.path.exists(output_fn) and 
        ((filetype == 'nc') or 
         ((filetype == 'zarr') and (zarr_mode == "w")))):
        if not raw_overwrite_flag:
            if re.search(r'^'+dir_list['raw'],output_fn):
                raise FileExistsError('Trying to overwrite a file in the "raw" data directory '+dir_list['raw']+'. '+
                                      'If this is on purpose, set `raw_overwrite_flag=True`.\n'+
                                      'Attempted output filename: '+output_fn)

        if filetype == 'zarr':
            shutil.rmtree(output_fn)
        else:
            os.remove(output_fn)
        print(output_fn+' removed to allow overwrite!')

    if not keep_chunk_encoding:
        from funcs_aux import _remove_chunk_encoding
        ds = _remove_chunk_encoding(ds)

    if filetype == 'zarr':
        # If saving zarr
        ds.to_zarr(output_fn,**save_kwargs)

        # Add empty 'done' file to zarr store after 
        # completed saving if desired
        if add_done_flag:
            open(output_fn+'/.done', 'w').close()
    elif filetype == 'nc':
        # If saving netcdf
        ds.to_netcdf(output_fn,**save_kwargs)
    print(output_fn+' saved!')


def utility_print(output_fn,formats=['pdf','png']):
    if 'pdf' in formats:
        plt.savefig(output_fn+'.pdf')
        print(output_fn+'.pdf saved!')

    if 'png' in formats:
        plt.savefig(output_fn+'.png',dpi=300)
        print(output_fn+'.png saved!')

    if 'svg' in formats:
        plt.savefig(output_fn+'.svg')
        print(output_fn+'.svg saved!')



# This whole business is because np.nanargmax (and its 
# derivatives, including ds.argmax in xr) can't deal if 
# the whole column of months is NaNs. So in that case, we 
# just keep the value NaN, and run nanargmax on the columns
# that do have values (land pixels in GPCC, for example)

def nan_argmax_xr(x,val=0,dim='month'):
    """ Get the index of each maximum month in the 'month'
    dimension of an arbitrary dataarray with dimensions
    'month' and others. This spits out NaN for any 
    row of months that's entirely NaNs, and therefore 
    provides a workaround for np.argmax() and 
    np.nanargmax(), which both fail in this situation.
    Furthermore, it automatically stacks/unstacks for 
    the calculation, so the input can have an arbitrary
    number of dimensions. 
    """
    
    # Stack to have [__ x month]
    input_dims = list(x.dims)
    
    if dim not in input_dims:
        raise LookupError("no '"+dim+"' dimension found.")
    
    input_dims.remove(dim)
    
    if len(input_dims)>1:
        x = x.stack(alld=(tuple(input_dims)))
        unstack = True
    else: 
        unstack = False
    
    if x.ndim>1:
        # Pre-build np.nan
        out_vals = np.zeros((np.shape(x)[np.argmax([key!=dim for key in x.dims])]))*np.nan
    
        #out_vals[~np.isnan(x[0,:])] = x[:,~np.isnan(x.values[0,:])].argsort(0).isel({dim:-1-val})
        #out_vals = xr.DataArray(out_vals,dims=x.dims[1],coords={x.dims[1]:x[x.dims[1]]})
        if not np.all(np.isnan(x).all(dim)): #else keep it nan
            out_vals[~np.isnan(x).all(dim)] = x[:,~np.isnan(x).all(dim)].argmax(dim)
        out_vals = xr.DataArray(out_vals,dims=x.dims[1],coords={x.dims[1]:x[x.dims[1]]})
    else:
        if np.all(np.isnan(x)):
            out_vals = np.nan
        else:
            out_vals = x.argmax(dim)
        # Pretty sure this just needs to be one value... to be consistent
        #out_vals[~np.isnan(x)] = x[~np.isnan(x.values)].argsort(0).isel({dim:-1-val})
        #out_vals = xr.DataArray(out_vals,dims=x.dims[0],coords={x.dims[0]:x[x.dims[0]]})
        out_vals = xr.DataArray(out_vals)
    
    if unstack:
        out_vals = out_vals.unstack()
    
    return out_vals

def nan_argmin_xr(x,val=0,dim='month'):
    """ Get the index of each maximum month in the 'month'
    dimension of an arbitrary dataarray with dimensions
    'month' and others. This spits out NaN for any 
    row of months that's entirely NaNs, and therefore 
    provides a workaround for np.argmax() and 
    np.nanargmax(), which both fail in this situation.
    Furthermore, it automatically stacks/unstacks for 
    the calculation, so the input can have an arbitrary
    number of dimensions. 
    """
    
    # Stack to have [__ x month]
    input_dims = list(x.dims)
    
    if dim not in input_dims:
        raise LookupError("no '"+dim+"' dimension found.")
    
    input_dims.remove(dim)
    
    if len(input_dims)>1:
        x = x.stack(alld=(tuple(input_dims)))
        unstack = True
    else: 
        unstack = False
    
    if x.ndim>1:
        # Pre-build np.nan
        out_vals = np.zeros((np.shape(x)[np.argmax([key!=dim for key in x.dims])]))*np.nan
        
        #out_vals[~np.isnan(x[0,:])] = x[:,~np.isnan(x.values[0,:])].argsort(0).isel({dim:-1-val})
        #out_vals = xr.DataArray(out_vals,dims=x.dims[1],coords={x.dims[1]:x[x.dims[1]]})
        if not np.all(np.isnan(x).all(dim)): #else keep it nan
            out_vals[~np.isnan(x).all(dim)] = x[:,~np.isnan(x).all(dim)].argmin(dim)
        out_vals = xr.DataArray(out_vals,dims=x.dims[1],coords={x.dims[1]:x[x.dims[1]]})
    else:
        if np.all(np.isnan(x)):
            out_vals = np.nan
        else:
            out_vals = x.argmin(dim)
        # Pretty sure this just needs to be one value... to be consistent
        #out_vals[~np.isnan(x)] = x[~np.isnan(x.values)].argsort(0).isel({dim:-1-val})
        #out_vals = xr.DataArray(out_vals,dims=x.dims[0],coords={x.dims[0]:x[x.dims[0]]})
        out_vals = xr.DataArray(out_vals)
    
    if unstack:
        out_vals = out_vals.unstack()
    
    return out_vals
