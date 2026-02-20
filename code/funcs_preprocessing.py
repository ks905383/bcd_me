import xarray as xr
import xagg as xa
import xesmf as xe
import pandas as pd
import cftime
import os
import re
import numpy as np
import fsspec
import dask
import zarr
from fsspec.implementations import local

from funcs_support import get_params,get_filepaths
from funcs_aux import utility_save,_verify_file_rows,_find_main_variable

def reshape_for_rolling(ds,wwidth,drop_leaps=True,
                        wrap_years = False):
    ''' Reshape ds from `time` to wrapped-around `dayofyear x year`

    Parameters
    --------------
    ds : :py:class:`xr.Dataset` or :py:class:`xr.DataArray` 
        an xarray object (can be Dataset or DataArray)

    wwidth : int
        how wide the subsequent rolling window will be;
        the xarray object will be wrapped around such 
        that it is [1:365, 1:wwidth]

    drop_leaps : bool, by default True
        if True, then 366th days of the year are dropped

    wrap_years : bool, by default False
        if False, then wrapping around happens within a year,
        i.e., some days will be repeated. Use this when the 
        temporal ordering *within* a year doesn't matter, as 
        this could result in more efficient processing,
        depending on chunking.

        if True, then wrapping happens across years; i.e., 
        a given year will start with days from a previous year
        and end with days in a subsequent year. Note that this
        results in nans in the first/last year of the sample. 
        Within a year, days will thus not be repeated. 
        

    Returns
    --------------
    ds_out : :py:class:`xr.Dataset` or :py:class:`xr.DataArray`
        the input xarray object, reshaped to dayofyear x 
        year, and wrapped to [1:365, 1:wwidth] days of year

    '''
    ds = ds.drop_duplicates('time')
    
    # Reshape to dayofyear x year 
    ds = ds.assign_coords({'year':ds.time.dt.year,
                'dayofyear':ds.time.dt.dayofyear})
    
    ds = ds.set_index(time=['dayofyear','year']).unstack()

    if drop_leaps:
        # (remember that .sel is inclusive) 
        ds = ds.sel(dayofyear=slice(1,365))

    if wrap_years:
        if not drop_leaps:
            raise KeyError('`wrap_years` currently only compatible with leap-removed datasets')

        # Wrap around such that each year looks like this: 
        # [(d364,y-1),(d365,y-1),(d1,y),...(d365,y),(d1,y+1),(d2,y+1)], etc.
        # Note that in this case, the previous year's days are listed as daysofyear
        # counting backwards from 1, (i.e. *1-indexed*, *not* *0-indexed* - day-of-year
        # 0 means Dec 31 of the previous year)
        ds = xr.concat([(ds.isel(dayofyear=slice(-wwidth+1,None)).
                          assign_coords({'year':ds.year.values+1}).
                          assign_coords({'dayofyear':(ds.isel(dayofyear=slice(-wwidth+1,None)).dayofyear.values - 
                                                      (365 + (not drop_leaps)))})),
                         ds.sel(dayofyear=slice(1,365+(not drop_leaps))),
                         ds.sel(dayofyear=slice(1,wwidth-1)).
                            assign_coords({'year':ds.year.values-1}).
                            assign_coords({'dayofyear':np.arange(365+(not drop_leaps)+1,
                                                                 365+(not drop_leaps)+wwidth)}),
                       ],dim='dayofyear')
        # Remove thus created extra years at start and end
        ds = ds.isel(year=slice(1,-1))
    else:
        # Wrap around such that it's days of year [1:365, 1:wwidth]
        ds_doubled = ds.isel(dayofyear=slice(0,wwidth-1)).copy()
        ds_doubled['dayofyear'] = np.arange(ds.dayofyear.max()+1,ds.dayofyear.max()+wwidth)
    
        ds = xr.concat([ds,ds_doubled],dim='dayofyear')
    return ds

def from_cache(fn_in,fn_out,fs,save_kwargs = {'zarr_format':2}):
    ''' Decorator for caching step 

    Loads the zarr dataset at `fn_in`, applies a function to it,
    saves it at `fn_out`, and removes the original `fn_in`. 

    Parameters
    --------------
    fn_in : str
        Filename of a `zarr` store to load and then delete after processing

    fn_out : str
        Path of a `zarr` store or `netcdf` file to save processed data at

    fs : fsspec-compatible filesystem

    '''
    def decorator(func):
        def wrapper(*args,**kwargs):
            # Load cached data
            ds = xr.open_zarr(fn_in)
    
            # Process
            ds = func(ds,*args,**kwargs)
    
            # Save modified data
            utility_save(ds.drop_encoding(),fn_out,save_kwargs = save_kwargs)
            
            # Remove original cache
            fs.rm(fn_in,recursive=True)
        return wrapper
    return decorator

def rechunk_ds(ds,chunking):
    ''' Rechunks the array `ds`

    '''
    ds = ds.chunk(chunking).unify_chunks()

    return ds

def regrid_ds(ds,ref_grid=None,rgrd=None,regrid_method = 'bilinear'):
    ''' Regrid to a reference grid

    Must specify either `ref_grid` grid (in which rgrd = xe.Regridder(ds,ref_grid,regrid_method) 
    is called first) or `rgrd`, an `xe.Regridder()` instance.

    '''

    if ((ref_grid is None) and (rgrd is None)) or ((ref_grid is not None) and (rgrd is not None)):
        raise ValueError('One (and only one) of `ref_grid` and `rgrds` must be specified')

    if ref_grid is not None:
        # Periodic if reaches to within one lon step of 180/-180
        if (ds.lon.max() >= (180-np.max(ds.lon.diff('lon')))) and (ds.lon.min() <= (-180+np.max(ds.lon.diff('lon')))):
            periodic = True
        else:
            periodic = False
        rgrd = xe.Regridder(ds,ref_grid,regrid_method,ignore_degenerate = False,periodic = periodic)
    ds = rgrd(ds)

    # Some regridders in xesmf return all 0s when something is out of bounds
    # instead of nan. If all in a latitude band or all in a longitude band are 0, 
    # then assume those are out of bounds pixels and change to nans
    keyvar = _find_main_variable(ds) # Find most likely process variable as the one with the highest dimensionality
    lat_band_nanflag = (ds[keyvar]==0).all([d for d in ds[keyvar].sizes if d != 'lat'])
    lon_band_nanflag = (ds[keyvar]==0).all([d for d in ds[keyvar].sizes if d != 'lon'])
    ds = ds.where(~lat_band_nanflag & ~lon_band_nanflag)
    #ds = ds.isel(lat = np.where(~lat_band_nanflag)[0],lon = np.where(~lon_band_nanflag)[0])
    return ds

def reshape_rechunk_ds(ds,wwidth,chunking = None):
    ''' Wrapper for `reshape_for_rolling` that also rechunks 
    output to have one chunk across the two time variables

    Parameters
    -------------
    ds : xr.Dataset

    wwidth : int
        Reshaping window width to pipe into :py:meth:`reshape_for_rolling()`

    chunking : dict or None
        If None, the dataset is rechunked to {'dayofyear':-1,'year':-1,'lat':20,'lon':20}
        after processing through :py:meth:`reshape_for_rolling()`
    '''
    # Reshape
    ds = reshape_for_rolling(ds,wwidth)

    # Rechunk to one temporal chunk
    if chunking is None:
        ds = ds.chunk({'dayofyear':-1,'year':-1,'lat':20,'lon':20})
    else:
        ds = ds.chunk(chunking)

    return ds

def preprocess_models(file_rows,
                      gwl_info,
                      params_var,
                      params_proc,
                      params_subset,
                      ref_grid,
                      fs,
                      override_filestem = None,
                      extra_filename_slots = [],
                      regrid_method = 'bilinear',
                      chunk_sizes = {'geo':20,'time':500},
                     remove_leaps = True):
    ''' Preprocess model data
    Load data, regrid to `ref_grid`, rechunk so there's 
    only one chunk in time.
    '''
    #-------- Load and preprocess --------
    # This process inspired by https://github.com/carbonplan/cmip6-downscaling/blob/44a140c4e1bbdb8394600594eee6fa90c6677b48/flows/methods/bcsd/flow.py#L25
    # and how they handle rechunking / regridding 
    # (literally temp files for every step, and it 
    # does indeed speed things up massively)

    # Get directory structure
    dir_list = get_params()

    # Verify provided filepaths
    file_rows,mod,run,exp = _verify_file_rows(file_rows)

    # Get filenames for temporary files 
    if override_filestem is None:
        if type(extra_filename_slots) == str:
            extra_filename_slots = [extra_filename_slots]
        
        filestem = (dir_list['proc']+mod+'/'+
                    '_'.join([params_var['var'], params_var['freq'], mod, exp, 
                              run,*extra_filename_slots]))
    else:
        filestem = override_filestem
    fns_out = {'rchk': filestem + '_CACHE-RECHUNK',
               'rgrd': filestem + '_CACHE-REGRID',
               'rgrd_rchk': filestem + '_CACHE-RECHUNK2',
               'reshp': filestem + '_CACHE-RESHAPE'}
    
    for fn in fns_out:
        if 'suffix' in params_var:
            fns_out[fn] = fns_out[fn]+'_'+params_var['suffix']
        fns_out[fn] = fns_out[fn]+'.zarr'

    if not fs.exists(fns_out['reshp']):
        # Load (and concat in time) historical + future data
        # for a given var/freq/model/exp/run
        # (with the restrictions suggested by https://github.com/pydata/xarray/issues/8778 )
        # use_cftime = True, since some of the future 
        # scenarios have times that go beyond the datetime
        # limits
        try:
            ds = xr.open_mfdataset(file_rows.path,use_cftime=True,chunks={'lat':-1,'lon':-1,'time':chunk_sizes['time']}, 
                               data_vars='minimal', coords='minimal', join='exact', compat='override',
                               concat_dim = 'time',combine='nested')
        except Exception:
            # If could not decode... 
            ds = []
            for row in file_rows.iterrows():
                ds_tmp = xr.open_dataset(row[1]['path'],decode_times = False)
                if ds_tmp.sizes['time'] % 365 == 0:
                    # If multiple of 365, assume 365-day year
                    ds_tmp['time'] = [cftime.DatetimeNoLeap(int(row[1]['time'][0:4]),1,1) + pd.Timedelta(days = n)
                                         for n in range(ds_tmp.sizes['time'])]
                else:
                    ds_tmp['time'] = [cftime.DatetimeProlepticGregorian(int(row[1]['time'][0:4]),1,1) + 
                                      pd.Timedelta(days = n)
                                         for n in range(ds_tmp.sizes['time'])]
                ds.append(ds_tmp.drop_vars(['lat_bounds','lon_bounds','lat_bnds','lon_bnds',
                                            'time_bounds','time_bnds'],
                                           errors='ignore'))

            ds = xr.concat(ds,dim='time')
            # This would still be in the old convention, drop it
            ds = ds.drop_vars(['time_bnds'],errors='ignore')

        # Standardize variable names/dimension structure
        lon_attrs = ds.lon.attrs
        if len(lon_attrs) == 0:
            lon_attrs = {'axis':'X',
                         'bounds':'lon_bnds',
                         'long_name':'Longitude',
                         'standard_name':'longitude',
                         'units':'degrees_east'}
        with xr.set_options(keep_attrs=True):
            ds = xa.fix_ds(ds)
        # Hack to make sure longitude attributes stick around / cf knows what's what
        # (since xa.fix_ds can remove longitude attributes when standardizing to 
        # -180:180 degrees)
        ds.lon.attrs = lon_attrs
        
        # Sort by time (some cmip files are unsorted)
        ds = ds.sortby('time')
        
        # Subset to desired time
        ds = ds.sel(time=slice(str(int(gwl_info.start_year.min())),
                               str(int(gwl_info.end_year.max()))))
        
        # Remove time_bounds and other aux variables, which are not 
        # necessary and can cause saving issues
        ds = ds.drop_vars(['time_bounds','time_bnds','height'],errors='ignore')
        
        # Add GWL dimension
        if len(gwl_info) == 1:
            ds = ds.expand_dims({'gwl':[gwl_info.warming_level]})
        
        #-------- Rechunk to one chunk for space --------
        # Rechunk to 1 chunk geographically
        ds = ds.chunk({'lat':-1,'lon':-1,'time':chunk_sizes['time']}).unify_chunks()
        
        # Save rechunked as temporary zarr
        utility_save(ds.drop_encoding(),fns_out['rchk'],save_kwargs = {'zarr_format':3})
        
        #-------- Regrid to 1x1 --------
        # Use decorator to manage loading / removing temporary file 
        regrid_fromcache = from_cache(fns_out['rchk'],fns_out['rgrd'],fs)(regrid_ds)
        # Regrid to reference grid 
        regrid_fromcache(ref_grid=ref_grid,regrid_method=regrid_method)

        # Subset to desired space (done after regridding to not cause
        # regridding issues
        if params_subset is not None:
            ds = xr.open_zarr(fns_out['rgrd'])
            ds = ds.sel(**params_subset)
            ds.to_zarr(fns_out['rgrd'],mode='r+')
        
        #-------- Rechunk to one chunk for time --------
        # Use decorator to manage loading / removing temporary file 
        rechunk_fromcache = from_cache(fns_out['rgrd'],fns_out['rgrd_rchk'],fs)(rechunk_ds)
        # Rechunk
        rechunk_fromcache({'time':-1,'lat':chunk_sizes['geo'],'lon':chunk_sizes['geo']})
        
        #-------- Reshape to DOY x time --------
        # Use decorator to manage loading / removing temporary file 
        reshape_fromcache = from_cache(fns_out['rgrd_rchk'],fns_out['reshp'],fs)(reshape_rechunk_ds)
        # Reshape to doy x time and rechunk
        reshape_fromcache(params_proc['wwidth'],{'dayofyear':-1,'year':-1,'lat':chunk_sizes['geo'],'lon':chunk_sizes['geo']})
    else:
        print(fns_out['reshp']+' already exists, skipped!')
