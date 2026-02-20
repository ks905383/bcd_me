import xarray as xr
from numba import njit
import numpy as np
import pandas as pd
import dask
import zarr

from funcs_support import get_params,get_filepaths
from funcs_aux import (utility_save, _verify_file_rows, _create_filenames, 
                       _restore_doys, extract_gwl, repeat_ds)


@njit
def sliding_ranks(data, wwidth):
    ''' Numbaized function to calculate ranks across a sliding window

    CURRENTLY ASSUMES INPUT IN FORM OF dayofyear x year, WITH SLIDING 
    WINDOW ACROSS dayofyear

    '''
    n_dayofyear, n_year = data.shape
    ranks = np.full_like(data, np.nan)

    half_window = wwidth // 2
    #for i in range(n_dayofyear):
    for doy_idx in range(half_window,n_dayofyear-half_window):
        # Define the sliding window along the dayofyear dimension
        start = doy_idx - half_window
        end = doy_idx + half_window + 1

        # Gather data for the sliding window across all years
        window = data[start:end, :].ravel()
        
        for year_idx in range(n_year):
            current_value = data[doy_idx, year_idx]
            rank = np.sum(window < current_value) + 1  # Rank starts at 1
            ranks[doy_idx, year_idx] = rank

    return ranks/(wwidth*n_year)


def calc_pctof_models(file_rows,gwl_info,gwl_info_rea,params_var,params_proc,
                       fs,chunk_sizes = {'geo':20}):
    ''' Calculate the quantile of each value within a rolling window by dayofyear

    '''

    # Verify provided filepaths
    file_rows, mod, run, exp = _verify_file_rows(file_rows)

    # Get filenames for temporary files 
    fns_out = _create_filenames(gwl_info_rea,params_var,params_proc,file_rows=file_rows)

    if not fs.exists(fns_out['pctof']):
        # Load
        ds = xr.open_zarr(fns_out['reshp_mod'])

        # Get GWLs
        ds = xr.concat([extract_gwl(ds, gwl_row).drop_vars(['time_bnds'],errors='ignore')
                       for gwl_row in gwl_info.iterrows()], dim='gwl').chunk({'year':-1,'dayofyear':-1})

        # Apply the Numba function for percentile of across sliding 
        # windows along daysofyear
        ds_pctof = xr.apply_ufunc(sliding_ranks,
                                  ds,
                                  input_core_dims=[['dayofyear', 'year']],
                                  output_core_dims=[['dayofyear', 'year']],
                                  kwargs={'wwidth': params_proc['wwidth']},
                                  vectorize=True,
                                  dask='parallelized', output_dtypes=[float]
                                 )

        # Subset to the 365 days actually processed
        ds_pctof = _restore_doys(ds_pctof, params_proc)

        # Sometimes the chunks get messed up
        ds_pctof = ds_pctof.chunk({'lat':chunk_sizes['geo'],'lon':chunk_sizes['geo']})

        # Export to zarr
        utility_save(ds_pctof.drop_encoding(), fns_out['pctof'],save_kwargs = {'zarr_format':3},
                     zarr_mode = 'w')
    else:
        print(fns_out['pctof']+' exists, skipped!')


def calc_pctof_bcmodels(file_rows,gwl_info_rea,params_var,params_proc,
                       fs,bc_type = 'qdm',subset_params = None):
    ''' Calculate the quantile of each value within a rolling window by dayofyear
    Similar to `calc_pctof_models`, but with the slightly different file format of 
    the bias-corrected models, including repeating to the 0.25 deg grid
    
    '''
    # Verify provided filepaths
    file_rows, mod, run, exp = _verify_file_rows(file_rows)

    # Get filenames for temporary files 
    fns_out = _create_filenames(gwl_info_rea,params_var,params_proc,file_rows=file_rows)

    if not fs.exists(fns_out['pctof_tiled']):
        # 1. Load bias-corrected model data
        ds = xr.open_zarr(fns_out['biascrct_'+bc_type])
        if subset_params is not None:
            ds = ds.sel(**subset_params)
        
        # Make double precision, nothing more precise is needed
        # (and this might speed up things a ton from 64-bit default)
        ds[params_var['var']] = ds[params_var['var']].astype(np.float32)
        
        # 2. Reshape model data for rolling
        # Wrap around such that it's days of year [1:365, 1:wwidth]
        ds_doubled = ds.isel(dayofyear=slice(0,params_proc['wwidth']-1)).copy()
        ds_doubled['dayofyear'] = np.arange(ds.dayofyear.max()+1,ds.dayofyear.max()+params_proc['wwidth'])
        
        ds = xr.concat([ds,ds_doubled],dim='dayofyear')
        
        # Make sure there's still only one temporal chunk
        ds = ds.chunk({'dayofyear':365+params_proc['wwidth']-1})

        # 3. Get rolling ranks for each model data value
        # 3 minutes on 8 cores of Glade for 5 GWLs
        ds_pctof = xr.apply_ufunc(sliding_ranks,
                                  ds,
                                  input_core_dims=[['dayofyear', 'year']],
                                  output_core_dims=[['dayofyear', 'year']],
                                  kwargs={'wwidth': params_proc['wwidth']},
                                  vectorize=True,
                                  dask='parallelized', output_dtypes=[float]
                                 )
        
        # 4. Tile model pct of to match downscaled reanalysis
        ds_pctof = repeat_ds(ds_pctof,4)
        ds_pctof = ds_pctof.chunk({'lat':20,'lon':20})

        # Subset to the 365 days actually processed
        ds_pctof = _restore_doys(ds_pctof, params_proc)

        # Export to zarr
        utility_save(ds_pctof.drop_encoding(), fns_out['pctof_tiled'],save_kwargs = {'zarr_format':3})
    else:
        print(fns_out['pctof']+' exists, skipped!')


@njit
def calc_quantiles(data, quantiles):
    '''Numbaized quantiles calculator
    Produces identical results to `np.quantile()`, by
    linearly interpolating between values if a given 
    quantile doesn't have an integer index match 
    (but it is numbaized / dask-compatible, which 
    `np.quantile()` is not) 

    Parameters
    --------------
    data : array
        Array of which to calculate quantiles

    quantiles : array
        Quantiles to calculate from data

    Returns 
    --------------
    results : len(quantiles) array
        The quantiles `quantiles` of `data`
    '''
    sorted_data = np.sort(data)
    n = len(sorted_data)
    results = np.empty(len(quantiles))
    
    for i, q in enumerate(quantiles):
        # Get the index of the quantile
        idx = q * (n - 1)
        # Basically round the index up and
        # down; if they're the same, then 
        # the quantile matches an index exactly
        low = int(np.floor(idx))
        high = int(np.ceil(idx))
        
        if low == high:
            # If the quantile matches an index exactly
            # (i.e., quantile = 0.5 for 9 values), then
            # just return that index's value
            results[i] = sorted_data[low]
        else:
            # If the quantile does not match an index exactly
            # (i.e., quantile = 0.5 for 10 values), then
            # return a relevant linear interpolation 
            fraction = idx - low
            results[i] = (1 - fraction) * sorted_data[low] + fraction * sorted_data[high]
    
    return results


@njit
def calc_quantile_diffs_gwl(data, wwidth, quantiles):
    '''Compute quantile differences between GWLs.
    Assumes first index in GWL dimension is the reference GWL
    
    [Quantiles of data ] - [Quantiles of data[0,:,:]] 

    Parameters
    --------------
    data : gwl x dayofyear x year array
        Data to calculate sliding window 

    wwidth : window width
        Width of sliding window in dayofyear dimension

    quantiles : array 
        Quantiles of window onto data to calculate

    Returns
    --------------
    quantile_diffs : gwl x dayofyear x quantile array
        Rolling quantile differences between all gwls
        and the gwl in the first index
    '''
    n_gwl, n_doy, n_year = data.shape
    n_quantiles = len(quantiles)
    quantile_diffs = np.full((n_gwl, n_doy, n_quantiles), np.nan)
    half_window = wwidth // 2

    for doy_idx in range(half_window, n_doy - half_window):
        # Define the sliding window
        start = doy_idx - half_window
        end = doy_idx + half_window + 1
        # To avoid "reshape only supports contiguous array" issue
        window = data[:, start:end, :].copy().reshape(n_gwl, wwidth * n_year)

        # Compute quantiles for each GWL
        reference_quantiles = None
        for gwl_idx in range(n_gwl):
            valid_data = window[gwl_idx][~np.isnan(window[gwl_idx])]
            if len(valid_data) > 0:
                q_vals = calc_quantiles(valid_data, quantiles)
                if gwl_idx == 0:
                    reference_quantiles = q_vals
                quantile_diffs[gwl_idx, doy_idx, :] = q_vals - reference_quantiles

    return quantile_diffs


@njit
def calc_quantile_diffs_array(data_mod, data_ref, wwidth, quantiles):
    '''Compute quantile differences between two arrays.

    [Quantiles of data_ref] - [quantiles of data_mod]
    
    Parameters
    --------------
    data_mod : dayofyear x year array
        Data to calculate sliding window 

    data_ref : dayofyear x year array

    wwidth : window width
        Width of sliding window in dayofyear dimension

    quantiles : array
        Quantiles of window onto data to calculate

    Returns
    --------------
    quantile_diffs : dayofyear x quantile array
        Rolling quantile differences between mod and ref
    '''
    
    if not np.allclose(data_mod.shape,data_ref.shape):
        raise Exception('Make sure `data_mod` and `data_ref` have the same shapes. '+
                         'Current shapes: \n`data_mod`: '+', '.join([str(k) for k in data_mod.shape])+'\n'+
                         '`data_ref`: '+', '.join([str(k) for k in data_ref.shape])+'.')
        
    n_doy, n_year = data_mod.shape
    n_quantiles = len(quantiles)
    quantile_diffs = np.full((n_doy, n_quantiles), np.nan)
    half_window = wwidth // 2

    for doy_idx in range(half_window, n_doy - half_window):
        # Define the sliding window
        start = doy_idx - half_window
        end = doy_idx + half_window + 1
        # To avoid "reshape only supports contiguous array" issue
        window_mod = data_mod[start:end, :].flatten()
        window_ref = data_ref[start:end, :].flatten()

        # Compute quantiles for each array
        nan_flags = np.isnan(window_mod) | np.isnan(window_ref)
        window_mod = window_mod[~nan_flags]
        window_ref = window_ref[~nan_flags]
        if len(window_mod) > 0:
            q_mod = calc_quantiles(window_mod, quantiles)
            q_ref = calc_quantiles(window_ref, quantiles)
            # Get quantile differences, ref dataset - model dataset
            quantile_diffs[doy_idx, :] = q_ref - q_mod

    return quantile_diffs


@njit
def calc_rolling_quantiles(data, wwidth, quantiles):
    '''Compute quantile differences between two arrays.

    Rolling quantiles of `data`
    
    Parameters
    --------------
    data : dayofyear x year array
        Data to calculate sliding window 

    wwidth : window width
        Width of sliding window in dayofyear dimension

    quantiles : array
        Quantiles of window onto data to calculate

    Returns
    --------------
    quantiles : dayofyear x quantile array
        Rolling quantiles
    '''
        
    n_doy, n_year = data.shape
    n_quantiles = len(quantiles)
    qs_out = np.full((n_doy, n_quantiles), np.nan)
    half_window = wwidth // 2

    for doy_idx in range(half_window, n_doy - half_window):
        # Define the sliding window
        start = doy_idx - half_window
        end = doy_idx + half_window + 1
        # To avoid "reshape only supports contiguous array" issue
        window = data[start:end, :].flatten()

        # Compute quantiles for each array
        nan_flags = np.isnan(window)
        window = window[~nan_flags]
        if len(window) > 0:
            qs_out[doy_idx, :] = calc_quantiles(window, quantiles)

    return qs_out


def calc_quantilediffs_models(file_rows,params_var,gwl_info,gwl_info_rea,params_proc,
                   fs,chunk_sizes = {'geo':20}):
    ''' Calculate quantile differences between model and reanalysis

    '''
    # Verify provided filepaths
    file_rows,mod,run,exp = _verify_file_rows(file_rows)

    # Get filenames of temporary / intermediate files
    fns_out = _create_filenames(gwl_info_rea,params_var,params_proc,file_rows=file_rows)

    if not fs.exists(fns_out['qdiff']):
        # Load model
        ds = xr.open_zarr(fns_out['reshp_mod'])
        
        # Get GWLs
        ds = extract_gwl(ds,[row for row in gwl_info.iterrows()][0]).isel(gwl=0,drop=True)
        # XX TO-DO: make it so that if only one row is inputted into extract_gwl, that it 
        # doesn't do the GWL dimension (or at least optionally not?)

        # Load reanalysis
        ds_rea = xr.open_zarr(fns_out['reshp_rea'])
        # Change year to generic counter (e.g. 0:19)
        ds_rea['year'] = ds['year'].values

        # Homogenize grids (if the original input `ds` doesn't fully cover
        # the `ref_grid`, it gets trimmed. Make sure the `ds_rea` also
        # gets trimmed to the same in that case
        ds_rea = ds_rea.sel(lat=ds.lat,lon=ds.lon)
        
        # Get quantiles to fit
        qs = np.arange(1/params_proc['nqs']/2,
                       (1-1/params_proc['nqs']/2+(1/(params_proc['nqs']*10))),
                       1/params_proc['nqs'])
        
        # Apply the Numba function for sliding quantile differences
        dqs = xr.apply_ufunc(calc_quantile_diffs_array,
                             ds[params_var['var']],
                             ds_rea[params_var['var']],
                             input_core_dims = [['dayofyear','year'],['dayofyear','year']],
                             output_core_dims = [['dayofyear','quantile']],
                             kwargs = {'wwidth':params_proc['wwidth'],'quantiles':qs},
                             dask='parallelized',
                             vectorize=True,
                             dask_gufunc_kwargs={'output_sizes':{'quantile':params_proc['nqs']}},
                             output_dtypes = [float]
                             )
        
        # Subset to the 365 days actually processed 
        dqs = _restore_doys(dqs,params_proc)

        # Turn back to dataset
        dqs = dqs.to_dataset(name='q'+params_var['var']+'diff')

        # Sometimes the chunks get messed up 
        dqs = dqs.chunk({'lat':chunk_sizes['geo'],'lon':chunk_sizes['geo']})
        
        # Export to zarr
        utility_save(dqs,fns_out['qdiff'],save_kwargs = {'zarr_format':2})
    else:
        print(fns_out['qdiff']+' exists, skipped!')


def calc_equantile_diffs(data_source,data_dest, wwidth, quantiles=None):
    ''' Function to calculate differences empirical quantiles (sort) between 
    datasets across a sliding window

    CURRENTLY ASSUMES INPUT IN FORM OF dayofyear x year, WITH SLIDING 
    WINDOW ACROSS dayofyear

    '''
    if not np.allclose(data_source.shape,data_dest.shape):
        raise Exception('Source and destination data do not have the same shape '+
                        '('+str(data_source.shape)+' vs. '+str(data_dest.shape)+', respectively)')
    
    n_dayofyear, n_year = data_source.shape
    #diffs = np.full((n_dayofyear,(wwidth*n_year)), np.nan)
    if quantiles is not None:
        diffs = np.full_like(quantiles, np.nan)
    else:
        diffs = np.full((n_dayofyear, n_year*wwidth),np.nan)

    # Return nan array if all nan, calculate quantile differences otherwise
    if not np.all(np.isnan(data_source)):
        half_window = wwidth // 2
        for doy_idx in range(half_window,n_dayofyear-half_window):
            # Define the sliding window along the dayofyear dimension
            start = doy_idx - half_window
            end = doy_idx + half_window + 1
    
            # Gather data for the sliding window across all years
            window_source = data_source[start:end, :].ravel()
            window_dest = data_dest[start:end, :].ravel()
    
            # Sort
            sorted_window_source = np.sort(window_source)
            sorted_window_dest = np.sort(window_dest)
    
            if quantiles is not None:
                sorted_window_source = sorted_window_source[quantiles[doy_idx,:]]
                sorted_window_dest = sorted_window_dest[quantiles[doy_idx,:]]
    
            diffs[doy_idx,:] = sorted_window_dest - sorted_window_source

    return diffs


@njit
def get_quantile_diffs(dqs,quantiles):
    '''Query quantile differences (numbaized)
    `quantiles` is the quantile whose change you want to query
        (i.e., the `pctof` output)
    `dqs` is the change in quantiles calculated

    Assuming that quantiles is a dayofyear x year array
    and dqs is a dayofyear x quantile array

    Output is dayofyear x year

    Parameters
    --------------
    dqs : dayofyear x quantile array
        Quantile differences calculaed by :py:meth:`quantilediffs_models()`

    quantiles : dayofyear x year array
        Quantiles to query in the quantile differences `dqs`

    Returns 
    --------------
    dqs_out : dayofyear x year array
        The quantile difference from `dqs` for each quantile in `quantiles`
    '''
    n_doy,n_year = quantiles.shape
    n_doy_qs,n_quantile = dqs.shape
    if n_doy != n_doy_qs:
        raise Exception('`dqs` and `quantiles` do not have the same number of daysofyear (1st dimension): '+
                        str(n_doy_qs)+' and '+str(n_doy)+', respectively.')

    # Create blank output array
    dqs_out = np.full_like(quantiles,np.nan)

    # Process by dayofyear
    for doy_idx in range(n_doy):
        for year, q in enumerate(quantiles[doy_idx,:]):
            if (q < 0) or (q > 1):
                raise Exception('quantiles must be between 0 and 1; a quantile of value '+str(q)+' was encountered.')
            # Get the index of the quantile (assuming equally
            # spaced quantiles were calculated, e.g [0.05, 0.15, 
            # ...0.95] for idxs [0, 1, ... 9])
            idx = (q-(1/(2*n_quantile))) * n_quantile
            # Basically round the index up and
            # down; if they're the same, then 
            # the quantile matches an index exactly
            low = int(np.floor(idx))
            high = int(np.ceil(idx))

            if low == high:
                # If the quantile matches an index exactly
                # (i.e., quantile = 0.5 for 9 values), then
                # just return that index's value
                dqs_out[doy_idx,year] = dqs[doy_idx,low]
            else:
                # If the quantile does not match an index exactly
                # (i.e., quantile = 0.5 for 10 values), then
                # return a relevant linear interpolation 
                fraction = idx - low
                if low == -1:
                    # If below lowest quantile, interpolate linearly
                    # using difference between lowest and second-to-
                    # lowest quantiles instead
                    dqs_out[doy_idx,year] = (1 - idx) * dqs[doy_idx,low+1] + idx * dqs[doy_idx,high+1]
                elif high == n_quantile:
                    # If above highest quantile, interpolate linearly
                    # using difference between highest and second-to-
                    # highest quantiles instead
                    dqs_out[doy_idx,year] = (1 + idx - (n_quantile-1)) * dqs[doy_idx,low] + (-idx + (n_quantile-1)) * dqs[doy_idx,low-1]
                else:
                    # Otherwise just do a simple linear interpolation
                    # between the two adjacent calculated quantile diffs
                    dqs_out[doy_idx,year] = (1 - fraction) * dqs[doy_idx,low] + fraction * dqs[doy_idx,high]
    
    return dqs_out


@njit
def bias_correct_qm(dsf,dqs,quantiles):
    ''' Bias-correct using QM/quantile-differences

    Applies quantile differences (between base warming level 
    model and reference data) from `dqs` to the model data in 
    `dsf`, using each of the values in `dsf`'s percentiles 
    stored in `quantiles`. 

    Parameters
    ---------------
    dsf : the future data to bias-correct

    dqs : the quantile 'biases'/differences

    quantiles : the quantiles of each element of future data
    
    '''
    dsf_out = np.full_like(dsf,np.nan)

    n_gwl,n_doy,n_year = dsf.shape

    for gwl_idx in range(n_gwl):
        # Query bias correction factor for each quantile of 
        # the future data
        dqs_out = get_quantile_diffs(dqs,quantiles[gwl_idx,:,:])

        # Add factor to future data
        # (dqs is calculated as ref_data - mod_data, so adding
        # becomes [mod_data + (ref_data - mod_data)])
        dsf_out[gwl_idx,:,:] = dsf[gwl_idx,:,:] + dqs_out

    return dsf_out

@njit
def bias_correct_qdm(qsi,dsf,dqs,quantiles):
    ''' Bias-correct using QDM/quantile-differences
    XX TO-DO, rename `get_quantile_diffs` to `query_quantiles` or something like that

    Applies quantile differences (between base warming level 
    model and reference data) from `dqs` to the base data in 
    , using each of the values in `dsf`'s percentiles 
    stored in `quantiles`. 

    Parameters
    ---------------
    qsi : the base quantiles for the delta mapping
    
    dsf : the future data to bias-correct 
        (currently only used for shape...)

    dqs : the quantile 'biases'/differences in the model 

    quantiles : the quantiles of each element of future data
    
    '''
    dsf_out = np.full_like(dsf,np.nan)

    n_doy,n_year = dsf.shape

    # Query change in quantile for each quantile of 
    # the future data
    dqs_out = get_quantile_diffs(dqs,quantiles[:,:])
    
    # Query base data at each quantile of the future 
    # data
    qsi_out = get_quantile_diffs(qsi,quantiles[:,:])
    
    # Add factor to base data data
    dsf_out[:,:] = qsi_out + dqs_out

    return dsf_out


def wrapper_bias_correct(file_rows,gwl_info,gwl_info_rea,params_var,params_proc,
                         fs):
    ''' Bias-correct using pre-calculated quantile differences

    To be run after :py:meth:`quantilediffs_models()`

    '''
    # Get filenames of temporary / intermediate files
    fns_out = _create_filenames(gwl_info_rea,params_var,params_proc,file_rows=file_rows)

    if not fs.exists(fns_out['biascrct']):

        #--------- Setup ---------
        # Load percentiles of model data
        ds_pctof = xr.open_zarr(fns_out['pctof'])
    
        # Load changes in quantiles
        dqs = xr.open_zarr(fns_out['qdiff'])
    
        # Load reshaped / preprocessed model data 
        dsf = xr.open_zarr(fns_out['reshp_mod'])
        # Reshape it by GWL
        dsf = xr.concat([extract_gwl(dsf,gwl_row).drop_vars(['time_bnds'],errors='ignore') 
                         for gwl_row in gwl_info.iterrows()],
                    dim='gwl').chunk({'year':-1}) # For some reason this rechunks year into smaller chunks otherwise
        # Remove extra DOYs
        dsf = _restore_doys(dsf,params_proc)
    
        #--------- Process ---------
        # Apply bias-correction
        dsf_out = xr.apply_ufunc(bias_correct_qdm,
                                 dsf[params_var['var']].chunk({'gwl':-1}),
                                 dqs['q'+params_var['var']+'diff'],
                                 ds_pctof[params_var['var']].chunk({'gwl':-1}),
                                 input_core_dims = [['gwl','dayofyear','year'],
                                                    ['dayofyear','quantile'],
                                                    ['gwl','dayofyear','year']],
                                 output_core_dims = [['gwl','dayofyear','year']],
                                 dask='parallelized',
                                vectorize=True,
                                output_dtypes = [float])
    
    
        #--------- Save ---------
        # Turn to dataset
        dsf_out = dsf_out.to_dataset(name = params_var['var'])

        # Get save attributes
        attrs = {'SOURCE':'wrapper_bias_correct()',
                 'DESCRIPTION':'QDM bias correction by GWL, using '+params_proc['mod_rea']+' as a reference base.',
                 'REF_DATA':params_proc['mod_rea']}
    
        # Save
        utility_save(dsf_out,fns_out['biascrct'],attrs=attrs,save_kwargs = {'zarr_format':2})
    else:
        print(fns_out['biascrct']+' already exists, skipped!')



# 7. Now get damage function parameters
@njit
def numba_expsums(data,degs,C_from_K = False):
    ''' Numbaized sum of exponents function
    '''

    sums = np.full(len(degs),np.nan)

    if C_from_K:
        data = data-273.15

    if not np.all(np.isnan(data)):
        for i,deg in enumerate(degs):
            sums[i] = sum(data**deg)

    return sums


def dmgf_params_carleton(da,degs,dim='dayofyear'):
    ''' Calculate necessary parameters for Carleton et al. Damage Function

    Get T, T^2, T^3, T^4 for each location

    Compare to the equation in B.2.4 (Supp Material), which is 
       T_it = [ Sum_day ( Sum_pix ( w_pix (T_pix,day)^k ) ) for k in 1, 2, 3, 4]

    Since summations are commutative, I change this to 
       T_it = [ Sum_pix ( w_pix * Sum _day (T_pix,day)^k ) for k in 1, 2, 3, 4]

    Parameters
    ---------------
    da : :py:meth:`xr.DataArray`

    degs : list / array
        Exponential degrees (the ks in T**k), e.g. [1,2,3,4]
        
    dim : :py:meth:`str`, by default 'dayofyear'
        The dimension over which to conduct the histogram 

    Returns
    ---------------
    params : :py:meth:`xr.DataArray` 
        With the daysofyear dimension replaced with n_params (in this case 4)
    
    '''

    params = xr.apply_ufunc(
        numba_expsums,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[['param']],
        kwargs={'degs': degs},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs = {'output_sizes':{'param':len(degs)}}
    )
    params['param'] = ['T'+str(k) for k in degs]
    return params


@njit
def numba_histogram(data, bins, edges = 'inf'):
    ''' Numbaized histogram function

    '''
    if edges == 'inf':
        bins = np.array([-np.inf,*bins,np.inf])
        
    out_len = len(bins) - 1

    # int16 more than sufficient for max value of 365
    hist = np.zeros(out_len, dtype=np.int16)
    if not np.all(np.isnan(data)):
        for value in data:
            for i in range(out_len):
                if bins[i] <= value < bins[i + 1]:
                    hist[i] += 1
                    break
    return hist

def dmgf_params_bins(da, bins, dim='dayofyear'):
    ''' Generate histogram bins

    Parameters
    ---------------
    da : :py:meth:`xr.DataArray`

    bins : iterable
        The bins of the histogram

    dim : :py:meth:`str`, by default 'dayofyear'
        The dimension over which to conduct the histogram

    Returns
    ---------------
    bins_out : :py:meth:`xr.DataArray` 
        With the `dim` dimension replaced with n_bins

    '''
    bins_out =  xr.apply_ufunc(
            numba_histogram,
            da,
            input_core_dims=[[dim]],
            output_core_dims=[['bin']],
            kwargs={'bins': bins},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int16],
            dask_gufunc_kwargs = {'output_sizes':{'bin':len(bins)+1}}
        )
    bins_out['bin'] = np.array([bins[0] - np.mean(np.diff(bins))/2,
                               *(bins[0:-1] + np.diff(bins)/2),
                               bins[-1] + np.mean(np.diff(bins))/2])
    
    return bins_out


def dmgf_params_max5d(ds,wwidth):
    ''' Calculate mean of yearly maximum of rolling mean

    '''
    
    # Get wraparound 
    ds = xr.concat([(ds.isel(dayofyear=slice(-wwidth+1,None)).
                              assign_coords({'year':ds.year.values+1}).
                              assign_coords({'dayofyear':(ds.isel(dayofyear=slice(-wwidth+1,None)).dayofyear.values - 
                                                          365)})),
                             ds.sel(dayofyear=slice(1,365)),
                             ds.sel(dayofyear=slice(1,wwidth-1)).
                                assign_coords({'year':ds.year.values-1}).
                                assign_coords({'dayofyear':np.arange(365+1,
                                                                     365+wwidth)}),
                           ],dim='dayofyear')
    # Remove thus created extra years at start and end, plus the years that 
    # would have nans in their rolling means 
    ds = ds.isel(year=slice(2,-2))
    
    # Rechunk so that dayofyear is still only one chunk... 
    # (possible performance hit here though?, but otoh, dayofyear
    # _should be_ coming in as one chunk from the start in 
    # this workflow...)
    ds = ds.chunk({'dayofyear':-1})

    # Calculate interannual mean of yearly maximum of rolling mean
    ds = ds.rolling(dayofyear=wwidth).mean().max('dayofyear').mean('year')

    return ds


def calc_uncerts(da,dim_order = ['sidx','proj_base','run','model'],
                 names = {'sidx':'uncreg','proj_base':'uncobs','run':'uncint','model':'uncmod'}):
    ''' Calculate variances

    Parameters
    -------------
    da : :py:class:`xr.DataArray`

    dim_order : :py:class:`list`

    names : :py:class:`dict`

    '''
    # Figure out which dimensions are in the dataset
    dim_order = [dim for dim in dim_order if dim in da.sizes]

    # Get functions to apply
    funcs = [np.roll([np.var,*([np.mean]*(len(dim_order)-1))],shift=shift) for shift in range(len(dim_order))]

    # Get names of output variables
    varnames = [names[dim] for dim in dim_order]

    # Apply 
    def apply_functions_sequentially(da, funcs, dims):
        """
        Apply a sequence of functions to specified dimensions in order.
    
        Parameters:
        - da (xarray.DataArray or xarray.Dataset): Input data.
        - func_dim_mapping (dict): Ordered mapping {function: dimension} specifying 
          the operations to apply in sequence.
    
        Returns:
        - xarray.DataArray or xarray.Dataset: Transformed data after sequential application.
        """
        for func, dim in zip(funcs,dims):
            da = func(da, axis=da.get_axis_num(dim))
        return da
    
    uncerts = xr.merge([apply_functions_sequentially(da,funclist,dim_order).to_dataset(name=varname)
                        for funclist,varname in zip(funcs,varnames)])

    # Get each as a fraction of their sum (`.to_array()` turns the dataset into a dataarray
    # with dimension 'variable')
    uncerts_norm = uncerts / uncerts.to_array().sum('variable')


    uncerts = xr.concat([uncerts,uncerts_norm],
          dim = pd.Index(['raw','norm'],name='typ'))
    

    return uncerts
