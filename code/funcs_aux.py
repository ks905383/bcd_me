import os
import re
import numpy as np
import fsspec
import dask
import zarr
import xarray as xr
from fsspec.implementations import local
import warnings
import shutil

from funcs_support import get_params,get_filepaths
dir_list = get_params()

def _create_filenames(gwl_info_rea,params_var,params_proc,file_rows=None):
    ''' Get filenames for processing

    NB: don't use in :py:meth:`preprocess_models()`, which is agnostic as to
    whether a model or reanalysis etc. product is included
    '''

    # Get directory structure for saving / loading
    dir_list = get_params()

    # Get reanalysis output and temporary filenames
    fns_out = {'reshp_rea':(dir_list['proc']+params_proc['mod_rea']+'/'+
                       '_'.join([params_var['var'],params_var['freq'],
                                 params_proc['mod_rea'],'historical','reanalysis','CACHE-RESHAPE',
                                 ])),
               'reshp_rea_fine':(dir_list['proc']+params_proc['mod_rea']+'/'+
                       '_'.join([params_var['var'],params_var['freq'],
                                 params_proc['mod_rea'],'historical','reanalysis',
                                 'FINEgrid','CACHE-RESHAPE',
                                 ])),
               'reshp_rea_finefromcoarse':(dir_list['proc']+params_proc['mod_rea']+'/'+
                       '_'.join([params_var['var'],params_var['freq'],
                                 params_proc['mod_rea'],'historical','reanalysis',
                                 'FINEgridNN','CACHE-RESHAPE',
                                 ])),
               'rea_rgrd_diffs':(dir_list['proc']+params_proc['mod_rea']+'/'+
                                 '_'.join([params_var['var']+'qdiff',params_var['freq'],
                                           params_proc['mod_rea'],'historical','reanalysis',
                                           (str(int(gwl_info_rea['start_year']))+'0101-'+
                                            str(int(gwl_info_rea['end_year']))+'1231'),
                                           'GWL'+re.sub(r'\.','-',str(gwl_info_rea['warming_level'])),
                                           'COARSE-FINE-scale',
                                           ])),
               'qrea':(dir_list['proc']+params_proc['mod_rea']+'/'+
                           '_'.join([params_var['var']+'q',params_var['freq'],params_proc['mod_rea'],
                                     'historical','reanalysis',
                                     (str(int(gwl_info_rea['start_year']))+'0101-'+
                                            str(int(gwl_info_rea['end_year']))+'1231'),
                                           'GWL'+re.sub(r'\.','-',str(gwl_info_rea['warming_level']))
                                    ]))}

    # Get model-run-specific output and temporary filenames
    if file_rows is not None:
        # Verify provided filepaths
        file_rows,mod,run,exp = _verify_file_rows(file_rows)

        # Get filenames for temporary files 
        fns_out = {**fns_out,'pctof':(dir_list['proc']+mod+'/'+
                           '_'.join(['pctof'+params_var['var'],params_var['freq'],
                                     mod,exp,run,'ALLGWLS'])),
                   'reshp_mod':(dir_list['proc']+mod+'/'+
                           '_'.join([params_var['var'],params_var['freq'],
                                     mod,exp,run,'CACHE-RESHAPE'])),
                   'qdiff':(dir_list['proc']+mod+'/'+
                           '_'.join(['q'+params_var['var']+'diff',params_var['freq'],
                                     mod,exp,run,'GWL'+re.sub(r'\.','-',str(gwl_info_rea['warming_level']))])+
                                    'projQDM-base'+params_proc['mod_rea']),
                   'qdiff_mod':(dir_list['proc']+mod+'/'+
                           '_'.join(['q'+params_var['var']+'diff',params_var['freq'],
                                     mod,exp,run,'GWL'+re.sub(r'\.','-',str(gwl_info_rea['warming_level']))])+
                                    'mod-mod_base'+params_proc['mod_rea']),
                   'biascrct_qm':(dir_list['proc']+mod+'/'+
                           '_'.join([params_var['var'],params_var['freq'],
                                    mod,exp,run,'ALLGWLS','projQM-base'+params_proc['mod_rea']])),
                   'biascrct_qdm':(dir_list['proc']+mod+'/'+
                           '_'.join([params_var['var'],params_var['freq'],
                                    mod,exp,run,'ALLGWLS','projQDM-base'+params_proc['mod_rea']]))}
        # Get filename for downscaled file (currently unused in workflow, 
        # except to generate below)
        for typ in ['qm','qdm']:
            fns_out['dwscld_'+typ] = fns_out['biascrct_'+typ]+'_dwnsclQPLAD-target025deg'

    # Add suffix, zarr ending to all filenames
    for fn in fns_out:
        # Add potential suffix
        if 'suffix' in params_var:
            fns_out[fn] = fns_out[fn]+'_'+params_var['suffix']
        # Add zarr file ending
        fns_out[fn] = fns_out[fn]+'.zarr'

    # Finish generating model-specific filenames
    if file_rows is not None:
        for typ in ['qm','qdm']:
            # Get filename for output damage function parameters
            fns_out['dmgf_params_'+typ] = re.sub(mod+r'\/'+params_var['var'],mod+'/'+params_var['var']+'dmgfparams',fns_out['dwscld_'+typ])
    
            # Get filenames for split up damage function parameters
            for var in ['binF','binC','sumpoly','mx5d']:
                fns_out[var+'_'+typ] = re.sub('dmgfparams',var,fns_out['dmgf_params_'+typ])
            
            # Get filename for output statistics (currently unused in workflow)
            fns_out['stats_'+typ] = re.sub(mod+r'\/'+params_var['var'],mod+'/'+params_var['var']+'stats',fns_out['dwscld_'+typ])
    
            # Percentiles of, from bias-corrected, on 0.25 grid
            fns_out['pctof_tiled'] = re.sub(r'\.zarr','_4x.zarr',fns_out['pctof'])

    return fns_out

def _load_gwls(gwl_source = 'leap_esgf',dir_list = get_params()):
    ''' Get years corresponding to GWLs for each model-run

    Parameters
    ----------------
    gwl_source : str, by default leap_esgf
        If `leap_esgf`, then gwl info is taken from the `gwl_ann_*_wsomeESGFruns.nc` file
        If `leap_amon`, then gwl info is taken from the `gwl_ann_*_fromAmon.nc` file
        If `mathause`, then gwl info is taken from Mathias Hauser's GWLs

    dir_list : by default the output of get_params()


    Returns
    -----------------
    gwl_info : pd.DataFrame
        pd DataFrame with multiindex model name and ensemble, giving start and years
        of 20-year chunks around the GWL definiton


    '''
    if gwl_source not in ['leap_esgf','leap_amon','mathause']:
        raise KeyError(f'`gwl_source` must in be in ["leap_esgf","leap_amon","mathause"] but is {gwl_source}')
    
    if gwl_source in ['leap_esgf','leap_amon']:
        if gwl_source == 'leap_esgf':
            gwls = xr.open_dataset(dir_list['aux']+'gwl_ann_CMIP6_ALLEXPs_ALLRUNs_1860-2090_wsomeESGFruns.nc')#fromAmon.nc')
        elif gwl_source == 'leap_amon':
            gwls = xr.open_dataset(dir_list['aux']+'gwl_ann_CMIP6_ALLEXPs_ALLRUNs_1860-2090_fromAmon.nc')
        gwls_calc = [0.61,1,1.5,2,2.5,3,3.5,4]
        
        gwls['start_year'] = xr.DataArray(np.ones((len(gwls_calc),gwls.sizes['memberid']))*np.nan,
                                          dims = ['warming_level','memberid'],
                                          coords = {'warming_level':gwls_calc,'memberid':gwls.memberid})
        gwls['end_year'] =  xr.DataArray(np.ones((len(gwls_calc),gwls.sizes['memberid']))*np.nan,
                                          dims = ['warming_level','memberid'],
                                          coords = {'warming_level':gwls_calc,'memberid':gwls.memberid})
        
        max_end_year = gwls.year.max()
        for member_id in gwls.memberid:
            maxt_counter = False 
            id_data = gwls.sel(memberid=member_id).tasanom.dropna('year',how='all')
            if id_data.sizes['year'] == 0:
                continue
            maxt = id_data.max()
            for gwl in gwls_calc:
                if gwl <= maxt:
                    # Get central year, using Mathias Hauser's definition
                    # First year for which the 20-year rolling average is
                    # greater than the GWL
                    central_year = id_data.year[np.where(id_data>gwl)[0][0]]
                    start_year = int(central_year - 20 / 2)
                    end_year = int(central_year + (20 / 2 - 1))
            
                    if end_year <= max_end_year:
                        # Append to GWLs
                        gwls['start_year'].loc[{'warming_level':[gwl],'memberid':[member_id]}] = start_year
                        gwls['end_year'].loc[{'warming_level':[gwl],'memberid':[member_id]}] = end_year
        
        gwl_info = gwls[['start_year','end_year']].to_dataframe().rename({'run':'ensemble','experiment':'exp'},axis=1)
        gwl_info = gwl_info.dropna(how='any')
        gwl_info['start_year'] = gwl_info['start_year'].astype(int)
        gwl_info['end_year'] = gwl_info['end_year'].astype(int)
        gwl_info = gwl_info.reset_index().drop('memberid',axis=1).set_index(['model','ensemble'])#,'exp','warming_level'])
    elif gwl_source == 'mathause':
        mathause_gwls = yaml.load(open('../aux_data/mathause-cmip_warming_levels-f47853e/warming_levels/'+
                   'cmip6_all_ens/cmip6_warming_levels_all_ens_1850_1900.yml'),
                  Loader=yaml.CLoader)
        def load_gwls(gwl_info):
            for k in gwl_info:
                gwl_info[k] = pd.DataFrame(gwl_info[k])
                gwl = re.split(r'\_',k)[-1]
                gwl = float(gwl[0]+'.'+gwl[1:None])
                gwl_info[k]['warming_level'] = gwl
            gwl_info = pd.concat(gwl_info)
        
            gwl_info = gwl_info.reset_index().drop(['level_0','level_1'],axis=1).set_index(['model','ensemble'])
            return gwl_info
        
        gwl_info = load_gwls(mathause_gwls)

    return gwl_info

def _find_main_variable(ds):
    ''' Identifies the variable with the highest dimensionality in a dataset
    '''

    sizes = {var:len(ds[var].sizes) for var in ds}
    var = [k for k in sizes][np.argmax([v for k,v in sizes.items()])]
    if len(sizes)>1:
        warnings.warn('No processing variable given; assuming processing variable is '+var+
                      ', since it has the most dimensions.')
    return var

def _remove_chunk_encoding(ds):
    # xarray carries around the old encoding structure, with the original
    # chunks, this is a workaround until it's fixed in xarray
    # (see https://github.com/pydata/xarray/issues/5219#issuecomment-828071017)
    for var0 in [*[v for v in ds],*[v for v in ds.coords]]:
        if 'chunks' in ds[var0].encoding:
            del ds[var0].encoding['chunks']
    if 'chunks' in ds.encoding:
        del ds.encoding['chunks']
    return ds

def _restore_doys(ds,params_proc):
    ''' Remove extra duplicate DOYs from reshape_for_rolling
    '''
    # Subset to the 365 days actually processed 
    ds = ds.isel(dayofyear=slice(int(params_proc['wwidth']/2),-int(params_proc['wwidth']/2)))
    doys = ds['dayofyear'].values
    doys[doys>365] = doys[doys>365] % 365
    ds['dayofyear'] = doys
    ds = ds.sortby('dayofyear')
    return ds

def _verify_file_rows(file_rows):
    ''' Verify suitability of filepaths in `file_rows`
    To be used mainly by `preprocess_models`. Make sure
    `file_rows` has files of only:
        - one model
        - one run
        - one exp (+ historical)
    ''' 
    standard_exception_str = 'Make sure `file_rows` has files for only one model, run, experiment (+ historical)'
    
    mod = np.unique(file_rows.model)
    if len(mod)>1:
        raise Exception('More than one model is found in `file_rows`: '+', '.join(mod)+'. '+standard_exception_str)
    mod = str(mod[0])

    run = np.unique(file_rows.run)
    if len(run)>1:
        raise Exception('More than one run is found in `file_rows`: '+', '.join(run)+'. '+standard_exception_str)
    run = str(run[0])
    
    exps = np.unique(file_rows.exp)
    if len(exps)>2: 
        raise Exception('More than two exps are found in `file_rows`: '+', '.join(exps)+'. '+standard_exception_str)
    if len(exps) > 1:
        exp = [exp for exp in exps if exp != 'historical'][0]
    else:
        exp = exps[0]
    exp = str(exp)

    return file_rows,mod,run,exp

def _verify_gwl_range(file_rows,gwl_info,gwl_info_rea,years_from_files=False):
    ''' Figure out which GWLs can be analysed with local data
    (by timeframe)

    Raises error if the reference GWL timeframe isn't present 
    in local data. 

    Returns `gwl_info`, but subset to only GWLs with local data. 
    '''

    if years_from_files:
        # TODO: do more sophisticated than just the first / last year 
        # across all local files - use actual file ranges
        # Get start / end years for local files
        start_years = [xr.open_dataset(fn).time.dt.year.min().values for fn in file_rows.path]
        end_years = [xr.open_dataset(fn).time.dt.year.max().values for fn in file_rows.path]
    else:
        start_years = [int(t[0:4]) for t in file_rows.time]
        end_years = [int(t[9:13]) for t in file_rows.time]
    
    # If the local files start after the reference GWL timeframe, 
    # then drop the analysis for this model / exp / run, since 
    # won't be able to compare to the reference GWL
    mod_ref_gwl = gwl_info.loc[gwl_info.warming_level == gwl_info_rea.warming_level,:]
    if np.min(start_years) > mod_ref_gwl.start_year.values[0]:
        raise Exception('The local files '+
                      ' start later than the required reference GWL ('+
                      str(gwl_info_rea.warming_level)+', '+
                      '-'.join([str(y.values[0]) for y in [mod_ref_gwl.start_year,mod_ref_gwl.end_year]])+
                      ') for the same model / exp / run combination.')
    
    
    # Now, figure out which GWLs overlap with the local file range
    gwl_info = gwl_info.sort_values('start_year')
    keep_gwls = ((gwl_info.start_year>=np.min(start_years)) &
                 (gwl_info.end_year<=np.max(end_years)))
    
    print('Using GWLs '+', '.join([str(t) for t in 
                    gwl_info.where(keep_gwls).dropna(how='all').warming_level]))
    
    # Drop GWLs from analysis that don't overlap with local file range
    if np.any(~keep_gwls):
        warnings.warn('(dropping GWLs '+', '.join([str(t) for t in 
                    gwl_info.where(~keep_gwls).dropna(how='all').warming_level])+
                     ' since the local files do not overlap temporally with the GWL range)')
    gwl_info = gwl_info.where(keep_gwls).dropna(how='all')

    return gwl_info

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

def extract_gwl(ds,gwl_row):
    ''' Subsample dataset by GWL start/end years along new `gwl` dimension
    '''
    ds = ds.sel(year=slice(gwl_row[1].start_year,
                             gwl_row[1].end_year))
    ds = ds.expand_dims({'gwl':[gwl_row[1].warming_level]})
    ds['year'] = np.arange(1,ds.sizes['year']+1)
    ds = ds.chunk({'year':-1})
    return ds

def dask_isel(ds, dim, idxs):
    """
    Use dask arrays for isel-like operation on xarray.Dataset.
    (Made possible through `xr.map_blocks()`, since `ds.isel()` does
    not work with dask arrays as indices)

    Parameters:
        ds: xarray.Dataset - the input dataset
        idxs: xarray.DataArray - integer indices with dimensions (lat, lon)

    Returns:
        xarray.Dataset with q dimension indexed by idxs
    """
    # .isel has been actually super fast, faster than a numba indexer
    # (I think because at the end of the day, you're calling 
    # the above separately 568*1440*20*5 times, and each time
    # it's only indexing on a relatively small, 620-unit thing,
    # so it's not actually providing a great speedup. Using `dask_isel()` instead.     
    # Get downscaled onto 0.25 grid by adding 1-->0.25 deg transfer function
    # for each value, at the relative quantile of each of those values
    def subset_func(ds, dim, idxs):
        # Extract the corresponding block and subset along the 'q' dimension
        return ds.isel({dim:idxs})

    # Transpose in same dimension order (required by map_blocks)
    idxs = idxs.transpose(*[*[d for d in ds.sizes if d in idxs.sizes],
                                                 *[d for d in idxs.sizes if d not in ds.sizes]])

    # map_blocks runs functions on blocks that are loaded into memory
    return xr.map_blocks(
        subset_func,
        ds,
        args=(dim,idxs,),
        template=idxs#ds.isel({dim:0}),  # Provide a template for the resulting structure
    )

def get_landmask(ds,lm_source = 'carleton'):
    ''' Get landmask for a lat / lon grid 

    Parameters
    ---------------
    ds : :py:meth:`xr.Dataset` or :py:meth:`xr.DataArray`
        An xarray object containing a lat / lon grid (in a format 
        interpretable by `xagg`

    lm_source : str, by default `'carleton'`
        Polygons used to determine landmask. 
        - if 'carleton', then uses CIL / Carleton et al. impact regions
    
    Returns
    ---------------
    landmask : :py:meth:`xr.DataArray`
        A datarray returning 1 if a grid cell contains land

    '''
    import geopandas as gpd
    import xagg as xa
    
    if lm_source == 'carleton':
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore') # Silencing invalid winding order from gpd.read_file()
            regs_for_lm = gpd.read_file(f'{dir_list['aux']}geo_data/impact-region.shp').make_valid()
        # Turn into single geometry
        regs_for_lm = regs_for_lm.union_all()
    
    # Create polygons for each grid cell
    pix_polys = xa.core.create_raster_polygons(ds)['gdf_pixels']
    # Get locations of grid cells with intersecting bounding boxes with test geom
    int_bboxes = list(pix_polys.sindex.query(regs_for_lm, predicate='intersects'))
    # Now, filter for just locs that truly intersect (not just bbox)
    int_full = pix_polys.iloc[int_bboxes][pix_polys.iloc[int_bboxes].intersects(regs_for_lm)]
    
    # Now, mark land mask in original grid
    pix_polys = pix_polys.set_index('pix_idx')
    pix_polys.loc[int_full.pix_idx,'land'] = 1
    pix_polys.loc[:,'land'] = pix_polys.loc[:,'land'].where(~np.isnan(pix_polys.loc[:,'land']),0)
    
    # Turn back into xarray
    landmask = pix_polys.set_index(['lat','lon']).loc[:,'land'].to_xarray()

    return landmask


def repeat_ds(ds,n):
    ''' Repeat each value in lat/lon dimensions

    '''

    nlat,nlon = ds.sizes['lat'],ds.sizes['lon']

    # Get output coordinates that equally split the 
    # input lat/lon step 
    coord_diffs = dict()
    for dim in ['lat','lon']:
        og_geo_step = np.unique(ds[dim].diff(dim))[0]
        new_geo_step = og_geo_step/n
        coord_diffs[dim] = np.arange(-og_geo_step/2+new_geo_step/2,og_geo_step/2,og_geo_step/n)
    
    # The poor man's nearest-neighbor regridding
    # (waaaaay faster than xesmf's nearest_s2d)
    # First, select each lat/lon coordinate N times
    ds = ds.sel(lat=np.repeat(ds.lat,n).values,
                lon=np.repeat(ds.lon,n).values)
    
    # Then, change the coordinates to the new, finer
    # coordinates
    ds['lat'] = ds.lat + np.tile(coord_diffs[dim],nlat)
    ds['lon'] = ds.lon + np.tile(coord_diffs[dim],nlon)

    return ds

def subset_idv(ds,
                nruns = 5,
                nruns_min = 5,
                output = 'ds', # or 'idvs' for just returning the subset indices
                group_coords = ['model','proj_base'], 
                # (can't deal with multiple string 
                # coordinates at this point)
                other_subsets = {'gwl':[0.61,1,2,3]},
                # Variable for which to look for nans
                ref_var = 'dmort_carleton',
                # Dimension across which to look for nans
                ref_dim = 'hierid',
                # if not None, drops values for which at least 
                # X% of the ref_var is nan along this dimension.
                # Helps capture corrupt / only partially processed 
                # files (the landmask nan issue affects ~ 2% of 
                # polygons, so a threh of .95 is reasonable)
                ref_dim_nonanthresh = 0.95,
                # If true, then also filters on coords of
                # the first element of `group_coords`, 
                # dropping any coordinates that has any 
                # all-nan slices
                balanced_panel = True 
              ):
               
    if output not in ['idvs','ds']:
        raise KeyError('`output` must be "idvs" or "ds"')
    
    if type(ds) == xr.core.dataarray.DataArray:
        if ds.name is None:
            var = 'tmpvar'
        else:
            var = ds.name
        ds = ds.to_dataset(name=ref_var)
        from_dataarray = True
    else:
        from_dataarray = False

    # Other 
    subset_dims = [dim for dim in other_subsets]
    
    # Subset by 
    if len(other_subsets)>0:
        ds = ds.sel(**other_subsets)
    
    # 1) Find indices of non-ref_dim coords (e.g., `idv`, `gwl`) where everything is nan
    # Get a mask for *all* locations (technically, all values along
    # `ref_dim`, but generally will be `hierid`) being nan (since
    # sometimes, *some* locations are nan without it being an issue)
    if ref_dim_nonanthresh is None:
        all_locs_nan_mask = ds[ref_var].isnull().all(dim=ref_dim)
    else:
        # Alternately, filter by those where too much of the ref dim is nan
        all_locs_nan_mask = (ds[ref_var].isnull().sum(dim=ref_dim)/ds.sizes[ref_dim])>(1-ref_dim_nonanthresh)
    
    # 2) Find idvs that have data for all `other_subsets`  
    # Get list of coords in idv that have nans in any dim in 
    # `other_subsets`
    nan_idvs = all_locs_nan_mask.any(dim=subset_dims)
    
    ds = ds.sel(idv = (~nan_idvs))
    
    # Now, find idvs for which there are nruns_min <= n <= nruns runs for each
    # `group_coords` dim
    idvs = ds.idv.load()
    
    # Reset multiindex, since it doesn't play nice with the following pandas steps
    if len(idvs.indexes) != 0:
        idvs = idvs.reset_index('idv')
    
    # To pandas
    idvs = idvs.to_dataframe()
    
    # Get number of runs across group coords
    idv_nruns = idvs.groupby(group_coords)['idv'].count()
    
    # Keep those with at least nruns_min runs
    idv_nruns = idv_nruns.where(idv_nruns >= nruns_min).dropna()
    
    # Keep only `nruns` runs 
    idvs = (idvs.set_index(group_coords).loc[idv_nruns.index].
            reset_index(group_coords). # reset to allow groupby
            groupby(group_coords).head(nruns))
    
    if len(idvs) == 0:
        raise Exception('No `group_coords` combinations have at least '+str(nruns_min)+' runs with non-nan data.')
    
    # Subset
    ds = ds.sel(idv=idvs.idv.values)

    if output == 'idvs':
        if balanced_panel:
            warnings.warn('Cannot yet guarantee a balanced panel with output == "idvs".')
        return idvs
    elif output == 'ds':
    
        # Make the `runs` variable generic (meaning, just integer instead of metadata) 
        # in the dataframe...
        idvs['run'] = idvs.groupby(group_coords).cumcount()
        # ... and apply it to the ds
        ds['run'] = ('idv',idvs['run'].values)
        
        # Reindex now along the group_coords + the new generic run variable
        ds = (ds.reset_index('idv').
         drop_vars([dim for dim in ds.idv.coords if dim not in ['idv','run',*group_coords]]).
         set_index(idv = [*group_coords,'run']))
        
        # And now, unstack along 'idv' to create *group_coords, 'idv' dims
        ds = ds.unstack('idv')
    
        if balanced_panel:
            # If requiring a balanced panel, assuming the first listed group_coord
            # is the most important one, and dropping coordinates along that if there 
            # are any sub-coordinates that have all nans on the ref_dim
            ds = ds.where(~(ds[ref_var].isnull().all(ref_dim).all('run').
                            any([*subset_dims,*group_coords[1:None]])),drop=True)
    
        return ds
