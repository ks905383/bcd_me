[![DOI](https://zenodo.org/badge/1162734427.svg)](https://doi.org/10.5281/zenodo.22311273)

# The Bias-Corrected and Downscaled Massive Ensemble (BCD-ME)

This repository contains sample notebooks detailing how to access the BCD-ME and the code necessary to replicate it. 

For more information, see the data descriptor: 
- [Schwarzwald et al., 2026, _Sci. Data_](https://www.nature.com/articles/s41597-026-08391-w)

and see the BCD-ME pages on NSF NCAR GDEX:
- [BCD-ME on GDEX](https://gdex.ucar.edu/datasets/d164444/dataaccess/)
  
and the Earthmover Data Marketplace: 
- [BCD-ME 1° bias-corrected (QDM) temperature time series](https://app.earthmover.io/marketplace/696aaa41490a002f0d47b8a1)
- [BCD-ME 0.25° bias-corrected (QDM) and downscaled temperature statistics](https://app.earthmover.io/marketplace/696aaa63490a002f0d47b8a4)


For sample code on how to access the BCD-ME, see [code/sample_data_access_gdex.ipynb](https://github.com/ks905383/bcd_me/blob/main/code/sample_data_access_gdex.ipynb) for access through GDEX or [code/sample_data_access_earthmover.ipynb](https://github.com/ks905383/bcd_me/blob/main/code/sample_data_access_earthmover.ipynb) for access through Earthmover. Both notebooks also contain slightly different examples for some of the many things one can do with this data!

For best results, we recommend creating a new conda environment using the included `code/environment.yml` file: 

```
conda env create -f environment.yml

# or

mamba env create -f environment.yml

```

### Known issues
We strive to provide a perfect dataset, but every once in a while, a bug slips through our nets. These bugs are patched regularly on Earthmover, but take longer to patch on GDEX. We have a helper function that nans out any known issues, see [code/sample_data_access_gdex.ipynb](https://github.com/ks905383/bcd_me/blob/main/code/sample_data_access_gdex.ipynb) for details. 

#### On Earthmover
- None known

#### On GDEX
- `qdm`: 
	- Some members with faulty data (3 for `tas`, 6 for `tasmax`), can be filtered through `bcd_me_helpers.filter_problems()`
- `qdm-qplad`:
	- Some members with faulty data (8 for `tas`-based variables, 6 for `tasmax`-based variables), can be filtered through `bcd_me_helpers.filter_problems()`
- `qdm-byyr`:
	- Some files with faulty data (7 for `tas`, 3 for `tasmax`), can be filtered through `bcd_me_helpers.filter_problems()` 
	- GWL metadata confusing; these files are stored by calendar year, not GWL (with GWL 0.61 == 1982-2001)

### Citing the BCD-ME
If you use the BCD-ME, please cite both the data descriptor in _Scientific Data_ and also the exact source of the data (GDEX, Earthmover, etc.). If you can, cite the version as well (present in the metadata of each file).  The data descriptor provides more details. 


### License
Data bias-corrected and downscaled to ERA5, MERRA-2, and GMFD is available under a [CC BY](https://creativecommons.org/licenses/by/4.0/) 4.0 Creative Commons Attribution license (in other words, you may do what you want with it as long as you attribute where you got the data from and if any changes were made, and cite us appropriately). Data bias-corrected and downscaled to JRA-3Q is available under a [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) 4.0 license ("non-commercial, share-alike", which means you may only use it for non-commerical means, and if you remix, transform, or build upon the material, you have to share it under the same or a compatible license). 

### Replication code usage
To replicate the processing of the BCD-ME or the analysis in the data descriptor: 
- ERA5, GMFD, and JRA-3Q reanalysis data can be downloaded using `preprocess_ERA5.ipynb`, `preprocess_GMFD.ipynb`, and `preprocess_JRA3Q.ipynb` (which, like the files below, can be found in the `code` subdirectory), which assume access to the NCAR HPC system. These data can alternatively be downloaded using external links from their respective GDEX pages, or through the original published, but must be preprocessed as in these files
- MERRA-2 reanalysis data was downloaded from the NASA GES DISC [archive](https://doi.org/10.5067/9SC1VNTWGWV3)
- CMIP6 ESM data was downloaded from the Pangeo data store through `preprocess_cmip6_general.ipynb`
- 1$^\circ$ bias-corrected time series were created through `bias_correct_qdm.ipynb`
- 0.25$^circ$ bias-corrected and downscaled statistics were created through `downscale_qplad.ipynb`
- Figure 2 was created through `figure_ssps_gwls.ipynb`
- Figures 3 and 4 were created through `diag_qdm.ipynb`
- Figure 5 was created through `diag_vs_nexgddp.ipynb`
- Figure 6 was created through `figure_uncertpart.ipynb`
- Diagnostics and verifications were run through `diag_final_files.ipynb` and `diag_nans_extremes.ipynb`
- Data was collated through `transfer_to_campaign.ipynb` and uploaded to arraylake through `prep_for_earthmover.ipynb`

A .csv named `code/dir_list.csv` is needed to run these notebooks, giving important paths. This csv should contain two columns, `dir_name` and `dir_path`, and rows corresponding to:
- `figs`: path to `figures` in this directory
- `aux`: path to `aux_data` in this directory
- `aux_bigmem`: path to scratch storage that can handle ~100GB of data
- `raw`: path to "raw" data (where pre-processed reanalysis and ESM data are stored, under sub-directories named after reanalysis or ESM names)
- `proc`: path to "processed" data (where bias-corrected and downscaled data are saved)
- `final`: path to directory where final collated zarr stores are saved
