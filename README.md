# The Bias-Corrected and Downscaled Massive Ensemble (BCD-ME)

This repository contains sample notebooks detailing how to access the BCD-ME and the code necessary to replicate it. 

For more information, see our preprint: 
- [Schwarzwald et al., submitted to _Nature Sci. Data_](https://eartharxiv.org/repository/view/11902/)

and see the BCD-ME pages on GDEX:
- [BCD-ME on GDEX](https://gdex.ucar.edu/datasets/d164444/dataaccess/)
and the Earthmover Data Marketplace: 
- [BCD-ME 1° bias-corrected (QDM) temperature time series](https://app.earthmover.io/marketplace/696aaa41490a002f0d47b8a1)
- [BCD-ME 0.25° bias-corrected (QDM) and downscaled temperature statistics](https://app.earthmover.io/marketplace/696aaa63490a002f0d47b8a4)


For sample code on how to access the BCD-ME, see [code/sample_data_access.ipynb](https://github.com/ks905383/bcd_me/blob/main/code/sample_data_access.ipynb). 

For best results, we recommend creating a new conda environment using the included `code/environment.yml` file: 

```
conda env create -f environment.yml

# or

mamba env create -f environment.yml

```

### Replication code usage
To replicate the processing of the BCD-ME or the analysis in the data descriptor: 
- ERA5, GMFD, and JRA-3Q reanalysis data can be downloaded using `preprocess_ERA5.ipynb`, `preprocess_GMFD.ipynb`, and `preprocess_JRA3Q.ipynb`, which assume access to the NCAR HPC system. These data can alternatively be downloaded using external links from their respective GDEX pages, or through the original published, but must be preprocessed as in these files.
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

A .csv named `dir_list.csv` is needed to run these notebooks, giving important paths. This csv should contain two columns, `dir_name` and `dir_path`, and rows corresponding to:
- `figs`: path to `figures` in this directory
- `aux`: path to `aux_data` in this directory
- `aux_bigmem`: path to scratch storage that can handle ~100GB of data
- `raw`: path to "raw" data (where pre-processed reanalysis and ESM data are stored, under sub-directories named after reanalysis or ESM names)
- `proc`: path to "processed" data (where bias-corrected and downscaled data are saved)
- `final`: path to directory where final collated zarr stores are saved