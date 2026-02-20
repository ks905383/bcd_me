# The Bias-Corrected and Downscaled Massive Ensemble (BCD-ME)

This repository contains sample notebooks detailing how to access the BCD-ME and the code necessary to replicate it. 

For more information, see the BCD-ME pages on Arraylake (and keep an eye out for our upcoming preprint): 
- [BCD-ME 1° bias-corrected (QDM) temperature time series](https://app.earthmover.io/marketplace/696aaa41490a002f0d47b8a1)
- [BCD-ME 0.25° bias-corrected (QDM) and downscaled temperature statistics](https://app.earthmover.io/marketplace/696aaa63490a002f0d47b8a4)


For sample code on how to access the BCD-ME, see [code/sample_data_access.ipynb](https://github.com/ks905383/bcd_me/blob/main/code/sample_data_access.ipynb). 

For best results, we recommend creating a new conda environment using the included `environment.yml` file: 

```
conda env create -f environment.yml

# or

mamba env create -f environment.yml

```