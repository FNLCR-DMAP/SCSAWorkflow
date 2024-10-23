# Analysis of SPAtial Single Cell Datasets (SPAC)

SPAC is a scalable, automated pipeline, under under the Single Cell Spatial Analysis Workflow (SCSAWorkflow) project aiming at analyzing single-cell spatial protein data of multiplexed whole-slide tissue images generated from technologies such as MxIF Codex and Imaging Mass Cytometry (IMC).
This Python-based package leverages the anndata framework for easy integration with other single-cell toolkits. It includes a multitude of functional and visualization modules, test utilities, and is capable of running in user-friendly web interfaces. Spac offers insights into cell interactions within various environments, aiding in studies of the cancer microenvironment, stem cell niches, and drug response effects etc.

## Establish SPAC working environment with Conda
Please run the following command to establish the Conda environment supporting usage and contribution to spac package:
```bash
cd <home directory of SCSAWorkflow folder>
# If conda is not activate
conda activate

# Create the Conda environment from environment.yml
conda env create -f environment.yml

# Once environment is established
conda activate spac
```
The envrionment works for Linux and noarc, if your are working on amd processor (commonly seen for latest Mac users), please replace the ` - numpy=1.19.5` with `numpy>=1.19.5,<2.0.0`

## License

`spac` was created by Fang Liu, Rui He, and George Zaki. It is licensed under the terms of the BSD 3-Clause license.

## Credits

`spac` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
