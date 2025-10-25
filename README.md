# Analysis of SPAtial Single Cell Datasets (SPAC)

SPAC is a scalable, automated pipeline, under the Single Cell Spatial Analysis Workflow (SCSAWorkflow) project aiming at analyzing single-cell spatial protein data of multiplexed whole-slide tissue images generated from technologies such as MxIF Codex and Imaging Mass Cytometry (IMC).
This Python-based package leverages the anndata framework for easy integration with other single-cell toolkits. It includes a multitude of functional and visualization modules, test utilities, and is capable of running in user-friendly web interfaces. Spac offers insights into cell interactions within various environments, aiding in studies of the cancer microenvironment, stem cell niches, and drug response effects etc.

## Installing SPAC with Conda
Run the following command to establish the Conda environment supporting usage and contribution to spac package:
Latest released version is v0.9.0 at 5/23/2025
```bash
cd <home directory of SCSAWorkflow folder>
git checkout origin/address_reviewer_comments
# If conda is not activate
conda activate

# Adding constumized scimap conda pacakge channel supported by DMAP
conda config --add channels https://fnlcr-dmap.github.io/scimap/

# Create the Conda environment from environment.yml
conda env create -f environment.yml

# Once environment is established
conda activate spac

# Install the SPAC package in development mode
pip install -e .
```
The envrionment works for Linux and noarc, if your are working on amd processor (commonly seen for latest Mac users), please replace the ` - numpy=1.19.5` with `numpy>=1.19.5,<2.0.0`

If error occurs suggesting SSL certificate not found for our scimap channel, please run the following command before the environment creation:
```
conda config --set ssl_verify false
```
Then set the verification to True after the installation:
```
conda config --set ssl_verify true
```

## Using SPAC with Docker

For a reproducible environment, you can use Docker to run SPAC:

### Build the Docker Image
```bash
docker build -t spac .
```

### Run Jupyter Notebook Server with Your Data
Mount your working directory to access notebooks and data:
```bash
# Stop any existing containers using port 8888 (if needed)
docker stop $(docker ps -q --filter "publish=8888") 2>/dev/null || true

# From the project root directory
docker run --rm -p 8888:8888 -v $(pwd)/paper/examples:/workspace spac

# Or mount any directory containing your notebooks and data
docker run --rm -p 8888:8888 -v /path/to/your/data:/workspace spac
```

Then open your browser to: `http://localhost:8888`

### Test SPAC Installation
To validate that SPAC works correctly, run the notebook execution test:
```bash
# Navigate to the paper directory  
cd paper

# Run the test script in Docker (mounts examples directory and test script)
docker run --rm -v $(pwd)/examples:/workspace -v $(pwd)/test_notebook_execution.sh:/test_script.sh spac bash /test_script.sh
```

This test will:
- ✅ Verify SPAC and scimap installation
- ✅ Execute the example lymphnode analysis notebook  
- ✅ Create a timestamped output file (e.g., `lymphnode_analysis_executed_20231023_134803.ipynb`)
- 📓 Provide instructions for viewing results in Jupyter

### View Executed Notebooks  
After running the test, you can view the executed notebook in Jupyter:
```bash
# Navigate to paper/examples directory
cd paper/examples

# Stop any existing containers using port 8888 (if needed)
docker stop $(docker ps -q --filter "publish=8888") 2>/dev/null || true

# Start Jupyter server with your data mounted
docker run --rm -p 8888:8888 -v $(pwd):/workspace spac
```

Then open your browser to: `http://localhost:8888` and navigate to the timestamped executed notebook file.

### Validate Performance Improvements
To verify the performance speedups implemented in SPAC:
```bash
# Navigate to the paper directory  
cd paper

# Run the performance validation script in Docker (mounts current directory for results)
# Note: Ensure Docker has at least 16GB memory allocated for full validation
docker run --rm -v $(pwd):/workspace -v $(pwd)/test_performance_validation.sh:/test_script.sh spac bash /test_script.sh
```

This test will:
- ✅ Run boxplot performance benchmarks (up to 10M cells: `boxplot` vs `boxplot_interactive`)
- ✅ Run histogram performance benchmarks (up to 10M cells: `histogram_old` vs `histogram`)  
- ✅ Generate detailed speedup analysis with concrete performance improvements
- 📊 Generate detailed speedup analysis and performance reports
- � Save results to your local `performance_results/` directory

Performance results are saved locally as timestamped files in `performance_results/`:
- `boxplot_performance_YYYYMMDD_HHMMSS.txt`
- `histogram_performance_YYYYMMDD_HHMMSS.txt`

### Interactive Shell Access
For debugging or manual exploration:
```bash
docker run --rm -it spac bash
```

### Mount Local Data
To work with your own data, mount a local directory:
```bash
# Stop any existing containers using port 8888 (if needed)
docker stop $(docker ps -q --filter "publish=8888") 2>/dev/null || true

# Mount your data directory
docker run --rm -p 8888:8888 -v /path/to/your/data:/data spac
```

### Docker Cleanup
If you need to clean up Docker resources:
```bash
# Stop all SPAC containers
docker stop $(docker ps -q --filter "ancestor=spac") 2>/dev/null || true

# Remove stopped containers (optional)
docker container prune -f

# Remove SPAC image (if you want to rebuild from scratch)
docker rmi spac
```

## Contirbuting to SPAC:
Review the [developer guide](CONTRIBUTING.md)

## License

`spac` was created by Fang Liu, Rui He, and George Zaki. It is licensed under the terms of the BSD 3-Clause license.

## Credits

`spac` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
