from setuptools import setup, find_packages

setup(
    name='spac',
    version="0.7.15",
    description=(
        'SPatial Analysis for single-Cell analysis (SPAC)'
        'is a Scalable Python package for single-cell spatial protein data '
        'analysis from multiplexed whole-slide tissue images.'
    ),
    long_description=(
        'SPAC is a scalable, Python-based package for analyzing single-cell '
        'spatial protein data from multiplexed whole-slide tissue images. It '
        'integrates with other single-cell toolkits through the anndata '
        'framework and provides various functional and visualization modules. '
        'SPAC enables user-friendly web interfaces and offers insights into '
        'cell interactions in diverse environments, benefiting studies of '
        'cancer microenvironments, stem cell niches, and drug responses.'
    ),
    author='Fang Liu, Rui He, and George Zaki',
    url='https://github.com/FNLCR-DMAP/SCSAWorkflow',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas==1.4.3',
        'anndata==0.8.0',
        'numpy==1.19.5',
        'scikit-learn==1.1.1',
        'squidpy==1.2.2',
        'matplotlib==3.5.2',
        'seaborn==0.12.2',
        'scanpy==1.8.0',
        'phenograph==1.5.7',
        'zarr==2.12.0',
        'numba',
        'Pillow',
        'datashader',
        'plotly'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD 3-Clause License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
