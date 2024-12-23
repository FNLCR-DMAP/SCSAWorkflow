from setuptools import setup, find_packages

setup(
    name='spac',
    version="0.7.16",
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
        'pandas',
        'anndata',
        'numpy',
        'scikit-learn',
        'squidpy',
        'matplotlib',
        'seaborn',
        'scanpy',
        'phenograph',
        'zarr',
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
