from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mosna',
    version='0.0.1',
    description='multi-omics spatial network analysis library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = "https://github.com/AlexCoul/mosna",
    author='Alexis Coullomb',
    author_email='alexis.coullomb.pro@gmail.com',
    license='GNU GPLv3',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: GNU GPLv3',
                 'Programming Language :: Python :: 3.8',
                 'Operating System :: OS Independent'],
    packages=find_packages(exclude=['build', 'docs', 'templates', 'data']),
    include_package_data=True,
    install_requires=['matplotlib>=2.2.3', 'numpy', 'seaborn',
                      'pandas>=1.4.3','scipy', 'statsmodels', 
                      'scikit-image', 'scikit-learn', 'ipykernel',
                      'pyarrow', 'tqdm', 'napari', 'colorcet',
                      'composition_stats', 'tysserand', 'ipywidgets',
                      'igraph', 'leidenalg', 'openpyxl', 'odfpy',
                      'fastcluster', 'lifelines', 'hdbscan', 'umap-learn',
                      'dask',
                     ],
    keywords = 'spatial networks cells transcriptomics sociology econometrics'
)
