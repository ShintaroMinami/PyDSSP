from setuptools import setup
from codecs import open
from os import path

dir_path = path.abspath(path.dirname(__file__))

with open(path.join(dir_path, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pydssp',
    packages=['pydssp'],
    license='MIT',
    url='https://github.com/ShintaroMinami/PyDSSP',
    description='A simplified implementation of DSSP algorithm for PyTorch and NumPy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['DSSP', 'Secondary Structure', 'Protein Structure'],

    author='Shintaro Minami',
    author_twitter='@shintaro_minami',

    use_scm_version={'local_scheme': 'no-local-version'},

    setup_requires=['setuptools_scm'],
    install_requires=['numpy', 'torch', 'einops', 'tqdm'],
 
    include_package_data=True,
    scripts=[
        'scripts/pydssp',
    ],

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)