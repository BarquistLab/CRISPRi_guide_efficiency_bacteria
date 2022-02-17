from distutils.core import setup

setup(
    name='CrispriGuideEfficiency',
    version='0.1',
    packages=['src'],
    install_requires=['pandas','numpy','scikit-learn','scipy','pytorch_lightning','seaborn','matplotlib'],
    long_description=open('README.md').read(),
    author='Lisa B. A. Sousa & Erinc Merdivan',
)