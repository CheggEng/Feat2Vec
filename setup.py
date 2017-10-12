from setuptools import setup

setup(name='feat2vec',
      version='0.1.0',
      description='Feat2Vec Library',
      author='Anonymous Author for Peer Review',
      packages=[
                'feat2vec'

               ],
      test_suite='nose.collector',
      tests_require=['nose'],
      setup_requires=[
                    'scipy>=0.17.1',
                    'numpy>=1.10.4',
                    'Cython>=0.23.5',
      ],
      install_requires=[
                     "tensorflow>=1.1.0",
                     "keras>=2.0.8",
                     "dask>=0.15.0", 
                     "pandas>=0.19.2"
                     "numpy>=1.13.1"
                     "tables>=3.3.0"
      ],
      extras_require={
                    'test': [
                              "nltk==3.2.2"
                            ]
      },
      zip_safe=False)
