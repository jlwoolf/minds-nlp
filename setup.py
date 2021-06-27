try:
    from setuptools import setup

    setup  # quiet "redefinition of unused ..." warning from pyflakes
    # arguments that distutils doesn't understand
    setuptools_kwargs = {
        'install_requires': [
            'gensim',
            'python-Levenshtein',
        ],
        'provides': ['minds_nlp'],
    }
except ImportError:
    from distutils.core import setup

    setuptools_kwargs = {}

setup(name='minds-nlp',
      version="1.0",
      description=(
          """
          Library for generating NLP datasets for MIL
          """
      ),
      author='Jonathan Woolf',
      author_email='jlwoolf@protonmail.com',
      url='https://github.com/garydoranjr/misvm.git',
      packages=['minds_nlp'],
      platforms=['unix'],
      scripts=[],
      **setuptools_kwargs)