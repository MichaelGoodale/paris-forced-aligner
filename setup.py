import setuptools

setuptools.setup(name='paris_forced_aligner',
      version='0.01',
      description='Forced alignment using Wav2Vec2',
      url='http://github.com/michaelgoodale/paris-forced-aligner',
      author='Michael Goodale',
      author_email='michael.goodale@mail.mcgill.ca',
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
      ],
      packages=setuptools.find_packages(),
      python_requires='>=3.7')
