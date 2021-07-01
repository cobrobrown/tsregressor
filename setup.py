import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='tsregressor',
    version='0.1.0',    
    description='Time series forecasting with lagged features and exogenous variables.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cobrobrown/tsregressor",
    author='Conner Brown',
    author_email='cobrobrown@gmail.com',
    license='BSD 2-clause',
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.8',
    ],
)