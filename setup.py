from setuptools import setup, find_packages

setup(
    name='faiss-imputer',
    version='0.1.0',
    description='Impute missing values using faiss',
    author='Hakkil Kim',
    author_email='scionkim@gmail.com',
    packages=find_packages(),
    install_requires=['numpy', 'faiss-cpu', 'scikit-learn']
)