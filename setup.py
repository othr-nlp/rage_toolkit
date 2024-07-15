from setuptools import setup, find_packages

setup(
    name='rage_toolkit',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas'
    ],
    author='',
    author_email='your.email@example.com',
    description='A framework for retrieval augmented generation evaluation  (RAGE)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/',
    classifiers=[
    ],
    python_requires='>=3.6',
)