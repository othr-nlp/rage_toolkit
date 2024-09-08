from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='rage_toolkit',
    version='0.0.2',
    packages=find_packages(),
    install_requires=[
        'pandas',
    ],
    author='Vinzent Penzkofer, Timo Baumann',
    author_email='vinzent.penzkofer@outlook.de, timo.baumann@oth-regensburg.de',
    description='A framework for retrieval augmented generation evaluation (RAGE)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/othr-nlp/rage_toolkit',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    keywords='retrieval, augmented generation, evaluation, RAG, NLP',
    include_package_data=True,
)