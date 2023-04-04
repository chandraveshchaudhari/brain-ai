# Template for python project
this project I created for easily launch project to 
pip and create the automated documentations

## Steps to upload to pip:
- first edit the setup.py and write in the details by replacing previous details
- install_requires need requirements of project to be automatically install before pip install your src
- version should start from 0.0.1
- LICENSE.txt already have MIT license, just change year and name or use https://choosealicense.com/ to select other 
license
- README.md learn from https://guides.github.com/features/mastering-markdown/
- skip to Generating distribution archives, upload to pip using instructions in https://packaging.python.org/tutorials/packaging-projects/ 
- optional: you can also share your documentation via https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html

## Steps to generating the documentation:
- good resources for understanding documentation https://realpython.com/documenting-python-code/#docstring-types
- we are using : pip install -U Sphinx
- go to docs folder using terminal and run : sphinx-quickstart
- edit config.py in docs folder
- uncomment "import sys" and put in next line
sys.path.append("/home/os_username/PycharmProjects/python_package_name/src")
- put the extension according to your docstring type.
- extension for numpy and google docstring put 'sphinx.ext.napoleon' in extensions. put 'sphinx.ext.autodoc' for reStructured docstring.
- use default options, just put project name, author name
- run : sphinx-apidoc -o . /home/os_username/PycharmProjects/python_package_name/src
- copy the modules.rst content above "Indices and tables" of index.rst
- to install other themes run : pip install sphinx-rtd-theme
- remember to edit config.py and change theme to 'sphinx_rtd_theme'
- run : make html
- if any errors occur run : make clean
- rerun : make html after fixing errors

#####  If you want to mention your source code with documentation. 
- paste following values in conf.py
```
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.coverage',
              'sphinx.ext.doctest',
              'sphinx.ext.extlinks',
              'sphinx.ext.ifconfig',
              'sphinx.ext.napoleon',
              'sphinx.ext.todo',
              'sphinx.ext.viewcode',
              ]
# Add `code-include` so that the code-include directives used in this documentation work
extensions += [
    "code_include.extension",
]
```
- pip install sphinx-code-include
- run : make html
