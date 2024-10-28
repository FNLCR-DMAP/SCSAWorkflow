# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Types of Contributions

### Report Bugs

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

You can never have enough documentation! Please feel free to contribute to any
part of the documentation, such as the official docs, docstrings, or even
on the web in blog posts, articles, and such.

### Submit Feedback

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `spac` for local development.

1. Download a copy of `spac` locally.
2. Install `spac` using `poetry`:

    ```console
    $ poetry install
    ```

3. Use `git` (or similar) to create a branch for local development and make your changes:

    ```console
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

4. When you're done making changes, check that your changes conform to any code formatting requirements and pass any tests.

5. Commit your changes and open a pull request.


## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include additional tests if appropriate.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for all currently supported operating systems and versions of Python.


## SPAC terminologies:

SPAC adopted a general terminlogy to abstract technical terms used in AnnData object from less technical users Here is the guide: 

* **cells:** Rows n the X matrix in AnnData
* **Features:** Columns in the X matrix in AnnData. Can represent gene expression of antibody intensity.
* **Tables:** Originally **layers** in AnnData, to represent transformed features
* **Associated Tables:** Corresponds to .obsm in Anndata. Can store spatial coordinates, UMAP, etc.
* **Annotation:** Corrresponds to .obs in AnnData. Can store cells phenotypes, experiment name, slideID, etc.

## General Tips on Writing Code 

* API: Isolate the core functionalities that works with numpy/dataframes from the high level functions that deals with adata if needed.  
* Code should be modular, and functions should do one thing. 
* Data are represented using the AnnData format. See anndata - https://anndata.readthedocs.io/en/latest/
  * Rows of obs,obsm,X,etc correspond to cells. Columns of var, X, varm, etc correspond to features (usually genes but could be protein markers or any single-cell measurement). 
* All contributed code should have unittests. These are stored in the **tests/** directories. There should be a test for each file and function. For example, the file **src/transformations.py** has tests in **tests/test_transformations/*** 
* Functions that change adata for dataframe in place, should not return that object.
  * There are some notable exceptions to this but the general rule is that inplace updates should be inplace  
* Landing functions start with adata, use standard variable names for layer, annotation, associated tables, etc.
* Use the numpy style guide for documenting the code, and standard pytho PEP 8 for code style 

 

### Unittests 

1. Start with the simplest scenario  
   1. Use minimum number of cells, markers, observations, etc. 
2. Check corner cases  
   1. Test with one feature, one cell, one annotation 
3. Trigger and test handled exceptions  
4. Aim for comprehensive code coverage  
5. Hard code ground truth  
   1. Add comments explaining why the ground truth makes sense 
6. Consider the reader  
   1. Write clear, easily understandable tests 
   2. Prioritize quick comprehension  
   3. Use descriptive test names and concise setups 


### Error Handling, Warnings, Exception Checks, and Logging:  

1. Reuse utility functions for exception handling (see src/utils.py for examples) 
2. Error messages should include enough context on what is expected, and what has been entered. All the values entered by the user should be surrounded by double quotes .  
3. Warning and error message format (use "") to provide context. 
4. Make sure to log error messages. 


### Figures 

1. Include axis names and titles for all figures 
2. Use programmatic labeling:  
   1. Fetch data labels dynamically 
   2. Avoid hard-coded titles and axis names 
   3. Example: For UMAP plots, reflect coordinates in labels 

 
