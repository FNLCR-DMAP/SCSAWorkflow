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

Ready to contribute? Follow up the guide in this [README](README.md) to install
SPAC locally.

1. Use `git` (or similar) to create a branch for local development and make your changes:

    ```console
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

2. When you're done making changes, check that your changes conform to any code formatting requirements and pass any tests.

3. Commit your changes and open a pull request.

## Commit Message Guidelines for Automatic Version Bumping

To write commit messages for automatic version bumping, follow these guidelines, based on **Conventional Commits**. This structure is often used with tools like **Semantic Release** to automatically determine version changes (e.g., `1.0.0` → `1.1.0` or `2.0.0`).

## Commit Message Structure

Use the format:

`<type>(scope): <short description>`

- **`type`**: The nature of the change, which determines the version bump.
- **`scope`**: The section of the codebase affected (optional but recommended).
- **`description`**: A brief explanation of the change.

## Common Commit Types and Version Impacts

1. **Major (`BREAKING CHANGE`)**:
   - **Purpose**: Introduces incompatible API changes.
   - **Example**:
     ```
     feat(core): add new API endpoint for user data retrieval

     BREAKING CHANGE: Updated endpoint requires authentication token.
     ```

2. **Minor (`feat`)**:
   - **Purpose**: Adds a new feature that is backward-compatible.
   - **Example**:
     ```
     feat(auth): add OAuth support for Google login
     ```

3. **Patch (`fix`)**:
   - **Purpose**: Fixes a bug that is backward-compatible.
   - **Example**:
     ```
     fix(ui): correct alignment of login button on mobile devices
     ```

4. **Other Commit Types**:
   - **`docs`**: Documentation updates only.
     ```
     docs(README): add usage instructions for authentication API
     ```
   - **`style`**: Code style updates (e.g., whitespace, formatting).
     ```
     style(header): adjust spacing in navigation bar
     ```
   - **`refactor`**: Code refactoring without new features or bug fixes.
     ```
     refactor(auth): simplify OAuth flow for better readability
     ```
   - **`test`**: Adding or updating tests.
     ```
     test(auth): add tests for Google login flow
     ```
   - **`chore`**: Miscellaneous tasks (e.g., updating dependencies).
     ```
     chore(deps): update express to v4.17.1
     ```

## Tips for Effective Commit Messages

- Keep descriptions under 50 characters, ideally.
- Use present-tense and imperative form (e.g., "add", not "added").
- Add a `BREAKING CHANGE:` line in the message body if the commit has breaking changes. This ensures a major version bump.


## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include additional tests if appropriate.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for all currently supported operating systems and versions of Python.


## SPAC Terminology

SPAC uses general terminology to simplify technical terms from the AnnData object for less technical users. Here is a quick guide:

* **Cells:** Rows in the `X` matrix of AnnData.
* **Features:** Columns in the `X` matrix of AnnData, representing gene expression or antibody intensity.
* **Tables:** Originally called **layers** in AnnData, these represent transformed features.
* **Associated Tables:** Corresponds to `.obsm` in AnnData and can store spatial coordinates, UMAP embeddings, etc.
* **Annotation:** Corresponds to `.obs` in AnnData and can store cell phenotypes, experiment names, slide IDs, etc.


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

 
