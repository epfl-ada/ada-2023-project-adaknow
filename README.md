# Project Title: Gender Disparity Through Time, Genres, and TV Tropes

## Abstract

(150 words)
[Provide a concise description of the project, its motivation, focus on gender disparity, specific genres, TV tropes, and the intended insights or story.]

## Research Questions

[List the specific questions the project aims to address.]

1. Question 1
2. Question 2
3. ...

## Datasets

All datasets need to be included in the directory `data/`

### Baseline Datasets

- `movie.metadata.tsv`:
- `character.metadata.tsv`:
- `plot_summaries.txt`:
- `name.clusters`:
- `tvtropes.clusters`:

### Additional Datasets

[Detail any additional datasets used, their size, format, and relevance to the primary data.]

- Dataset 1: tvtropes (<https://tvtropes.org/pmwiki/pmwiki.php/Main/UnisexTropes>)
- Dataset 2: bechdel.csv (<https://www.kaggle.com/datasets/treelunar/bechdel-test-movies-as-of-feb-28-2023>)
- ...

## Methods

### Exploratory Data Analysis (EDA)

[Describe the EDA. This might include initial data visualization, identification of patterns, anomalies, or preliminary insights about gender disparities in your datasets.]

#### Univariate Analysis

#### Bivariate Analysis (focused on gender)

### Data Preprocessing

[Outline the preprocessing steps needed for your datasets. This could involve cleaning data, handling missing values, normalizing or transforming data, feature selection, etc.]

- Remove all instances of from the `character.metadata.tsv` that are missing the gender of the actor/caracter

- Added a caracter `Role` attribute to the `character.metadata.tsv` that is either Main Character (MC) or Supporting Character (SC). This was done following a naive approach, looking wether the name or surname of a character is present in the plot summary of the movie in the `plot_summaries.txt` file. If the name or surname of the character is present in the plot summary, then the character is considered a Main Character, otherwise it is considered a Supporting Character.

- Parsed the json-like structures given in the `Language`, `Country` and `Genre` attributes of the `movie.metadata.tsv` file. The parsing transformed the json structures into a list of strings, where each string is a language, country or genre of the movie.

- Reduced the number of genres from 363 down to 78, by mapping each genre to a set of more general genres (e.g. `Action/Adventure` -> `Action` and `Adventure`). The mapping was done manually, by looking at the genres and trying to find the most general genres that could be used to describe the movie. Note that it was attempted to perform the mapping through a hierarchical agglomerative clustering, but the results were not satisfactory.

### Analysis Techniques

[Discuss the analytical methods and techniques we plan to use, with essential mathematical details.]

## Proposed Timeline

- Milestone 1: Date and Objectives
- Milestone 2: Date and Objectives
- ...

## Organization Within the Team

[List roles and responsibilities of each team member, including internal milestones.]

- Alexandre Ben Ahmed Kontouli: Responsibilities
- Aristotelis Dimitriou: Responsibilities
- Juliette Dutheil: Responsibilities
- Maria Eleni Peponi: Responsibilities
- Stavros Papaiakovou: Responsibilities

## Questions for TAs (Optional)

## [Notebook Name]

[Describe the contents of the Jupyter Notebook, including initial analyses, EDA, and data handling pipelines.]
