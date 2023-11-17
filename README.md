# Project Title: Evolution of gender roles in film across the decades

## Abstract

In this milestone we aim to assess the feasibility of the 'Evolution of gender roles in film across the decades'. Ultimately, we have limited the goal of the project to the analysis of gender representation and gender stereotypes utilising time, movie genres, film tropes and actor attributed as covariates. 

In order to establish a solid framework and ensure the feasibility of our goal, we have preprocessed, and performed an exploratory data analysis on the datasets provided. Furthermore, an additional dataset provided in the paper [AnalyzingGenderBias](https://aclanthology.org/2020.nlpcss-1.23/) by Gala et al. as well as their proposed scoring method was used to assess the 'genderedness' of each trope. The model achieves an accuracy of 80% and 73% in predicting female and male tropes respectively, and a Cohen's kappa of 0.43 when compared to expected classifications.

The results of the analysis are promising and we are confident that the project can be completed within the given time frame.

## Research Questions

1. How are highly gendered tropes used in movies across different genres and decades? 
2. How does the nature of tropes correlate with their genderedness?
3. What covariates affect the representation of males and females in movies across time?

## Proposed Additional Datasets

- Dataset 1: tvtropes (<https://tvtropes.org/pmwiki/pmwiki.php/Main/UnisexTropes>)
- Dataset 2: bechdel.csv (<https://www.kaggle.com/datasets/treelunar/bechdel-test-movies-as-of-feb-28-2023>)
- Dataset 3: film_tropes
- Dataset 4: 

## Methods

### Exploratory Data Analysis (EDA)

- Histograms and box plots of genderedness score distributions.

- Cohen's kappa is used to assess the statistical significance of the agreement between the genderedness model and the expected results.

- Temporal scatter plots of gender counts per movie genre.

- Confidence interval analysis and overlapping interval identification on gender representation per movie genre.

- Missing value analysis for the `character_metadata`, `movie_metadata` and `film_tropes` datasets.

- Genre frequency distribution and analysis of male to female ratio per genre.

- Distribution of movies over the lifetime of an actor for the top 10 actors and actresses. 

- 

#### Univariate Analysis

#### Bivariate Analysis (focused on gender)

### Data Preprocessing

[Outline the preprocessing steps needed for your datasets. This could involve cleaning data, handling missing values, normalizing or transforming data, feature selection, etc.]

- Remove all instances of from the `character.metadata.tsv` that are missing the gender of the actor/caracter

- Added a caracter `Role` attribute to the `character.metadata.tsv` that is either Main Character (MC) or Supporting Character (SC). This was done following a naive approach, looking wether the name or surname of a character is present in the plot summary of the movie in the `plot_summaries.txt` file. If the name or surname of the character is present in the plot summary, then the character is considered a Main Character, otherwise it is considered a Supporting Character.

- Parsed the json-like structures given in the `Language`, `Country` and `Genre` attributes of the `movie.metadata.tsv` file. The parsing transformed the json structures into a list of strings, where each string is a language, country or genre of the movie.

- Reduced the number of genres from 363 down to 78, by mapping each genre to a set of more general genres (e.g. `Action/Adventure` -> `Action` and `Adventure`). The mapping was done manually, by looking at the genres and trying to find the most general genres that could be used to describe the movie. Note that it was attempted to perform the mapping through a hierarchical agglomerative clustering, but the results were not satisfactory.

- Preprocessing of `film_tropes.csv` on axes of interest:
    - The 'Trope' and 'Example' columns are searched for missing values. As there are no missing values in the 'Trope' column, the rows with missing values in the 'Example' column are filled with a dummy value considering that even though an example of a trope may be Nan in a particular movie, it may be present in other movies.
    - The 'Trope' and 'Movie Title' columns are explored for duplicate values. These are also checked against the 'title_id' and 'trope_id' columns respectively to ensure their uniqueness. 
    - Having ensured their uniqueness, the dataframe is grouped by 'Trope' and the examples are aggegated into a cell. 

- Tokenisation and Lemmatisation: 
    - The examples provided for each tropes are tokenised and lemmatised using the `NLTK` library to obtain a list of words associated with each trope. This process is also undertaken for the `female_word_file.txt` and `male_word_file.txt` for consistency.
    - A new column with the processed examples is added to the dataframe containing the list of tokenised, and lemmatised words related to each trope. 
    - It is again verified that there are no tropes with no processed examples. 
    
- Quantification of 'genderness':
    - The counts of gendered words in the processed example tokens of each trope are computed using the gendered words in `female_word_file.txt` and `male_word_file.txt`. The 'MaleCount' and 'FemaleCount' column act as features for the 'Genderedness Score'.
    - A 'Genderedness Score' between [-1, 1] is assigned to each trope with [-1, 0] values signifying dominance of male characteristics and scores between [0, 1] signifying femininity.  
    - The result is exported in the `tropes_wscores.csv` file.



### Analysis Techniques

[Discuss the analytical methods and techniques we plan to use, with essential mathematical details.]
- Computation of 'Genderedness':
  - Gala et al.'s method as well as their `film_tropes.csv` dataset was employed to compute genderedness scores for each trope.
  - The distribution of genderedness scores was visualized using histograms.
  - The performance of the model was assessed using standard metrics as well as simple statistical tests.

- Trope genderedness model uncertainty and verification of performance:
    - The number of female and male related tropes is computed. The results are visualised with histograms verifying the skewness towards male tropes. 
    - Lists of 'Always Female', 'Always Male', and 'Unisex' tropes are extracted from TvTropes. These are converted to the CamelCase format used in the `tropes_wscores` dataframe. They are further assigned a score of 1 when 'Always Female', -1 when 'Always Male' and 2 when 'Unisex'.
    - The 'Genderedness' column of `tropes_wscores` is discretised considering the uncertainty of the model. The following thresholds are used:
        - anything above the median for female characterised tropes (i.e. [0,1]) is clearly female and assigned a score of 1
        - anything below the median for male characterised tropes (i.e. [-1,0]) is clearly male and assigned a score of -1
        - anything in between is unisex and assigned a score of 2
    - The `tropes_wscores` dataframe is merged with the processed 'Always Female', 'Always Male', and 'Unisex' dataframes for comparison.
    - The accuracy, precision, and recall of the model are computed by comparing its predictions to the 'Always Female', 'Always Male', and 'Unisex' expectations. 


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
