# Project Title: Gender Roles in Cinema: An Ongoing Evolution

## Abstract

We have all grown up watching movies that we love. Despite the hours of joy cinema brings into our lives there is no doubt that it is a mirror of the society we live in. Perhaps we have all spent hours empathising for "The Underdog"" or laughing with "The Fool" but have we ever stopped and wondered how our favorite tropes and movies perpetuate societal biases? In this project we aim to do exactly that; using a range of datasets we set out to analyse the evolution of gender bias in the film industry. In particular we aim to answer the following research questions:

1. What role do gendered tropes play in the film industry? Is there a different between the tropes found in movies with a higher number of female than males and vice versa?
2. Are actors' careers affected by their gender and to what extent? Can we predict an actor's gender based on their career longevity?
3. If there is a disparity, is it justified? Are film ratings affected by the more prominent presence of female or actors in a movie?

To this end we utilised a variety of datasets and methods. In particular, the scoring method proposed in the paper [AnalyzingGenderBias](https://aclanthology.org/2020.nlpcss-1.23/) by Gala et al. was used to assess the 'genderedness' of each trope. Furthermore, a survival analysis was carried out to assess the disparity between career longevity of different genders while a neural network and more 'naive' machine learning methods were used to evaluate whether an actor's gender can be predicted based on their career longevity. Finally, a causal analysis using propensity score matching to eliminate confounders was used to assess the relationship between movie ratings and gender representation in movies.

Our analysis showed that while there is a statistically significant difference between the career longevity of female and male actors, this alone is not enough to predict their gender with high precision. Additionally, it was found that although tropes that are labelled as 'male' or 'female' do perpetuate gender stereotypes, their use in movies based on the naive scoring method used is not extensive enough to reach significant conclusions. Finally, it is revealed that female representation in movies does not affect their ratings.

In conlcusion, although our analysis did not reach widespread and robust conclusions we hope that it sets a stepping stone for future works. In particular we believe that our tropes analysis can be expanded via the use of more robust models following a more refined classification such that specific tropes can be linked to specific characters. This could lead to the use of tropes to establish more granular correlations with actor career longevity and movie ratings.

## Datasets

To import the datasets, run the following command (for MacOS and Linux):

```bash
bash get_external_datasets.sh
```

or (for Windows): use this [link](https://drive.google.com/uc?export=download&id=19C4MvZ6JAMHAnnBZUyS0xSRo8Q9NEU9w) and place the data in the `\data` directory.

**Baseline Datasets**:

- `movie.metadata.tsv` : used to extract core information about movies such as genre, release date, title, etc.
- `character.metadata.tsv`: used to extract the number of actors of a particular gender in a movie, as well as crucial information for the survival and causal analyses such as actor date of birth, last movie, etc.

**Additional Datasets**:

- `film_tropes.csv`: used to extract tropes commonly used in movies and their examples for the assignment of genderedness scores. Movie names are used to match the tropes to the movies in the baseline datasets.
- `female_word_file.txt`: contains words characterised as female. This was used in the assignment of genderedness scores to tropes. 
- `male_word_file.txt`: contains words characterised as male. This was used in the assignment of genderedness scores to tropes.
- `female_tvtropes.csv`: contains tropes characterised as 'Always Female' by the TVTropes community. This was used to assess the performance of the genderedness scoring method.
- `male_tvtropes.csv`: as above.
- `unisex_tvtropes.csv` as above.

## Methods

### Exploratory Data Analysis (EDA)

The EDA visualisations include:

- Histograms and box plots of genderedness score distributions.

- Cohen's kappa is used to assess the statistical significance of the agreement between the genderedness model and the expected results.

- Temporal scatter plots of gender counts per movie genre.

- Confidence interval analysis and overlapping interval identification on gender representation per movie genre.

- Missing value analysis for the `character_metadata`, `movie_metadata` and `film_tropes` datasets.

- Genre frequency distribution and analysis of male to female ratio per genre.

- Distribution of age for the number of movies of an actor for the top 10 actors and actresses.

- Distribution of ages for male and female actors at the time of a movie's release.

- Distribution of ages for actors.

- Log regression plot of the time evolution of the number of female and male actors.

### Data Preprocessing

The following preprocessing steps were performed on the datasets:

- Remove all instances of from the `character.metadata.tsv` that are missing the gender of the actor/caracter

- Added a caracter `Role` attribute to the `character.metadata.tsv` that is either Main Character (MC) or Supporting Character (SC).

- Parsed the json-like structures given in the `Language`, `Country` and `Genre` attributes of the `movie.metadata.tsv` file. The parsing transformed the json structures into a list of strings, where each string is a language, country or genre of the movie.

- Reduced the number of genres from 363 down to 78, by manually mapping each genre to a set of more general genres (e.g. `Action/Adventure` -> `Action` and `Adventure`) following unsatisfactory results with a hierarchical agglomerative clustering approach.

- Processed film_tropes.csv: filled missing 'Example' values, ensured 'Trope' and 'Movie Title' uniqueness, and aggregated examples by trope.

- Tokenized and lemmatized trope examples using NLTK; added a column for processed words and confirmed no missing processed examples.

- Developed a 'Genderedness Score' from gendered word counts in trope examples, ranging from [-1, 0](male) to [0, 1] (female), and recorded in `tropes_wscores.csv`.

### Analysis Techniques : Milestone 2

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
- 'Role' assessment:
  - This was done following a naive approach, looking wether the name or surname of a character is present in the plot summary of the movie in the `plot_summaries.txt` file. If the name or surname of the character is present in the plot summary, then the character is considered a Main Character, otherwise it is considered a Supporting Character.

### Analysis Techniques : Milestone 3

- Genderedness of Tropes:
    - Using the expected tropes' genderedness scores, threholds for what is considered male, female, or unisex were set by maximizing the recall of the model.
    - A Latent Dirichlet Allocation (LDA) model was used to cluster the tropes into topics within each gender category and the results were visualised
    - An analysis assessing the statistical significant of the relationship between representation and frequency of gendered tropes was performed.

- Survival Analysis and Gender Prediction:

- Causal Analysis:
    - Naive analysis of different movie characteristics in order to found possible confounders of the movie ratings to then be able to isolate the desired one (female fraction of actors).
    - Propensity score matching was used to create a balanced dataframe between the treatment and control group. The treatment group were movies with more female actors than male actors, whereas the control group was the opposite. 
    - Analysis performed on the balanced dataframe to show the real relationship between female fraction of actors and the ratings.


## Organization Within the Team

- Alexandre Ben Ahmed Kontouli:
    - Milestone 2:
        - The evolution of M/F ratio in genres over time
        - The evolution of the number of movies per actor/actress over career lifespan
    - Milestone 3:

- Aristotelis Dimitriou:
    - Milestone 2:
        - Missing value analysis of given dataframes
        - Gender distribution across genres and time
        - Role assessment of characters based on plot summaries
        - Genre reduction
    - Milestone 3:

- Juliette Dutheil:
    - Milestone 2:
        - Gender Repartition over actor set, countries, and number of actors per movie
        - Distribution of actor age per genre at movie release
    - Milestone 3
        - Set up of website 

- Maria Eleni Peponi:
    - Milestone 2:
        - Female and male count time evolution per genre and confidence interval analysis
        - Trope processing and scoring method as proposed by Gala et al. 
        - Naive model performance assessment (fixed thresholds)
    - Milestone 3:
        - Model performance assessment and recall maximisation strategy
        - LDA topic modelling of tropes and visualisations
        - Statistical analysis of gendered tropes
        - Tropes write up for the website
        - README write up

- Stavros Papaiakovou:
    - Milestone 2:
        - Naive analysis of gender repartition distribution across numerical columns of movie data.
        - Pearson and spearman correlations of aforementioned analysis and linear regression of the evolution of number of actor per gender over the years (with log transformation)
    - Milestone 3:
        - Creation of logistic regression and random forest models to predict actors' gender based on career longevity
        - Causal analysis to assess the relationship between ratings and female representation
        - Write up of the causal analysis for the website