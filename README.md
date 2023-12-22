# Project Title: Evolution of gender roles in film across the decades

## Abstract

In this milestone we aim to assess the feasibility of the 'Evolution of gender roles in film across the decades'. Ultimately, we have limited the goal of the project to the analysis of gender representation and gender stereotypes utilising time, movie genres, film tropes and actor attributed as covariates.

In order to establish a solid framework and ensure the feasibility of our goal, we have preprocessed, and performed an exploratory data analysis on the datasets provided. Furthermore, an additional dataset provided in the paper [AnalyzingGenderBias](https://aclanthology.org/2020.nlpcss-1.23/) by Gala et al. as well as their proposed scoring method was used to assess the 'genderedness' of each trope. The model achieves an accuracy of 80% and 73% in predicting female and male tropes respectively, and a Cohen's kappa ([CohensKappaWiki](https://en.wikipedia.org/wiki/Cohen%27s_kappa)) of 0.43 when compared to expected classifications.

The results of the analysis are promising and we are confident that the project can be completed within the given time frame.

## Research Questions

1. How are highly gendered tropes used in movies across different genres and decades?
2. How does the nature of tropes correlate with their genderedness?
3. What covariates affect the representation of males and females in movies across time? And can we use those covariates to accurately infer the gender of actors?
4. How much gender disparity is there in cinema across time and genres?

## Datasets

To import the datasets, run the following command (for MacOS and Linux):

```bash
bash get_external_datasets.sh
```

or (for Windows): use this [link](https://drive.google.com/uc?export=download&id=19C4MvZ6JAMHAnnBZUyS0xSRo8Q9NEU9w) and place the data in the `\data` directory.

**Baseline Datasets**:

- `movie.metadata.tsv`
- `character.metadata.tsv`
- `plot_summaries.txt`

**Additional Datasets**:

- `film_tropes.csv`
- `female_word_file.txt`
- `male_word_file.txt`
- `female_tvtropes.csv`
- `male_tvtropes.csv`
- `unisex_tvtropes.csv`

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

### Analysis Techniques

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

## Further analysis
An question we asked ourselves when treating all this data was whether we could accurately guess the gender of actors using some carefully selected covariates. So we preprocessed the data in order to be left with the biggest subset of data without ill-presented data (missing values or irrelevant data).

The following tasks remain to be completed:

- Examining the distribution of gendered tropes across film genres and time periods.
- Assessing the representation of genders and the prevalence of gendered tropes in different genres.
- Drawing conclusions on trope usage trends over time and across genres.
- Undertaking a survival analysis, predicting the longevity of active actors' careers.

## Proposed Timeline

The following tasks are to be undertaken sequentially over the course of the project:

1. Merging of the processed tropes dataset with the `movie.metadata.tsv` and `character.metadata.tsv` datasets.

2. Exploring the longevity of actors' careers through survival analysis.

3. Tailor the data to fit our fututure endeavours, such as the survival analysis, the analysis of the relation btween gendered tropes and other covariates, etc.

4. Find the potential evolution of the career of a given actor/actress such as the genres they are most prone to play in.

5. Investigate whether the use of higly stereotipically gendered tropes, based on our metric, has changed over time e.g. whether tropes have higher variability for each gender than before.

## Organization Within the Team

- Alexandre Ben Ahmed Kontouli:
  - The evolution of M/F ratio in genres over time
  - The evolution of the number of movies per actor/actress over career lifespan
  - Data pre-processing for trope analysis
  - Prediction of gender with Neural Networks
  - Website authoring
- Aristotelis Dimitriou:
  - Missing value analysis of given dataframes
  - Gender distribution across genres and time
  - Role assessment of characters based on plot summaries
  - Genre reduction
  - Prediction of gender with Neural Networks
  - Authoring website content
- Juliette Dutheil:
  - Naive analysis of gender repartition: Gender Repartition over actor set, over the movies by countries, over the number of actors per movie
  - Distribution of actor age per genre at movie release
- Maria Eleni Peponi:
  - Trope processing, scoring and model uncertainty and performance assessment
  - Female and male count time evolution per genre and confidence interval analysis
  - Trope analysis
  - Authorng website content
- Stavros Papaiakovou:
  - Naive analysis of gender repartition: Distribution of gender repartition over the age of the actors, movie runtime, year of movie release, month of movie release.
  - Pearson and spearman correlations of aforementioned analysis and linear regression of the evolution of number of actor per gender over the years (with log transformation)

## Questions for TAs (Optional)

## Best Notebook Ever

Pipeline:

1. Import data
2. Missing value analysis
3. Preprocessing
4. Exploratory Data Analysis
5. Tropes preprocessing
6. Genderedness model
