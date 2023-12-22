# Gender Roles in Cinema: An Ongoing Evolution (https://juliette75700.github.io/Gender-Evolution/)

## Abstract

We have all grown up watching movies that we love. Despite the hours of joy cinema brings into our lives there is no doubt that it is a mirror of the society we live in. Perhaps we have all spent hours empathising for "The Underdog"" or laughing with "The Fool", but have we ever stopped and wondered how our favorite tropes and movies perpetuate societal biases? In this project we aim to do exactly that. Using a range of datasets we set out to analyse the evolution of gender bias in the film industry. In particular, we aim to answer the following research questions:

1. What role do gendered tropes play in the film industry? Is there a different between the tropes found in movies with a higher number of female than males and vice versa?
2. Are actors' careers affected by their gender and to what extent? Can we predict an actor's gender based on their career longevity?
3. If there is a disparity, is it justified? Are film ratings affected by the more prominent presence of female or actors in a movie?

To this end, we utilised a variety of datasets and methods. In particular, the scoring method proposed in the paper [AnalyzingGenderBias](https://aclanthology.org/2020.nlpcss-1.23/) by Gala et al. was used to assess the 'genderedness' of each trope. Furthermore, a survival analysis was carried out to assess the disparity between career longevity of different genders while a neural network and more 'naive' machine learning methods were used to evaluate whether an actor's gender can be predicted based on their career longevity. Finally, a causal analysis using propensity score matching to eliminate confounders was used to assess the relationship between movie ratings and gender representation in movies.

Our analysis showed that while there is a statistically significant difference between the career longevity of female and male actors, this alone is not enough to predict their gender with high precision. Additionally, it was found that although tropes that are labelled as 'male' or 'female' do perpetuate gender stereotypes, their use in movies based on the naive scoring method used is not extensive enough to reach significant conclusions. Finally, it is revealed that female representation in movies does not affect their ratings.

In conlusion, although our analysis did not reach widespread and robust conclusions we hope that it sets a stepping stone for future works. In particular, we believe that our tropes analysis can be expanded via the use of more robust models following a more refined classification such that specific tropes can be linked to specific characters. This could lead to the use of tropes to establish more granular correlations with actor career longevity and movie ratings.

To find our interactive blog for this project you can visit: https://juliette75700.github.io/Gender-Evolution/.

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
- `IMDB_movies.tsv`: contains the names and the ID's of the movies from IMBD in order to merge with the ratings
- `IMDB_ratings.tsv`: contains the the ID's and the ratings of the movies from IMBD in order to contact the causal analysis

## Organization Within the Team

- Alexandre Ben Ahmed Kontouli:
    - Milestone 2:
        - The evolution of M/F ratio in genres over time
        - The evolution of the number of movies per actor/actress over career lifespan
    - Milestone 3:
        - Survival analysis
        - Data pre-processing for trope analysis
        - Prediction of gender with Neural Networks
        - Website authoring + website layout and design

- Aristotelis Dimitriou:
    - Milestone 2:
        - Missing value analysis of given dataframes
        - Gender distribution across genres and time
        - Role assessment of characters based on plot summaries
        - Genre reduction
    - Milestone 3:
        - Survival Analysis
        - Prediction of gender with Neural Networks
        - Website authoring + website layout and design

- Juliette Dutheil:
    - Milestone 2:
        - Gender Repartition over actor set, countries, and number of actors per movie
        - Distribution of actor age per genre at movie release
    - Milestone 3
        - Jekyll website set up and hosting using Github Pages
        - Front end development and work on the structure for the visual aspects  
        - Partipation to the causal analysis with Stavros

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
        - Merging of all notebooks into final deliverable notebook

- Stavros Papaiakovou:
    - Milestone 2:
        - Naive analysis of gender repartition distribution across numerical columns of movie data.
        - Pearson and spearman correlations of aforementioned analysis and linear regression of the evolution of number of actor per gender over the years (with log transformation)
    - Milestone 3:
        - Creation of logistic regression and random forest models to predict actors' gender based on career longevity
        - Causal analysis to assess the relationship between ratings and female representation
        - Write up of the causal analysis for the website
