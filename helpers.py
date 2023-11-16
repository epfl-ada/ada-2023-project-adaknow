import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ks_2samp, t, pointbiserialr
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from re import sub

def parse_encoded_col(encoded_str):
    try:
        return ast.literal_eval(encoded_str)
    except (ValueError, SyntaxError):
        return {}
    
def string_to_list(list_string):
    try:
        # This safely evaluates a string as a list
        return ast.literal_eval(list_string)
    except ValueError:
        # In case of error (e.g., empty strings), return an empty list
        return []
    

def map_genres(old_genres_list, mapping_dict):
    new_genres_list = []
    for genre in old_genres_list:
        # Get the new genres from the dictionary, if not found or None, it will return an empty list
        mapped = mapping_dict.get(genre, [])
        if mapped is not None:
            new_genres_list.extend(mapped)
    # Return the unique genres after mapping
    return list(set(new_genres_list))

def categorize_character(row, plot_summaries):
    """
    Categorize a character as Main Character (MC) or Secondary Character (SC) based on
    whether any part of the character's name appears in the plot summary.
    """
    # Extract movie ID and character name from the row
    movie_id = row['Wikipedia movie ID']
    character_name = row['Character name']

    # Check if character name is a string and split it into parts (first and last names)
    if isinstance(character_name, str):
        name_parts = character_name.split()

        # Retrieve the plot summary for the corresponding movie ID
        plot_summary = plot_summaries[plot_summaries['Wikipedia movie ID'] == movie_id]['Plot summary'].values
        if len(plot_summary) > 0:
            plot_summary = plot_summary[0]
            # Check if any part of the character name appears in the plot summary
            for part in name_parts:
                if part in plot_summary:
                    return "MC"  # Main Character
                else:
                    return "SC"  # Supporting Character
    else:
        return np.nan


def count_females(column):
    """Function to count number of females in a column """
    f_count = column.value_counts()['F']
    return f_count

# function to count number of males
def count_males(column):
    """Function to count number of males in a column """
    m_count = column.value_counts()['M']
    return m_count

# 
def plot_genres(data, start_idx, end_idx, rows = 5, cols = 6):
    '''Function to create plot for subset of genres because we have 77 of them '''
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(20,20), sharex=True)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    for (idx, row), ax in zip(data.iloc[start_idx:end_idx].iterrows(), axs.flatten()):
        ax.scatter(row['Release_date_list'], row['F_count_list'], color='#16A085', label='Female', marker = '^')
        ax.scatter(row['Release_date_list'], row['M_count_list'], color='#F1C40F', label='Male', marker = '^')
        ax.set_title(row['Genres Reduced'], fontsize=10)
        ax.set_xlabel('Year', fontsize=8)
        ax.set_ylabel('Gender count', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.show()

def mean_confidence_interval(data, confidence=0.95):
    """Compute confidence interval for given data."""
    n = len(data)
    m, se = np.mean(data), np.std(data)
    h = se * t.ppf((1 + confidence) / 2., n-1)/np.sqrt(n)
    return [m, m-h, m+h] # new dataframe

# Now check for overlapping intervals
def check_cioverlap(df, column_1 = 'M_CI-', column_2 = 'F_CI+', 
                    column_3 = 'F_CI-', column_4 = 'M_CI+', f_mean = 'F_mean', m_mean = 'M_mean'):
    '''Function that checks for overlapping confidence intervals and returns a binary indicator'''  
    overlap_indicator = []
    for index, row in df.iterrows():
        if row[f_mean] < row[m_mean] and row[column_1] < row[column_2]:
            overlap_indicator.append(1) 
        elif row[m_mean] < row[f_mean] and row[column_3] < row[column_4]:
            overlap_indicator.append(1) 
        else:
            overlap_indicator.append(0) 
    return pd.Series(overlap_indicator)

# Make function to create plot for subset of genres because we have 77 of them
def ci_plot_genres(data, start_idx, end_idx, n_rows=6, n_cols=5):
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20,20))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    for (idx, row), ax in zip(data.iloc[start_idx:end_idx].iterrows(), axs.flatten()):
        female_x, male_x = 0.1, 0.9
        ax.errorbar(x=female_x , y=row['F_mean'],
             yerr=[[row['F_mean'] - row['F_CI-']], [row['F_CI+'] - row['F_mean']]],
             fmt='none', capsize=5, color='#FFB139', label='Female')
        ax.errorbar(x=male_x , y=row['M_mean'],
             yerr=[[row['M_mean'] - row['M_CI-']], [row['M_CI+'] - row['M_mean']]],
             fmt='none', capsize=5, color='#16A085', label='Male')
        ax.scatter(x=female_x, y=row['F_mean'], color='#FFB139', zorder=3)
        ax.scatter(x=male_x , y=row['M_mean'], color='#16A085', zorder=3)

        # Fill area for overlapping CIs
        lower_fill = max(row['F_CI-'], row['M_CI-'])
        upper_fill = min(row['F_CI+'], row['M_CI+'])
        if lower_fill < upper_fill:  # Check for overlap
            ax.fill_between([female_x, male_x], lower_fill, upper_fill, color='#C0392B', alpha=0.2)
        
        ax.set_title(row['Genres Reduced'], fontsize=10)
        ax.set_xticks([female_x, male_x])
        ax.set_xticklabels(['Female', 'Male'])
        ax.set_xlabel('Gender', fontsize=8)
        ax.set_ylabel('Gender count with 95% CI', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.show()

def txt_to_list(text_file):
    """Function to convert text file with different words in each line to list"""
    file = text_file.read()
    text_to_list = file.split("\n") 
    return text_to_list

def preprocess_and_lemmatize(text):
    """Tokenizes and lemmatizes text
    Note: we don't remove stopwords bc it removes pronouns which we need for genderedness
    """
    lemmatizer = WordNetLemmatizer()
    # Convert to lowercase, tokenize, and remove punctuation
    tokens = word_tokenize(text.lower())

    # Only keep characters with alphabet letters
    tokens = [t for t in tokens if t.isalpha()]
    
    # Lemmatize tokens
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    return lemmas # list of lemmatised tokens

def count_tokens(token_list, lexicon):
    """Function that counts number of gendered word occurences"""
    gendered_token = []
    for token in token_list: # for every word in the list 
        if token in lexicon: # if the word also exists in the gendered lexicon
            gendered_token.append(token) # append it
    return len(gendered_token) # The length of this list is the number of gendered occurences in our token list

def camel_case(s):
    """Function that converts a string to camel case (LikeThis)"""

    # remove punctuation
    s = sub(r"(_|-)+", " ", s).title().replace(" ", "")

    # remove ;
    s = s.replace(';', '')
    
    # Join the string
    return ''.join([s[0], s[1:]])


def count_q(df, base_list):
    '''Function to count number of tropes in each quartile '''
    my_list = list(df)
    count_q = 0
    for i in base_list:
        for j in my_list:
            if i == j :
                count_q += 1
    return count_q