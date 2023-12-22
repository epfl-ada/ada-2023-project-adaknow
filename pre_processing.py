import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



def map_genre_to_category(genre):
    categories = {
        'Drama and Literary': ['Drama', 'Literary', 'Art Film', 'Biography', 'Black Cinema', 'European', 'World Cinema', 'Independent', 'Star Vehicle', 'Ensemble', 'Social', 'Black and White'],
        'Comedy and Romance': ['Comedy', 'Romance', 'Family', 'Teen'],
        'Action': ['Action', 'Adventure', 'Martial Arts', 'War', 'Cars', 'Epic', 'Low Budget'],
        'Horror': ['Horror', 'Thriller', 'Psychological', 'Dystopian', 'Prison', 'Thrillers'],
        'Sci-fi and fantasy': ['Science Fiction', 'Fantasy', 'Futuristic'],
        'Animation and Kids': ['Animation', 'Kids', 'Animated', 'Stop Motion'],
        'Historical': ['History', 'Religious', 'Chinese', 'Japanese', 'Asian', 'Indian', 'Filipino', 'USA', 'European', 'Black Cinema'],
        'Niche': ['LGBTQ', 'LBGTQ', 'Avant-Garde', 'Film Adaptation', 'Silent', 'Short Film', 'Exploitation', 'Adult', 'Nature', 'Women', 'Propaganda', 'Educational'],
        'Documentary': ['Documentary', 'Nature', 'Educational', 'Television'],
        'Crime and Mystery': ['Crime', 'Mystery', 'Political', 'Anthology'],
        'Music and Dance': ['Music/Dance'],
        'Sports': ['Sports']
    }
    for category, genres in categories.items():
        if genre in genres:
            return category
    return 'Other'  # for genres not listed

def map_country_to_region(country):
    regions = {
        'North America': ['United States of America', 'Canada', 'Mexico', 'Bahamas', 'Puerto Rico'],
        'South America': ['Brazil', 'Argentina', 'Chile', 'Uruguay', 'Peru', 'Colombia', 'Venezuela', 'Costa Rica'],
        'Europe': ['France', 'United Kingdom', 'Finland', 'Sweden', 'Hungary', 'Poland', 'Netherlands', 'Denmark', 'German Democratic Republic', 'West Germany', 'Switzerland', 'Italy', 'Czech Republic', 'Spain', 'Ireland', 'Norway', 'Soviet Union', 'England', 'Czechoslovakia', 'Romania', 'Russia', 'Albania', 'Belgium', 'Germany', 'Serbia and Montenegro', 'Austria', 'Lithuania', 'Azerbaijan', 'Bulgaria', 'Serbia', 'Bosnia and Herzegovina', 'Estonia', 'Iceland', 'Croatia', 'Greece', 'Yugoslavia', 'Portugal', 'Luxembourg', 'Kingdom of Great Britain', 'Republic of Macedonia', 'Northern Ireland', 'Ukraine', 'Slovakia', 'Slovenia', 'Isle of Man', 'Slovak Republic', 'Wales', 'Scotland', 'Socialist Federal Republic of Yugoslavia', 'Weimar Republic', 'Nazi Germany'],
        'Asia': ['India', 'China', 'Japan', 'Taiwan', 'Philippines', 'South Korea', 'Cambodia', 'Hong Kong', 'Bangladesh', 'Singapore', 'Nepal', 'Indonesia', 'Malaysia', 'Thailand', 'Vietnam', 'Mongolia', 'Korea', 'Pakistan', 'Armenia', 'Burma', 'Georgia'],
        'Middle East': ['Iran', 'Lebanon', 'Tunisia','Algeria', 'Egypt', 'Turkey', 'Mandatory Palestine', 'Israel', 'Iraq', 'United Arab Emirates', 'Jordan'],
        'Africa': ['South Africa', 'Senegal', 'Mali', 'Cameroon', 'Zambia', 'Zimbabwe', 'Nigeria', 'Kenya'],
        'Oceania': ['Australia', 'New Zealand'],
        'Caribbean and Central America': ['Bahamas', 'Cuba', 'Puerto Rico', 'Jamaica'],
        'Others': ['Ukrainian SSR', 'Ukranian SSR', 'Bhutan', 'Socialist Federal Republic of Yugoslavia']
    }
    for region, countries in regions.items():
        if country in countries:
            return region
    return 'Others'  # for countries not listed or fitting into the defined regions

def map_language_to_group(language):
    european = ['English Language', 'Spanish Language', 'German Language', 
                     'Dutch Language', 'French Language', 'Romanian Language', 'Norwegian Language', 
                     'Hungarian Language', 'Polish Language', 'Bulgarian Language', 'Icelandic Language', 
                     'Italian Language', 'Greek Language', 'Russian Language', 'Turkish Language', 
                     'Swedish Language', 'Danish Language', 'Serbo-Croatian', 
                     'Portuguese Language', 'Urdu Language', 'Latin Language', 'Slovenian Language', 
                     'German', 'Gaelic', 'Croatian Language', 'Ukrainian Language', 'Irish', 'Armenian Language', 
                     'Slovak Language', 'Lithuanian Language', 'Romani Language', 'Kurdish Language',
                     'Deutsch', 'French', 'Swiss German Language','Napoletano-Calabrese Language', 'Sami Languages','Scottish Gaelic Language', 'Old English Language',
                     'Galician Language','Plautdietsch Language', 'Saami, North Language','Catalan language', 'Bosnian language', 'Albanian language', 'Macedonian Language',
                     'Welsh Language', 'Armenian Language', 'American English', 'Luxembourgish language','Brazilian Portuguese']

    east_asian_se_asian = ['Standard Mandarin', 'Japanese Language', 'Thai Language', 'Korean Language', 
                           'Cantonese', 'Mandarin Chinese', 'Tagalog Language', 'Standard Cantonese', 
                           'Chinese Language', 'Min Nan', 'Filipino Language', 'Standard Tibetan', 
                           'Chinese, Jinyu Language', 'Vietnamese Language', 
                           'Taiwanese', 'Shanghainese', 'Chinese, Hakka Language', 'Hmong Language', 'Tibetan Languages', 'Shanxi', 'Chinese language','Hmong language']

    silent_sign_languages = ['Silent film', 'French Sign Language']

    south_asian = ['Malayalam Language', 'Kannada Language', 'Oriya Language', 'Telugu Language', 
                   'Sinhala Language', 'Nepali Language', 'Hindi Language', 'Tamil Language', 'Bengali Language', 
                   'Punjabi Language', 'Marathi Language', 'Dari', 'Khmer, Central Language']

    african_languages = ['Bambara Language', 'Swahili Language', 'Zulu Language', 'Wolof Language', 
                         'Xhosa Language', 'Mende Language', 'Yolngu Matha', 'Sotho Language', 
                         'Amharic Language','Kinyarwanda Language','South African English']

    middle_east_central_asian = ['Arabic Language', 'Khmer Language', 'Azerbaijani Language', 
                                 'Hebrew Language', 'Georgian Language', 'Assyrian Language', 'Arabic','Egyptian Arabic', 'Persian Language']

    austronesian_pacific = ['Indonesian Language', 'Malay Language', 'Māori Language', 
                            'Samoan Language', 'Hawaiian Language', 
                            'Fijian Language', 'Tongan Language', 'Tahitian Language', 'Gumatj Language', 'Aboriginal Malay Languages', 'Australian English']

    native_american_indigenous = ['Apache, Western Language', 'Albanian Language', 'Tzotzil Language', 
                                  'Sioux Language', 
                                  'Quechua', 
                                  'Crow Language', 'Maya, Yucatán Language', 
                                  'Jamaican Creole English Language', 'Hopi Language', 
                                  'Nahuatl Languages', 
                                  'Navajo Language', 'Inuktitut', 'Apache, Western Language','Tzotz']

    others = ['Gumatj Language', 
              'Assamese Language', 
              'Burmese Language', 'Mongolian language', 'Yiddish Language',
              'Chechen Language', 'Esperanto Language', '𐐖𐐲𐑉𐑋𐑌𐐲','Sumerian','Aramaic Language','Klingon Language']
    
    if language in european:
        return 'Indo-European Languages'
    elif language in east_asian_se_asian:
        return 'East Asian and Southeast Asian Languages'
    elif language in silent_sign_languages:
        return 'Silent and Sign Languages'
    elif language in south_asian:
        return 'South Asian Languages'
    elif language in african_languages:
        return 'African Languages'
    elif language in middle_east_central_asian:
        return 'Middle Eastern and Central Asian Languages'
    elif language in austronesian_pacific:
        return 'Austronesian and Pacific Languages'
    elif language in native_american_indigenous:
        return 'Native American and Indigenous Languages'
    elif language in others:
        return 'Others'
    else:
        return 'Unspecified'
    

def get_params():
    data = pd.read_csv('./data/surv_data_.csv')

    # Preprocessing steps based on the provided instructions
    # Fill missing values in 'Actor height (in meters)' with the mean
    data['Actor height (in meters)'].fillna(data['Actor height (in meters)'].mean(), inplace=True)

    # Remove instances with missing values in 'Genre 1', 'Language 1', and 'Country 1'
    data.dropna(subset=['Genre 1', 'Language 1', 'Country 1'], inplace=True)

    # Splitting the dataset into features (X) and target variable (y)
    X = data[['Actor height (in meters)', 'First movie year', 'Last movie year', 'Total movies', 
            'Genre 1', 'Language 1', 'Country 1', 'Career Length', 'Censored']]
    y = data['Actor gender']

    X['Genre 1'] = X['Genre 1'].apply(map_genre_to_category)
    X['Country 1'] = X['Country 1'].apply(map_country_to_region)
    X['Language 1'] = X['Language 1'].apply(map_language_to_group)
    X.rename(columns={"Language 1": "Language"}, inplace=True)
    X.rename(columns={"Country 1": "Region"}, inplace=True)
    X.rename(columns={"Genre 1": "Genre"}, inplace=True)

    # Separating instances where gender is missing (for testing)
    X_test = X[y.isnull()]
    y_test = y[y.isnull()]

    # Remaining data for training and validation
    X_train_val = X[~y.isnull()]
    y_train_val = y[~y.isnull()]
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)

    # Export this data
    X_train.to_csv('./data/X_train.csv', index=False)
    y_train.to_csv('./data/y_train.csv', index=False)
    X_val.to_csv('./data/X_val.csv', index=False)
    y_val.to_csv('./data/y_val.csv', index=False)
    X_test.to_csv('./data/X_test.csv', index=False)

    X_test = pd.read_csv('./data/X_test.csv')
    X_train = pd.read_csv('./data/X_train.csv')
    X_val = pd.read_csv('./data/X_val.csv')
    y_train = pd.read_csv('./data/y_train.csv')
    y_val = pd.read_csv('./data/y_val.csv')

    # One-hot encoding the 'Genre', 'Language', and 'Country' columns
    X_train_encoded = pd.get_dummies(X_train, columns=['Genre', 'Language', 'Region'])
    X_val_encoded = pd.get_dummies(X_val, columns=['Genre', 'Language', 'Region'])
    X_test_encoded = pd.get_dummies(X_test, columns=['Genre', 'Language', 'Region'])

    # Create a mask to identify rows to keep
    mask = (X_train_encoded['Actor height (in meters)'] < 50) & (X_train_encoded['Total movies'] < 600) & (X_train_encoded['Career Length'] < 80)

    # Apply the mask to both X_train_encoded and y_train
    X_train_encoded = X_train_encoded[mask]
    y_train = y_train[mask]

    y_val_binary = pd.DataFrame(y_val['Actor gender'].map({'M': 1, 'F': 0}))
    y_train_binary = pd.DataFrame(y_train['Actor gender'].map({'M': 1, 'F': 0}))

    # Create a MinMaxScaler object
    min_max_scaler = MinMaxScaler()

    numerical_features = ['Actor height (in meters)', 'First movie year', 'Last movie year', 'Total movies', 'Career Length']
    # Scaling the numerical features using Min-Max Scaler
    X_train_scaled = X_train_encoded.copy()
    X_val_scaled = X_val_encoded.copy()
    X_test_scaled = X_test_encoded.copy()

    # Apply the scaler to the numerical features
    X_train_scaled[numerical_features] = min_max_scaler.fit_transform(X_train_scaled[numerical_features])
    X_val_scaled[numerical_features] = min_max_scaler.transform(X_val_scaled[numerical_features])
    X_test_scaled[numerical_features] = min_max_scaler.transform(X_test_scaled[numerical_features])

    return X_train_scaled, y_train_binary, X_val_scaled, y_val_binary, X_test_scaled, y_test

if __name__ == "__main__":
    get_params()