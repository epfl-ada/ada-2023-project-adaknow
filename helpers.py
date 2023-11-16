import ast
import numpy as np

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