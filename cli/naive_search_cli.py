import json
import string

def naive_search(keyword:str):
    with open("data/movies.json", "r") as file:
        movie_data = json.load(file)
    
    def search_title(movie):
        translation_table = str.maketrans("", "", string.punctuation)

        return True if keyword.lower().translate(translation_table) in movie['title'].lower().translate(translation_table)  else False
    
    results = list(filter(search_title, movie_data['movies']))
    
    if len(results)>5:
        results = results[:5]
    
    return results