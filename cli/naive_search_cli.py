import json
import string
from typing import TypedDict
class Movie(TypedDict):
    id: int
    title: str
    description: str

class MoviesResponse(TypedDict):
    movies: list[Movie]


def search_logic(movie:Movie,keyword:str)->bool:
    # Translation table ensures that all punctuation is removed (from_string,to_string, remove_string)
    translation_table = str.maketrans("", "", string.punctuation)
    
    # we are trying to find any part of the keyword with any part of title
    parts = keyword.split()
    found = False
    
    for part in parts:
        found = part.lower().translate(translation_table) in movie['title'].lower().translate(translation_table)
        
        # once any part matches, the whole string search is true so we break out
        if found:
            break
    
    return found

def search_title(movie:Movie,keyword:str):        
    return True if search_logic(movie,keyword) else False

def naive_search(keyword:str):
    with open("data/movies.json", "r") as file:
        movie_data:MoviesResponse = json.load(file)
        
    results = list(filter(lambda movie: search_title(movie, keyword), movie_data['movies']))
    
    if len(results)>5:
        results = results[:5]
    
    return results