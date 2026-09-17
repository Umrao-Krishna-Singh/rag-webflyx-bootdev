import json
import string
from nltk.stem import PorterStemmer
from typing import TypedDict


class Movie(TypedDict):
    id: int
    title: str
    description: str


class MoviesResponse(TypedDict):
    movies: list[Movie]


stemmer = PorterStemmer()
stem = stemmer.stem


def search_logic(movie: Movie, keyword_parts: list[str]) -> bool:
    # we are trying to find any part of the keyword with any part of title
    title = list(map(stem, (preprocess_strings(movie["title"]).split())))

    found = False
    for part in keyword_parts:
        found = part in title

        # once any part matches, the whole string search is true so we break out
        if found:
            break

    return found


def preprocess_strings(part: str) -> str:
    # Translation table ensures that all punctuation is removed (from_string,to_string, remove_string)
    translation_table = str.maketrans("", "", string.punctuation)

    return part.lower().translate(translation_table)


def search_title(movie: Movie, keyword_parts: list[str]) -> bool:
    return True if search_logic(movie, keyword_parts) else False


def load_stop_words() -> list[str]:
    with open("data/stop_words.txt", "r") as file:
        stop_words = file.read().splitlines()

    stop_words = list(map(preprocess_strings, stop_words))

    return stop_words


def load_movies() -> MoviesResponse:
    with open("data/movies.json", "r") as file:
        movie_data: MoviesResponse = json.load(file)
    return movie_data


def naive_search(keyword: str):
    movie_data = load_movies()
    stop_words = load_stop_words()
    parts = keyword.split()
    keyword_parts = list(
        map(
            stem,
            map(preprocess_strings, filter(lambda part: part not in stop_words, parts)),
        )
    )

    results = list(
        filter(lambda movie: search_title(movie, keyword_parts), movie_data["movies"])
    )

    if len(results) > 5:
        results = results[:5]

    return results
