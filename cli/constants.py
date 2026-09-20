import json
import string
from typing import TypedDict
from nltk.stem import PorterStemmer


class Movie(TypedDict):
    id: int
    title: str
    description: str


class MoviesResponse(TypedDict):
    movies: list[Movie]


stemmer = PorterStemmer()
stem = stemmer.stem


def preprocess_strings(part: str) -> str:
    # Translation table ensures that all punctuation is removed (from_string,to_string, remove_string)
    translation_table = str.maketrans("", "", string.punctuation)

    return part.lower().translate(translation_table)


def load_movies() -> MoviesResponse:
    with open("data/movies.json", "r") as file:
        movie_data: MoviesResponse = json.load(file)
    return movie_data


def load_stop_words() -> list[str]:
    with open("data/stop_words.txt", "r") as file:
        stop_words = file.read().splitlines()

    stop_words = list(map(preprocess_strings, stop_words))

    return stop_words


loaded_movie_data = load_movies()
loaded_stop_words = load_stop_words()
BM25_K1 = 1.5
BM25_B = 0.75  # length normalization strength
