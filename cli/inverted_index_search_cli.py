from naive_search_cli import (
    Movie,
    MoviesResponse,
    load_movies,
    preprocess_strings,
    load_stop_words,
    stem,
)
from typing import Dict
from pickle import dump

movie_data = load_movies()
stop_words = load_stop_words()


class InvertedIndex:
    def __init__(self) -> None:
        self.movie_data = movie_data
        # maps token to list of ids in movie array that satisfies the token
        self.index: Dict[str, list[int]] = {}
        # docmap maps movie id to movie document index in the movies array
        self.docmap: Dict[int, int] = {}

    def __tokenize_text(self, sentence: str) -> list[str]:
        parts = sentence.split()
        tokens = list(
            set(
                map(
                    stem,
                    map(
                        preprocess_strings,
                        filter(lambda part: part not in stop_words, parts),
                    ),
                )
            )
        )
        return tokens

    def __add_document(self, doc_id: int, text: str):
        tokens = self.__tokenize_text(text)
        for token in tokens:
            self.index.setdefault(token, []).append(doc_id)

    def get_document(self, term: str) -> list[int] | None:
        doc_ids = self.index.get(term)
        if doc_ids:
            doc_ids.sort()

        return doc_ids

    def build(self):
        mv = self.movie_data.get("movies")

        for idx, movie in enumerate(mv):
            self.__add_document(
                movie.get("id"), f"{movie['title']} f{movie['description']}"
            )
            self.docmap[movie["id"]] = idx

    def save(self):
        with open("cache/index.pkl", "wb") as i:
            dump(self.index, i)

        with open("cache/docmap.pkl", "wb") as d:
            dump(self.docmap, d)

    def build_command(self):
        self.build()
        self.save()
