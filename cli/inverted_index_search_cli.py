from naive_search_cli import (
    Movie,
    MoviesResponse,
    load_movies,
    preprocess_strings,
    load_stop_words,
    stem,
)
from typing import Dict
from pickle import dump, load
from os.path import exists
from collections import Counter
import math

movie_data = load_movies()
stop_words = load_stop_words()


class InvertedIndex:
    def __init__(self) -> None:
        self.movie_data = movie_data
        # token->[movie.id] ===  maps token to list of ids in movie array that satisfies the token
        self.index: Dict[str, list[int]] = {}
        # movie.id->movie index === docmap maps movie id to movie document index in the movies array
        self.docmap: Dict[int, int] = {}
        # movie.id->Counter{movie tokens} frequency of tokens on each movie
        self.term_frequencies = {}

    def __tokenize_text(self, sentence: str) -> list[str]:
        parts = sentence.split()
        tokens = list(
            # set(
            map(
                stem,
                map(
                    preprocess_strings,
                    filter(lambda part: part not in stop_words, parts),
                ),
            )
            # )
        )
        return tokens

    def __add_document(self, doc_id: int, text: str):
        tokens = self.__tokenize_text(text)
        count = Counter(tokens)
        self.term_frequencies[doc_id] = count

        # remove duplicates here
        tokens = list(set(tokens))
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

        with open("cache/term_frequencies.pkl", "wb") as t:
            dump(self.term_frequencies, t)

    def build_command(self):
        self.build()
        self.save()

    def load(self):
        if not exists("cache/index.pkl"):
            raise LookupError("index.pkl file not found")
        if not exists("cache/docmap.pkl"):
            raise LookupError("docmap.pkl file not found")
        if not exists("cache/term_frequencies.pkl"):
            raise LookupError("term_frequencies.pkl file not found")

        with open("cache/index.pkl", "rb") as i:
            self.index = load(i)
        with open("cache/docmap.pkl", "rb") as d:
            self.docmap = load(d)
        with open("cache/term_frequencies.pkl", "rb") as t:
            self.term_frequencies = load(t)

    def get_tf(self, doc_id: int, term: str):
        return self.term_frequencies[doc_id][term]

    def search_mv(self, search_term: str) -> list[Movie]:
        self.load()
        search_tokens = self.__tokenize_text(search_term)
        movies: list[Movie] = []

        for st in search_tokens:
            doc_ids = self.get_document(st)
            if doc_ids is None:
                continue

            for doc_id in doc_ids:
                movies.append(self.movie_data["movies"][self.docmap[doc_id]])
                if len(movies) >= 5:
                    break

        return movies

    def calculate_idf(self, token: str) -> float:
        total_doc_count = len(self.movie_data["movies"])
        self.load()
        search_tokens = self.__tokenize_text(token)
        term_match_doc = self.get_document(search_tokens[0])
        if term_match_doc is None:
            raise ValueError(f"{token} not found!")
        term_match_doc_count = len(term_match_doc)

        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf

    def get_bm25_idf(self, term: str) -> float:
        self.load()
        N = len(self.movie_data["movies"])
        search_tokens = self.__tokenize_text(term)
        term_match_doc = self.get_document(search_tokens[0])
        if term_match_doc is None:
            raise ValueError(f"{term} not found!")
        df = len(term_match_doc)

        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf
