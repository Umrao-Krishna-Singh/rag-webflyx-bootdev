from constants import (
    Movie,
    preprocess_strings,
    stem,
    loaded_stop_words,
    loaded_movie_data,
    BM25_K1,
    BM25_B,
)
from typing import Dict
from pickle import dump, load
from os.path import exists
from collections import Counter
import math
import heapq


class InvertedIndex:
    def __init__(self) -> None:
        self.movie_data = loaded_movie_data
        # token->[movie.id] ===  maps token to list of ids in movie array that satisfies the token
        self.index: Dict[str, list[int]] = {}
        # movie.id->movie index === docmap maps movie id to movie document index in the movies array
        self.docmap: Dict[int, int] = {}
        # movie.id->Counter{movie tokens} frequency of tokens on each movie
        self.term_frequencies = {}
        # movie.id-> len(title+description) === BM25 docs length normalization
        self.docs_length = {}

    def __tokenize_text(self, sentence: str) -> list[str]:
        parts = map(preprocess_strings, sentence.split())
        tokens = list(
            # set(
            map(
                stem,
                filter(lambda part: part and part not in loaded_stop_words, parts),
            )
            # )
        )
        return tokens

    def __add_document(self, doc_id: int, text: str):
        tokens = self.__tokenize_text(text)
        count = Counter(tokens)
        self.term_frequencies[doc_id] = count
        self.docs_length[doc_id] = len(tokens)

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
                movie.get("id"), f"{movie['title']} {movie['description']}"
            )
            self.docmap[movie["id"]] = idx

    def save(self):
        with open("cache/index.pkl", "wb") as i:
            dump(self.index, i)

        with open("cache/docmap.pkl", "wb") as d:
            dump(self.docmap, d)

        with open("cache/term_frequencies.pkl", "wb") as t:
            dump(self.term_frequencies, t)

        with open("cache/docs_length.pkl", "wb") as l:
            dump(self.docs_length, l)

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
        if not exists("cache/docs_length.pkl"):
            raise LookupError("docs_length.pkl file not found")

        with open("cache/index.pkl", "rb") as i:
            self.index = load(i)
        with open("cache/docmap.pkl", "rb") as d:
            self.docmap = load(d)
        with open("cache/term_frequencies.pkl", "rb") as t:
            self.term_frequencies = load(t)
        with open("cache/docs_length.pkl", "rb") as l:
            self.docs_length = load(l)

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
        self.load()
        total_doc_count = len(self.movie_data["movies"])
        # we will probably not need to tokenize text here again - already done before search?
        search_tokens = self.__tokenize_text(token)
        term_match_doc = self.get_document(search_tokens[0])
        if term_match_doc is None:
            raise ValueError(f"{token} not found!")
        term_match_doc_count = len(term_match_doc)

        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf

    def get_bm25_idf(self, term: str) -> float:
        # self.load()
        N = len(self.movie_data["movies"])
        # we will probably not need to tokenize text here again - already done before search?
        search_tokens = self.__tokenize_text(term)
        term_match_doc = self.get_document(search_tokens[0])
        if term_match_doc is None:
            raise ValueError(f"{term} not found!")
        df = len(term_match_doc)

        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B):
        # we will probably not need to tokenize text here again - already done before search?
        search_tokens = self.__tokenize_text(term)
        tf = self.get_tf(doc_id, search_tokens[0])
        doc_length = self.docs_length[doc_id]
        avg_doc_length = self.__get_avg_doc_length()

        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def __get_avg_doc_length(self) -> float:
        avg_len = 0.0
        doc_count = len(self.docs_length) or 1
        for doc in self.docs_length:
            avg_len += self.docs_length[doc]

        return avg_len / doc_count

    def bm25(self, doc_id: int, term: str):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query: str, limit: int = 5):
        self.load()
        search_tokens = self.__tokenize_text(query)
        movies: list[tuple[Movie, float]] = []
        heap: list[tuple[float, int]] = []

        # movie.id -> bm25 score
        scores: Dict[int, float] = {}
        count = 0
        for st in search_tokens:
            count += 1
            doc_ids = self.get_document(st)
            if doc_ids is None:
                continue

            for doc_id in doc_ids:
                score = scores.get(doc_id)
                if score:
                    scores[doc_id] = score + self.bm25(doc_id, st)
                else:
                    scores[doc_id] = self.bm25(doc_id, st)

        for key in scores:
            heapq.heappush(heap, (-scores[key], key))

        for _ in range(limit):
            val, item = heapq.heappop(heap)
            movies.append((self.movie_data["movies"][self.docmap[item]], -val))

        return movies
