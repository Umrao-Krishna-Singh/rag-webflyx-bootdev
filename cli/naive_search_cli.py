from constants import (
    Movie,
    preprocess_strings,
    loaded_movie_data,
    loaded_stop_words,
    stem,
)


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


def search_title(movie: Movie, keyword_parts: list[str]) -> bool:
    return True if search_logic(movie, keyword_parts) else False


def naive_search(keyword: str):
    movie_data = loaded_movie_data
    stop_words = loaded_stop_words
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
