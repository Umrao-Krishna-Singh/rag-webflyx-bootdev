import argparse
from naive_search_cli import naive_search
from inverted_index_search_cli import InvertedIndex


def build(term: str):
    print("Starting build process...")
    i_idx = InvertedIndex()
    i_idx.build_command()
    docs = i_idx.get_document(term)

    if docs is None:
        raise ValueError(
            f"Could not find the document! This should not happen in this course! 'term': {term}"
        )

    print(f"First document for token '{term}' = {docs[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    build_parser.add_argument(
        "query",
        type=str,
        help="Search query from inverted index",
        nargs="?",
        default="merida",
    )
    term_frequency_parser = subparsers.add_parser(
        "tf", help="Search by doc id and term to get the frequency of the term"
    )
    term_frequency_parser.add_argument("id", type=int, help="document id to search")
    term_frequency_parser.add_argument("query", type=str, help="term to search")
    inv_doc_freq_parser = subparsers.add_parser(
        "idf", help="Search idf value for a term"
    )
    inv_doc_freq_parser.add_argument(
        "term", type=str, help="Search term to determine the idf value"
    )

    args = parser.parse_args()

    match args.command:
        # naive search approach
        # case "search":
        #     # print the search query here
        #     print(f"Searching for: {args.query}")
        #     matches = naive_search(args.query)
        #     for i, match in enumerate(matches):
        #         print(f"{i+1}. {match['title']}")

        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            movie_idx = InvertedIndex()
            matches = movie_idx.search_mv(args.query)
            for i, match in enumerate(matches):
                print(f"{i+1}. id:{match['id']} - Title: {match['title']}")

        case "build":
            build(args.query)

        case "tf":
            movie_idx = InvertedIndex()
            movie_idx.load()
            freq = movie_idx.get_tf(args.id, args.query)
            print(freq)

        case "idf":
            movie_idx = InvertedIndex()
            idf = movie_idx.calculate_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
