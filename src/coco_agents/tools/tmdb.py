import httpx


class TMDBClient:
    base_url = "https://api.themoviedb.org/3"

    def __init__(self, api_key: str) -> None:
        if not api_key.strip():
            raise ValueError("TMDB_API_KEY is required for movie searches")
        self.api_key = api_key

    def search_movies(self, query: str) -> list[dict]:
        if not query.strip():
            raise ValueError("Movie search query must not be empty")
        response = httpx.get(
            f"{self.base_url}/search/movie",
            params={
                "api_key": self.api_key,
                "query": query,
                "language": "en-US",
                "include_adult": "false",
            },
            timeout=15,
        )
        response.raise_for_status()
        return response.json().get("results", [])
