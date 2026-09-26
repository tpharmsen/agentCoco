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
        return self._get("/search/movie", query=query, language="en-US")

    def discover_movies(self, **filters: str | float) -> list[dict]:
        return self._get("/discover/movie", language="en-US", **filters)

    def movie_details(self, movie_id: int) -> dict:
        return self._get(f"/movie/{movie_id}", language="en-US")

    def recommendations(self, movie_id: int) -> list[dict]:
        return self._get(f"/movie/{movie_id}/recommendations", language="en-US")

    def similar_movies(self, movie_id: int) -> list[dict]:
        return self._get(f"/movie/{movie_id}/similar", language="en-US")

    def credits(self, movie_id: int) -> dict:
        return self._get(f"/movie/{movie_id}/credits", language="en-US")

    def watch_providers(self, movie_id: int) -> dict:
        return self._get(f"/movie/{movie_id}/watch/providers")

    def videos(self, movie_id: int) -> list[dict]:
        return self._get(f"/movie/{movie_id}/videos", language="en-US")

    def reviews(self, movie_id: int) -> list[dict]:
        return self._get(f"/movie/{movie_id}/reviews", language="en-US")

    def genres(self) -> list[dict]:
        return self._get("/genre/movie/list", language="en-US")

    def _get(self, path: str, **params: str | float) -> dict | list[dict]:
        response = httpx.get(
            f"{self.base_url}{path}",
            params=params,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "accept": "application/json",
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and "results" in payload:
            return payload["results"]
        return payload
