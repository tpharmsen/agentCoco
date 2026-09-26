import httpx


class TavilyClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str) -> list[dict]:
        response = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": self.api_key, "query": query, "search_depth": "basic"},
            timeout=15,
        )
        response.raise_for_status()
        return response.json().get("results", [])
