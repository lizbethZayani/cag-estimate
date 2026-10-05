"""OpenAI embedding vectorizer that never calls the network at construction.

redisvl's ``OpenAITextVectorizer`` performs a live embedding request in its
constructor to discover the vector size, which breaks cache setup when the
provider is unreachable. Here the dimensions are static; the network is only
used by ``embed`` (lookup/store time), where the cache already fails soft.
"""

from typing import Any

KNOWN_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class LazyOpenAIVectorizer:
    """Embeds text through an OpenAI client; exposes static ``dims``."""

    def __init__(self, *, model: str, client: Any, dims: int | None = None) -> None:
        resolved = dims if dims is not None else KNOWN_DIMS.get(model)
        if resolved is None:
            raise ValueError(
                f"Unknown embedding model '{model}': set embedding_dims explicitly "
                f"(known models: {', '.join(sorted(KNOWN_DIMS))})"
            )
        self.model = model
        self.dims = resolved
        self._client = client

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)
