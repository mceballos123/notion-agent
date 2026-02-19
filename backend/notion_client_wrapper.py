"""Thin wrapper around the Notion SDK — uses embedding-based semantic search."""

import logging
import os

from notion_client import Client
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
_embedding_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _get_embedding(text: str) -> list[float]:
    """Return the embedding vector for a text string."""
    r = _embedding_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float",
    )
    return r.data[0].embedding


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (no numpy needed)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class NotionNotes:
    """Searches and reads pages across the entire Notion workspace the integration can access."""

    def __init__(self):
        api_key = os.getenv("NOTION_API_KEY")
        if not api_key:
            raise ValueError("NOTION_API_KEY must be set in .env")
        self.client = Client(auth=api_key)
        # In-memory embedding cache: page_id -> (title, embedding)
        self._cache: dict[str, tuple[str, list[float]]] = {}

    def test_connection(self) -> bool:
        """Verify the Notion API key works by running a simple search."""
        try:
            self.client.search(query="", page_size=1)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def search_notes(self, query: str = "", limit: int = 10) -> list[dict]:
        """Search pages semantically using embeddings.

        If query is empty, falls back to listing recent pages (no embedding needed).
        """
        # Fetch all accessible pages from Notion
        all_pages = self._fetch_all_pages()

        if not all_pages:
            logger.info("No pages accessible in workspace")
            return []

        # No query → just return the most recent pages
        if not query.strip():
            return all_pages[:limit]

        # Embed the query
        query_embedding = _get_embedding(query)

        # Embed each page title (uses cache to avoid re-embedding)
        scored = []
        for page in all_pages:
            page_id = page["id"]
            title = page["title"]

            if page_id in self._cache and self._cache[page_id][0] == title:
                title_embedding = self._cache[page_id][1]
            else:
                title_embedding = _get_embedding(title)
                self._cache[page_id] = (title, title_embedding)

            score = _cosine_similarity(query_embedding, title_embedding)
            scored.append((score, page))

        # Sort by similarity (highest first) and return top results
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [page for score, page in scored[:limit] if score > 0.3]

        logger.info(
            f"Semantic search query='{query}' matched {len(results)} of {len(all_pages)} pages "
            f"(top score={scored[0][0]:.3f})"
        )
        return results

    def list_recent_notes(self, limit: int = 10) -> list[dict]:
        """Return the most recently edited pages."""
        return self.search_notes(query="", limit=limit)

    # ------------------------------------------------------------------
    # Page operations
    # ------------------------------------------------------------------

    def get_page_content(self, page_id: str) -> str:
        """Retrieve the text content (blocks) of a single page."""
        blocks = self.client.blocks.children.list(block_id=page_id, page_size=100)
        texts = []
        for block in blocks.get("results", []):
            btype = block.get("type", "")
            rich_text = block.get(btype, {}).get("rich_text", [])
            for rt in rich_text:
                plain = rt.get("plain_text", "")
                if plain:
                    texts.append(plain)
        return "\n".join(texts)

    def create_page(self, title: str, content: str = "") -> dict:
        """Create a new page in the workspace (as a child of first accessible page)."""
        children = []
        if content:
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": content}}]
                },
            })
        page = self.client.pages.create(
            parent={"page_id": self._get_root_page_id()},
            properties={
                "title": [{"type": "text", "text": {"content": title}}]
            },
            children=children,
        )
        # Invalidate cache so new page shows up in searches
        self._cache.pop(page["id"], None)
        return {"id": page["id"], "title": title, "url": page.get("url", "")}

    def append_to_page(self, page_id: str, text: str) -> bool:
        """Append a paragraph block to an existing page."""
        self.client.blocks.children.append(
            block_id=page_id,
            children=[{
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": text}}]
                },
            }],
        )
        return True

    def append_todo(self, page_id: str, text: str, checked: bool = False) -> bool:
        """Append a to-do checkbox block to an existing page."""
        self.client.blocks.children.append(
            block_id=page_id,
            children=[{
                "object": "block",
                "type": "to_do",
                "to_do": {
                    "rich_text": [{"type": "text", "text": {"content": text}}],
                    "checked": checked,
                },
            }],
        )
        return True

    def archive_page(self, page_id: str) -> bool:
        """Archive (soft-delete) a page."""
        self.client.pages.update(page_id=page_id, archived=True)
        self._cache.pop(page_id, None)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_all_pages(self) -> list[dict]:
        """Fetch all accessible pages from Notion (up to 100)."""
        result = self.client.search(
            query="",
            filter={"value": "page", "property": "object"},
            sort={"direction": "descending", "timestamp": "last_edited_time"},
            page_size=100,
        )
        return self._extract_pages(result)

    def _get_root_page_id(self) -> str:
        """Get the first accessible page to use as a parent for new pages."""
        result = self.client.search(
            query="",
            filter={"value": "page", "property": "object"},
            page_size=1,
        )
        pages = result.get("results", [])
        if not pages:
            raise ValueError("No accessible pages found to use as parent.")
        return pages[0]["id"]

    @staticmethod
    def _extract_pages(search_result: dict) -> list[dict]:
        """Flatten Notion page objects into simple dicts."""
        pages = []
        for page in search_result.get("results", []):
            props = page.get("properties", {})

            title = "(untitled)"
            for prop in props.values():
                if prop.get("type") == "title":
                    parts = prop.get("title", [])
                    if parts:
                        title = parts[0].get("plain_text", "(untitled)")
                    break

            pages.append({
                "id": page["id"],
                "title": title,
                "url": page.get("url", ""),
                "last_edited": page.get("last_edited_time", ""),
                "created": page.get("created_time", ""),
            })
        return pages
