"""Thin wrapper around the Notion SDK — uses search API, no database ID required."""

import logging
from notion_client import Client
import os

logger = logging.getLogger(__name__)

class NotionNotes:
    """Searches and reads pages across the entire Notion workspace the integration can access."""

    def __init__(self):
        api_key = os.getenv("NOTION_API_KEY")
        if not api_key:
            raise ValueError("NOTION_API_KEY must be set in .env")
        self.client = Client(auth=api_key)

    def test_connection(self) -> bool:
        """Verify the Notion API key works by running a simple search."""
        try:
            self.client.search(query="", page_size=1)
            return True
        except Exception:
            return False

    def search_notes(self, query: str = "", limit: int = 10) -> list[dict]:
        """Search pages by keyword across the whole workspace."""
        result = self.client.search(
            query=query,
            filter={"value": "page", "property": "object"},
            sort={"direction": "descending", "timestamp": "last_edited_time"},
            page_size=limit,
        )
        logger.info(f"Notion search query='{query}' returned {len(result.get('results', []))} results")
        if not result.get("results"):
            # Try without filter to see if there's anything at all
            raw = self.client.search(query="", page_size=5)
            logger.info(f"Unfiltered search returned {len(raw.get('results', []))} results")
            for r in raw.get("results", [])[:3]:
                logger.info(f"  -> object={r.get('object')}, id={r.get('id')}")
        return self._extract_pages(result)

    def list_recent_notes(self, limit: int = 10) -> list[dict]:
        """Return the most recently edited pages."""
        return self.search_notes(query="", limit=limit)

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

    @staticmethod
    def _extract_pages(search_result: dict) -> list[dict]:
        """Flatten Notion page objects into simple dicts."""
        pages = []
        for page in search_result.get("results", []):
            props = page.get("properties", {})

            # Find the title — could be under any property name
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
