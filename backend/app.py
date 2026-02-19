"""Fetch.ai uAgent that manages Notion notes via natural language."""

import json
import os
import requests
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
) 
from uagents_core.utils.registration import (
    register_chat_agent,
    RegistrationRequestCredentials,
)

from .notion_client_wrapper import NotionNotes

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AGENT_NAME = "notion-notes-agent"
SEED_PHRASE = os.getenv("AGENT_SEED_PHRASE", "notion-agent-seed")
AGENTVERSE_KEY = os.getenv("ILABS_AGENTVERSE_API_KEY")

ASI1_URL = "https://api.asi1.ai/v1/chat/completions"

def _asi1_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('ASI_ONE_API_KEY')}",
    }

# Notion client – initialised lazily after env vars are loaded
notion: NotionNotes | None = None


agent = Agent(
    name=AGENT_NAME,
    seed=SEED_PHRASE,
    port=8000,
    mailbox=True,
   #handle_messages_concurrently=True,
)

protocol = Protocol(spec=chat_protocol_spec)


SYSTEM_PROMPT = """\
You are an intent classifier for a Notion notes assistant.
Given the user message, respond ONLY with valid JSON (no markdown fences).

Possible intents:
- connect_notion   : user wants to test or verify the Notion connection
- search_notes     : user wants to find notes by keyword
- list_notes       : user wants to see recent/latest notes
- read_note        : user wants to read the content of a specific note (by title)
- create_note      : user wants to create a new note/page
- append_note      : user wants to add text to an existing note
- add_todo         : user wants to add a to-do/task item to a note
- archive_note     : user wants to archive/delete a note
- general_query    : anything else / general question

For search_notes, extract: query (search keyword string), limit (int, default 5).
For list_notes, extract: limit (int, default 5).
For read_note, extract: title (the note title to look up).
For create_note, extract: title (new page title), content (optional body text, default "").
For append_note, extract: title (existing note title), text (the text to append).
For add_todo, extract: title (existing note title to add the task to), task (the to-do item text).
For archive_note, extract: title (the note title to archive).

Response format:
{"intent": "<intent>", "params": {<extracted params or empty dict>}}
"""


def classify_intent(user_text: str) -> dict:
    """Send the user message to ASI:1 and parse the JSON intent."""
    try:
        r = requests.post(
            ASI1_URL,
            headers=_asi1_headers(),
            json={
                "model": "asi1",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                "max_tokens": 256,
            },
        )
        raw = r.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if the model adds them
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except Exception:
        return {"intent": "general_query", "params": {}}


def _ensure_notion() -> str | None:
    """Lazily initialise the Notion client. Returns an error string or None."""
    global notion
    if notion is not None:
        return None
    try:
        notion = NotionNotes()
        return None
    except Exception as e:
        return f"Notion is not configured: {e}"


def handle_connect_notion(ctx: Context) -> str:
    err = _ensure_notion()
    if err:
        return err
    ok = notion.test_connection()
    return "Notion connection successful!" if ok else "Failed to connect to Notion. Check your API key."


def handle_search_notes(ctx: Context, params: dict) -> str:
    err = _ensure_notion()
    if err:
        return err
    query = params.get("query", "")
    limit = int(params.get("limit", 5))
    notes = notion.search_notes(query=query, limit=limit)
    if not notes:
        return f"No notes found matching \"{query}\"."
    return _format_notes(notes)


def handle_list_notes(ctx: Context, params: dict) -> str:
    err = _ensure_notion()
    if err:
        return err
    limit = int(params.get("limit", 5))
    notes = notion.list_recent_notes(limit)
    if not notes:
        return "No notes found in your Notion workspace."
    return _format_notes(notes)


def handle_read_note(ctx: Context, params: dict) -> str:
    err = _ensure_notion()
    if err:
        return err
    title = params.get("title", "")
    notes = notion.search_notes(query=title, limit=1)
    if not notes:
        return f"Couldn't find a note titled \"{title}\"."
    note = notes[0]
    content = notion.get_page_content(note["id"])
    header = f"**{note['title']}**\n"
    if content:
        return header + content
    return header + "(This page has no text content.)"


def handle_create_note(ctx: Context, params: dict) -> str:
    err = _ensure_notion()
    if err:
        return err
    title = params.get("title", "Untitled")
    content = params.get("content", "")
    try:
        page = notion.create_page(title=title, content=content)
        return f"Created note **{title}**\n{page.get('url', '')}"
    except Exception as e:
        return f"Failed to create note: {e}"


def handle_append_note(ctx: Context, params: dict) -> str:
    err = _ensure_notion()
    if err:
        return err
    title = params.get("title", "")
    text = params.get("text", "")
    if not title or not text:
        return "I need both a note title and the text to append."
    notes = notion.search_notes(query=title, limit=1)
    if not notes:
        return f"Couldn't find a note titled \"{title}\"."
    notion.append_to_page(notes[0]["id"], text)
    return f"Added text to **{notes[0]['title']}**."


def handle_add_todo(ctx: Context, params: dict) -> str:
    err = _ensure_notion()
    if err:
        return err
    title = params.get("title", "")
    task = params.get("task", "")
    if not title or not task:
        return "I need both a note title and the task text."
    notes = notion.search_notes(query=title, limit=1)
    if not notes:
        return f"Couldn't find a note titled \"{title}\"."
    notion.append_todo(notes[0]["id"], task)
    return f"Added to-do \"**{task}**\" to **{notes[0]['title']}**."


def handle_archive_note(ctx: Context, params: dict) -> str:
    err = _ensure_notion()
    if err:
        return err
    title = params.get("title", "")
    if not title:
        return "I need the title of the note to archive."
    notes = notion.search_notes(query=title, limit=1)
    if not notes:
        return f"Couldn't find a note titled \"{title}\"."
    notion.archive_page(notes[0]["id"])
    return f"Archived **{notes[0]['title']}**."


def handle_general_query(user_text: str) -> str:
    """Fall back to ASI:1 for general questions."""
    try:
        r = requests.post(
            ASI1_URL,
            headers=_asi1_headers(),
            json={
                "model": "asi1",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful Notion notes assistant. "
                            "Answer the user's question concisely."
                        ),
                    },
                    {"role": "user", "content": user_text},
                ],
                "max_tokens": 1024,
            },
        )
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Sorry, I wasn't able to process that."


def _format_notes(notes: list[dict]) -> str:
    """Format a list of note dicts into a readable string."""
    lines = []
    for n in notes:
        edited = n.get("last_edited", "")[:10] if n.get("last_edited") else ""
        line = f"- **{n['title']}**"
        if edited:
            line += f" (edited {edited})"
        lines.append(line)
    return "\n".join(lines)


INTENT_HANDLERS = {
    "connect_notion": lambda ctx, p, t: handle_connect_notion(ctx),
    "search_notes": lambda ctx, p, t: handle_search_notes(ctx, p),
    "list_notes": lambda ctx, p, t: handle_list_notes(ctx, p),
    "read_note": lambda ctx, p, t: handle_read_note(ctx, p),
    "create_note": lambda ctx, p, t: handle_create_note(ctx, p),
    "append_note": lambda ctx, p, t: handle_append_note(ctx, p),
    "add_todo": lambda ctx, p, t: handle_add_todo(ctx, p),
    "archive_note": lambda ctx, p, t: handle_archive_note(ctx, p),
    "general_query": lambda ctx, p, t: handle_general_query(t),
}


@protocol.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    # Acknowledge receipt
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id),
    )

    # Extract text from the message
    text = ""
    for item in msg.content:
        if isinstance(item, TextContent):
            text += item.text

    if not text.strip():
        response_text = "I didn't receive any text. Please send me a message!"
    else:
        # Classify intent via ASI:1
        result = classify_intent(text)
        intent = result.get("intent", "general_query")
        params = result.get("params", {})

        ctx.logger.info(f"Intent: {intent} | Params: {params}")

        handler = INTENT_HANDLERS.get(intent, INTENT_HANDLERS["general_query"])
        response_text = handler(ctx, params, text)

    # Send response
    await ctx.send(
        sender,
        ChatMessage(
            timestamp=datetime.now(timezone.utc),
            msg_id=uuid4(),
            content=[
                TextContent(type="text", text=response_text),
                EndSessionContent(type="end-session"),
            ],
        ),
    )


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(protocol, publish_manifest=True)

README = """# Notion Notes Agent

![tag:notion-agent](https://img.shields.io/badge/notion-3D8BD3)
![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)

A Fetch.ai agent that lets you browse your Notion workspace with natural language. No database ID required — just connect your Notion integration and go.

## Features

- **List recent notes** from your workspace
- **Search notes** by keyword
- **Read note content** by title
- **General Q&A** powered by ASI:1

## Usage

Send a chat message like:
- "Show my latest notes"
- "Search for meeting notes"
- "Read my note titled Project Plan"
- "What pages do I have about marketing?"
"""

@agent.on_event("startup")
async def startup_handler(ctx: Context):
    """Initialize agent and register with Agentverse on startup."""
    ctx.logger.info(f"Agent starting: {ctx.agent.name} at {ctx.agent.address}")

    # Try connecting to Notion early (non-fatal if it fails)
    err = _ensure_notion()
    if err:
        ctx.logger.warning(f"Notion not connected on startup: {err}")
    else:
        ctx.logger.info("Notion connection verified")

    # Register with Agentverse
    if AGENTVERSE_KEY and SEED_PHRASE:
        try:
            register_chat_agent(
                AGENT_NAME,
                agent._endpoints[0].url,
                active=True,
                credentials=RegistrationRequestCredentials(
                    agentverse_api_key=AGENTVERSE_KEY,
                    agent_seed_phrase=SEED_PHRASE,
                ),
                readme=README,
                description="A Notion notes assistant that searches, lists, and reads your Notion pages via natural language.",
            )
            ctx.logger.info("Registered with Agentverse")
        except Exception as e:
            ctx.logger.error(f"Failed to register with Agentverse: {e}")
    else:
        ctx.logger.warning("AGENTVERSE_KEY or SEED_PHRASE not set, skipping Agentverse registration")

if __name__ == "__main__":
    agent.run()