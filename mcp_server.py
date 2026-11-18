"""Optional local MCP tools for the video clipping studio."""

from __future__ import annotations

import json

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

import pipeline


load_dotenv(pipeline.ROOT / ".env")
mcp = FastMCP("local-video-clipping-studio")


@mcp.tool()
def list_projects() -> str:
    """List locally stored video projects and their rendered clip metadata."""
    return json.dumps(pipeline.list_projects(), ensure_ascii=False, indent=2)


@mcp.tool()
def process_media(source: str, target_count: int = 5, framing_mode: str = "face") -> str:
    """Process a local video path or a YouTube URL into a local clipping project."""
    updates: list[dict[str, str | float]] = []
    project = pipeline.process_video(
        source=source,
        progress=lambda stage, message, percent: updates.append(
            {"stage": stage, "message": message, "progress": percent}
        ),
        target_count=target_count,
        framing_mode=framing_mode,
    )
    return json.dumps(
        {"project": project, "progress": updates[-8:]},
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def rerender_clip(
    project_id: str,
    clip_id: str,
    start: float | None = None,
    end: float | None = None,
    framing_mode: str | None = None,
    captions_enabled: bool | None = None,
    caption_text: str | None = None,
) -> str:
    """Re-render one stored project clip with new trims or caption settings."""
    result = pipeline.refine_clip(
        project_id=project_id,
        clip_id=clip_id,
        start=start,
        end=end,
        framing_mode=framing_mode,
        captions_enabled=captions_enabled,
        caption_text=caption_text,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
