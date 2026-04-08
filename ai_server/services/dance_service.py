from __future__ import annotations

from typing import Any

def analyze_dance(message: dict[str, Any]) -> dict[str, Any] | None:
    if message["type"] == "websocket.disconnect":
        return None
    
    video_chunk = message.get("bytes")
    if video_chunk is not None:
        return {
            "type": "video_chunk_received",
            "success": True,
            "chunk_size": len(video_chunk),
            "message": "video chunk received successfully",
        }
    
    text_message = message.get("text")
    if text_message is not None:
        return {
            "type": "text_received",
            "success": True,
            "message": text_message,
        }

    return {
        "type": "unsupported_message",
        "success": False,
        "message": "Unsupported websocket message",
    }

    return