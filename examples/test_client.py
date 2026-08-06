"""
Minimal Python client for the iDISCOVR VR Character Service.

Usage:
    python examples/test_client.py path/to/audio.wav [character_name]

Demonstrates the one-shot endpoint (simplest integration) and reuses the
returned session_id so conversation memory persists across calls — pass
it back in on the next request the same way a real client would.
"""

import sys
import requests

BASE_URL = "http://localhost:8000"


def send_message(audio_path: str, character_name: str = "Genie", session_id: str | None = None) -> dict:
    with open(audio_path, "rb") as f:
        files = {"audio_file": f}
        data = {"character_name": character_name}
        if session_id:
            data["session_id"] = session_id
        resp = requests.post(f"{BASE_URL}/v1/vr-chat-sync", files=files, data=data)

    if resp.status_code == 429:
        print("Service busy, try again shortly.")
        return {}
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py path/to/audio.wav [character_name]")
        sys.exit(1)

    audio_path = sys.argv[1]
    character_name = sys.argv[2] if len(sys.argv) > 2 else "Genie"

    result = send_message(audio_path, character_name)
    if not result:
        sys.exit(1)

    print(f"You said:      {result['user_transcript']}")
    print(f"{character_name} says: {result['character_transcript']}")
    print(f"Voice audio:   {BASE_URL}{result['voice_audio_url']}")
    if result.get("talking_video_url"):
        print(f"Video:         {BASE_URL}{result['talking_video_url']}")
    elif result.get("video_error"):
        print(f"Video failed:  {result['video_error']}")

    # Send a follow-up in the same session, to show memory persisting
    session_id = result["session_id"]
    print(f"\nSession: {session_id} (reuse this session_id to continue the conversation)")