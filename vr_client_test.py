"""
VR service integration test client.

Not a unit test (needs the real server running, with GPU + models loaded —
run `python vr_service.py` in another terminal first). This is a fast,
scriptable stand-in for "what Unity will do," used to prove the contract
works correctly — including genuine concurrent requests, which manual
Swagger UI testing can't exercise — before porting this logic into C#.

Usage:
    pip install requests
    python vr_client_test.py path/to/some.wav path/to/other.wav
"""

import sys
import threading
import requests

BASE_URL = "http://localhost:8000"


def send_message(character_name: str, audio_path: str, session_id: str | None = None) -> dict:
    """Send one voice message, mirroring exactly what Unity's UploadAudioToVRService will do."""
    with open(audio_path, "rb") as f:
        files = {"audio_file": (audio_path, f, "audio/wav")}
        data = {"character_name": character_name}
        if session_id:
            data["session_id"] = session_id
        response = requests.post(f"{BASE_URL}/v1/vr-chat-sync", files=files, data=data, timeout=120)
    response.raise_for_status()
    return response.json()


def download_file(relative_url: str, out_path: str) -> int:
    """Fetch a returned file URL, mirroring the second UnityWebRequest Unity will need
    to do for voice_audio_url / talking_video_url. Returns byte count so callers can
    sanity-check it's non-empty rather than trusting a 200 status alone."""
    response = requests.get(f"{BASE_URL}{relative_url}", timeout=60)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(response.content)
    return len(response.content)


def test_session_continuity(audio_path: str):
    print("\n=== Test 1: session continuity (sequential) ===")
    r1 = send_message("Genie", audio_path)
    session_id = r1["session_id"]
    print(f"  First message -> session_id={session_id}")
    print(f"  Reply: {r1['character_transcript'][:80]}...")

    r2 = send_message("Genie", audio_path, session_id=session_id)
    print(f"  Second message (same session) -> session_id={r2['session_id']}")
    print(f"  Reply: {r2['character_transcript'][:80]}...")

    assert r2["session_id"] == session_id, "session_id changed on reuse — bug"
    print("  PASS: session_id persisted across calls")

    voice_bytes = download_file(r1["voice_audio_url"], "test_voice.wav")
    assert voice_bytes > 0, "downloaded voice file is empty"
    print(f"  PASS: voice_audio_url downloadable ({voice_bytes} bytes)")

    if r1["talking_video_url"]:
        video_bytes = download_file(r1["talking_video_url"], "test_video.mp4")
        assert video_bytes > 0, "downloaded video file is empty"
        print(f"  PASS: talking_video_url downloadable ({video_bytes} bytes)")


def test_true_concurrency(audio_a: str, audio_b: str):
    print("\n=== Test 2: genuine concurrent requests (not just sequential) ===")

    results = {}
    errors = {}

    def worker(name: str, audio_path: str):
        try:
            results[name] = send_message("Genie", audio_path)
        except Exception as e:
            errors[name] = e

    t1 = threading.Thread(target=worker, args=("A", audio_a))
    t2 = threading.Thread(target=worker, args=("B", audio_b))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    if errors:
        print(f"  FAIL: exception(s) during concurrent requests: {errors}")
        return

    sid_a, sid_b = results["A"]["session_id"], results["B"]["session_id"]
    print(f"  Session A: {sid_a}")
    print(f"  Session B: {sid_b}")
    assert sid_a != sid_b, "concurrent requests with no session_id collided on the same ID"
    print("  PASS: concurrent requests got distinct session_ids, no crash, no exception")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vr_client_test.py <audio1.wav> [audio2.wav]")
        sys.exit(1)

    audio1 = sys.argv[1]
    audio2 = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]

    test_session_continuity(audio1)
    test_true_concurrency(audio1, audio2)

    print("\nAll tests completed.")
