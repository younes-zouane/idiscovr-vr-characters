import os
import requests
from dotenv import load_dotenv
load_dotenv()

def upload_file(filepath, content_type):
    """Upload file to tmpfiles.org and get back a public URL"""
    with open(filepath, "rb") as f:
        response = requests.post(
            "https://tmpfiles.org/api/v1/upload",
            files={"file": (os.path.basename(filepath), f, content_type)}
        )
    result = response.json()
    print(f"Upload response: {result}")
    # tmpfiles.org returns {"status":"success","data":{"url":"https://tmpfiles.org/..."}}
    # but the direct download URL needs /dl/ inserted
    url = result["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")
    print(f"Uploaded {filepath} → {url}")
    return url

def generate_talking_video(image_path, audio_path, output_path="test_output.mp4"):
    # Step 1: upload both files to get public URLs
    image_url = upload_file(image_path, "image/jpeg")
    audio_url = upload_file(audio_path, "audio/wav")

    # Step 2: call DeepInfra with URLs (flat JSON, no nesting)
    response = requests.post(
        "https://api.deepinfra.com/v1/inference/PrunaAI/p-video-avatar",
        headers={
            "Authorization": f"Bearer {os.environ['DEEPINFRA_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "image": image_url,
            "audio": audio_url,
            "video_prompt": "A person speaking naturally, subtle head movement.",
            "resolution": "720p",
            "disable_safety_filter": True
        },
        timeout=180
    )

    print("STATUS:", response.status_code)
    result = response.json()
    print("FULL RESPONSE:", result)

    # Step 3: download the video
    if "video_url" in result:
        video_url = result["video_url"]
        print("Downloading video from:", video_url)
        video_response = requests.get(video_url, timeout=60)
        with open(output_path, "wb") as f:
            f.write(video_response.content)
        print(f"Saved to {output_path}")
        return output_path
    else:
        print("No video_url in response")
        return None

result = generate_talking_video(
    image_path="character_images/genie.jpg",
    audio_path="reply.wav"
)