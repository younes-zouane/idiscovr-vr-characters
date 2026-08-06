// Minimal Unity client for the iDISCOVR VR Character Service.
// Attach to any GameObject, call SendToCharacter(wavBytes, "Genie").
// Not a full project — copy-paste material for someone who already knows Unity.

using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

public class VRCharacterClient : MonoBehaviour
{
    [SerializeField] private string baseUrl = "http://localhost:8000";
    private string sessionId = null; // populated after the first reply, then reused

    [Serializable]
    public class ChatResponse
    {
        public string status;
        public string session_id;
        public string character;
        public string user_transcript;
        public string character_transcript;
        public string voice_audio_url;
        public string talking_video_url;
        public string video_error;
    }

    public void SendToCharacter(byte[] wavBytes, string characterName)
    {
        StartCoroutine(SendCoroutine(wavBytes, characterName));
    }

    private IEnumerator SendCoroutine(byte[] wavBytes, string characterName)
    {
        WWWForm form = new WWWForm();
        form.AddBinaryData("audio_file", wavBytes, "recording.wav", "audio/wav");
        form.AddField("character_name", characterName);
        if (!string.IsNullOrEmpty(sessionId))
            form.AddField("session_id", sessionId);

        using (UnityWebRequest req = UnityWebRequest.Post($"{baseUrl}/v1/vr-chat-sync", form))
        {
            yield return req.SendWebRequest();

            if (req.responseCode == 429)
            {
                Debug.LogWarning("Service busy, retry shortly.");
                yield break;
            }
            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"Request failed: {req.error}");
                yield break;
            }

            ChatResponse resp = JsonUtility.FromJson<ChatResponse>(req.downloadHandler.text);
            sessionId = resp.session_id; // reuse on the next call to keep memory

            Debug.Log($"You said: {resp.user_transcript}");
            Debug.Log($"{characterName}: {resp.character_transcript}");

            StartCoroutine(PlayAudio($"{baseUrl}{resp.voice_audio_url}"));
            if (!string.IsNullOrEmpty(resp.talking_video_url))
                StartCoroutine(PlayVideo($"{baseUrl}{resp.talking_video_url}"));
        }
    }

    private IEnumerator PlayAudio(string url)
    {
        using (UnityWebRequest req = UnityWebRequestMultimedia.GetAudioClip(url, AudioType.WAV))
        {
            yield return req.SendWebRequest();
            if (req.result == UnityWebRequest.Result.Success)
            {
                AudioClip clip = DownloadHandlerAudioClip.GetContent(req);
                // Assign to an AudioSource and Play() here.
            }
        }
    }

    private IEnumerator PlayVideo(string url)
    {
        // Assign `url` to a VideoPlayer component's url field and Play().
        yield return null;
    }
}