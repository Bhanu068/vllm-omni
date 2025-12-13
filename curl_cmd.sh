curl -X POST \
http://localhost:8000/v1/audio/speech \
-H "Content-Type: application/json" \
-d '{
    "model": "yujiepan/qwen2.5-omni-tiny-random",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "alloy",
    "response_format": "mp3",
    "speed": 1.0
   }' \
   --output music.mp3
