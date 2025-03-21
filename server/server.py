import base64
import logging

import uvicorn
from auralis import TTS, TTSRequest
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI()
tts = (TTS(
    scheduler_max_concurrency=8,
    vllm_logging_level=logging.WARN)
.from_pretrained(
    "AstraMindAI/xttsv2",
    gpt_model="AstraMindAI/xtts2-gpt")
)

speaker = "server/speakers/alloy.wav"


class OpenAiRequest(BaseModel):
    # OpenAi параметры
    input: str
    # Auralis параметры
    stream: bool = False
    sample_rate: int = 8000
    voice_b64: str | None = None,
    enhance_speech: bool | None = None


@app.post("/v1/audio/speech")
async def generate_speech(request: OpenAiRequest):
    speaker_file = speaker
    if request.voice_b64 is not None:
        validate_request(request)
        speaker_file = base64.b64decode(request.voice_b64)
    tts_request = TTSRequest(
        text=request.input,
        stream=request.stream,
        speaker_files=[speaker_file],
        enhance_speech=request.enhance_speech
    )
    try:
        if request.stream:
            stream = await tts.generate_speech_async(tts_request)

            async def audio_stream():
                async for chunk in stream:
                    yield chunk.to_bytes()

            return StreamingResponse(audio_stream(), media_type="audio/wav")
        else:
            output = await tts.generate_speech_async(tts_request)
            output.resample(request.sample_rate)
            return Response(content=output.to_bytes(), media_type=f"audio/wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error generating audio: {str(e)}"})


def validate_request(request: OpenAiRequest):
    if request.enhance_speech is None:
        raise ValueError("Не указано поле 'enhance_speech'")
    try:
        base64.b64decode(request.voice_b64, validate=True)
    except Exception:
        raise ValueError(f"Ошибка в кодировке base64")


def main():
    if tts is None:
        raise ValueError("TTS engine not initialized")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
