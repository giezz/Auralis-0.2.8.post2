import logging

import uvicorn
from auralis import TTS, TTSRequest, AudioPreprocessingConfig
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI()
tts = (TTS(
    scheduler_max_concurrency=12,
    vllm_logging_level=logging.WARN)
.from_pretrained(
    "AstraMindAI/xttsv2",
    gpt_model="AstraMindAI/xtts2-gpt")
)

speaker = "speakers/alloy.wav"


class OpenAiRequest(BaseModel):
    input: str
    stream: bool | None = None
    sample_rate: int = 8000


@app.post("/v1/audio/speech")
async def generate_speech(request: OpenAiRequest):
    tts_request = TTSRequest(
        text=request.input,
        stream=request.stream,
        speaker_files=speaker,
        audio_config=AudioPreprocessingConfig(sample_rate=request.sample_rate))
    try:
        if request.stream:
            stream = await tts.generate_speech_async(tts_request)

            async def audio_stream():
                async for chunk in stream:
                    yield chunk.to_bytes()

            return StreamingResponse(audio_stream(), media_type="audio/wav")
        else:
            output = await tts.generate_speech_async(tts_request)

            return Response(content=output.to_bytes(), media_type=f"audio/wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error generating audio: {str(e)}"})


def main():
    if tts is None:
        raise ValueError("TTS engine not initialized")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
