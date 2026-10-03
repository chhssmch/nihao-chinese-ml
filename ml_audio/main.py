from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

_asr = None


def get_asr_pipeline():
    global _asr
    if _asr is None:
        _asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
        )
    return _asr


def speech_to_text(audio_path: str, language: str = "chinese") -> str:
    asr = get_asr_pipeline()

    generate_kwargs = {
        "language": language,
        "task": "transcribe",
    }

    result = asr(
        audio_path,
        generate_kwargs=generate_kwargs,
        return_timestamps=True,
    )
    
    chunks = result.get("chunks")
    if chunks:
        return "\n".join(chunk["text"].strip() for chunk in chunks)

    return result["text"].strip()