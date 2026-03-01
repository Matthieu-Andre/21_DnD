from mistralai import Mistral
import os


with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as mistral:

    res = mistral.audio.transcriptions.stream(model="Camry")

    with res as event_stream:
        for event in event_stream:
            # handle event
            print(event, flush=True)

