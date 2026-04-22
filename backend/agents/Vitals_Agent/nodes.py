import json
import numpy as np
from groq import Groq
from state import VitalsState
from config import settings

groq_client = Groq(api_key=settings.GROQ_API_KEY)


def ppg_interpreter(state: VitalsState) -> VitalsState:
    signal = state.get("ppg_signal")
    fps = state.get("ppg_fps") or 30.0

    if not signal or len(signal) < 60:
        return {**state, "error": "PPG signal too short."}

    sig = np.array(signal, dtype=float)
    sig = sig - np.mean(sig)

    N = len(sig)
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
    fft_vals = np.fft.rfft(sig)

    mask = (freqs >= 0.75) & (freqs <= 3.5)
    fft_filtered = fft_vals * mask
    filtered = np.fft.irfft(fft_filtered, n=N)

    power = np.abs(fft_filtered) ** 2
    if power[mask].sum() == 0:
        return {**state, "error": "No pulse detected."}

    peak_freq = freqs[mask][np.argmax(power[mask])]
    hr = round(float(peak_freq * 60), 1)

    peaks = [
        i for i in range(1, len(filtered) - 1)
        if filtered[i] > filtered[i - 1] and filtered[i] > filtered[i + 1]
    ]

    rmssd = 0.0
    if len(peaks) >= 2:
        rr = [(peaks[i+1] - peaks[i]) / fps * 1000 for i in range(len(peaks)-1)]
        diffs = np.diff(rr)
        rmssd = round(float(np.sqrt(np.mean(diffs**2))), 1) if len(diffs) > 0 else 0.0

    ac_red = float(np.std(sig))
    dc_red = float(np.mean(np.abs(signal)))
    r_ratio = ac_red / dc_red if dc_red > 0 else 0.04
    spo2 = round(max(90.0, min(100.0, 110.0 - 25.0 * r_ratio)), 1)

    snr = float(power[mask].sum() / (np.abs(fft_vals).sum() + 1e-9))
    quality = "good" if snr > 0.3 else "fair" if snr > 0.1 else "poor"

    return {
        **state,
        "heart_rate": hr,
        "hrv_rmssd": rmssd,
        "spo2_est": spo2,
        "ppg_quality": quality,
    }


def audio_classifier(state: VitalsState) -> VitalsState:
    transcript = state.get("audio_transcript", "")
    duration = state.get("audio_duration", 10.0)

    if not transcript.strip():
        return {
            **state,
            "respiratory_class": "unavailable",
            "resp_confidence": 0.0,
            "resp_note": "No transcript provided",
        }

    prompt = f"""Classify respiratory sound:
"{transcript}" ({duration}s)
Return JSON with class, confidence, note."""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        result = json.loads(resp.choices[0].message.content)
        return {
            **state,
            "respiratory_class": result["class"],
            "resp_confidence": result["confidence"],
            "resp_note": result["note"],
        }
    except Exception as e:
        return {**state, "resp_note": str(e)}


def clinical_summarizer(state: VitalsState) -> VitalsState:
    summary = (
        f"HR: {state.get('heart_rate')} BPM, "
        f"SpO₂: {state.get('spo2_est')}%, "
        f"Resp: {state.get('respiratory_class')}."
    )
    return {**state, "clinical_summary": summary}