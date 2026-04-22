'use client';
import { useEffect, useRef, useState, useCallback } from 'react';
import { X, Heart, Wind, Activity, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

// ─── Types ────────────────────────────────────────────────────────────────────

interface VitalsResult {
  heart_rate: number | null;
  hrv_rmssd: number | null;
  spo2_est: number | null;
  ppg_quality: string | null;
  respiratory_class: string | null;
  resp_confidence: number | null;
  resp_note: string | null;
  clinical_summary: string | null;
}

interface VitalModalProps {
  onClose: () => void;
  onComplete: (summary: string, vitals: VitalsResult) => void;
}

type Step = 'intro' | 'camera' | 'audio-intro' | 'audio' | 'processing' | 'results' | 'error';

const CAMERA_DURATION = 30; // seconds
const AUDIO_DURATION = 12;  // seconds

// ─── Helpers ──────────────────────────────────────────────────────────────────

function ring(value: number | null, max: number, color: string, label: string, unit: string) {
  const pct = value !== null ? Math.min(100, (value / max) * 100) : 0;
  const radius = 32;
  const circ = 2 * Math.PI * radius;
  const dash = (pct / 100) * circ;
  return (
    <div className="vm-gauge">
      <svg width="80" height="80" viewBox="0 0 80 80">
        <circle cx="40" cy="40" r={radius} stroke="#e8e8e8" strokeWidth="6" fill="none" />
        <circle
          cx="40" cy="40" r={radius}
          stroke={color} strokeWidth="6" fill="none"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 40 40)"
          style={{ transition: 'stroke-dasharray 1s ease' }}
        />
        <text x="40" y="44" textAnchor="middle" fontSize="13" fontWeight="700" fill="#1a1a1a">
          {value !== null ? value : '—'}
        </text>
      </svg>
      <span className="vm-gauge-unit">{unit}</span>
      <span className="vm-gauge-label">{label}</span>
    </div>
  );
}

const RESP_LABELS: Record<string, string> = {
  normal_breathing: 'Normal Breathing',
  dry_cough: 'Dry Cough',
  wet_cough: 'Wet Cough',
  wheezing: 'Wheezing',
  shortness_of_breath: 'Shortness of Breath',
  silent: '⬜ Silent / No Sound',
  unavailable: '— Not Available',
  unknown: '— Unknown',
  other: '🔵 Other',
};

// ─── Component ────────────────────────────────────────────────────────────────

export default function VitalModal({ onClose, onComplete }: VitalModalProps) {
  const [step, setStep] = useState<Step>('intro');
  const [countdown, setCountdown] = useState(CAMERA_DURATION);
  const [audioCountdown, setAudioCountdown] = useState(AUDIO_DURATION);
  const [vitals, setVitals] = useState<VitalsResult | null>(null);
  const [errorMsg, setErrorMsg] = useState('');

  // PPG state
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const waveCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const ppgSignalRef = useRef<number[]>([]);
  const ppgFpsRef = useRef(30);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const countdownIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Audio state
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const audioTranscriptRef = useRef('');
  const audioDurationRef = useRef(AUDIO_DURATION);
  const audioIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Waveform animation
  const waveDataRef = useRef<number[]>(Array(80).fill(0));
  const waveAnimRef = useRef<number>(0);

  // ── Cleanup ────────────────────────────────────────────────────────────────

  const stopCamera = useCallback(() => {
    if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
    if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    cancelAnimationFrame(waveAnimRef.current);
  }, []);

  const stopAudio = useCallback(() => {
    if (audioIntervalRef.current) clearInterval(audioIntervalRef.current);
    audioStreamRef.current?.getTracks().forEach(t => t.stop());
    audioStreamRef.current = null;
  }, []);

  useEffect(() => () => { stopCamera(); stopAudio(); }, [stopCamera, stopAudio]);

  // ── Draw waveform ──────────────────────────────────────────────────────────

  const drawWave = useCallback(() => {
    const canvas = waveCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const data = waveDataRef.current;
    const step = W / data.length;
    ctx.beginPath();
    ctx.strokeStyle = '#e74c3c';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    data.forEach((v, i) => {
      const x = i * step;
      const y = H / 2 - (v / 255) * (H / 2 - 4);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    waveAnimRef.current = requestAnimationFrame(drawWave);
  }, []);

  // ── Phase 1: Camera PPG ────────────────────────────────────────────────────

  const startCamera = useCallback(async () => {
    setStep('camera');
    setCountdown(CAMERA_DURATION);
    ppgSignalRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 320, height: 240, frameRate: 30 },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      ppgFpsRef.current = stream.getVideoTracks()[0].getSettings().frameRate || 30;

      // Start waveform animation
      drawWave();

      // Sample red channel every ~33ms (≈30fps)
      captureIntervalRef.current = setInterval(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || video.readyState < 2) return;

        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) return;
        ctx.drawImage(video, 0, 0, 16, 16); // tiny sample for speed
        const imageData = ctx.getImageData(0, 0, 16, 16).data;

        let redSum = 0;
        for (let i = 0; i < imageData.length; i += 4) {
          redSum += imageData[i]; // R channel
        }
        const avgRed = redSum / (16 * 16);
        ppgSignalRef.current.push(avgRed);

        // Update waveform buffer
        waveDataRef.current.shift();
        waveDataRef.current.push(avgRed);
      }, 33);

      // Countdown
      countdownIntervalRef.current = setInterval(() => {
        setCountdown(prev => {
          if (prev <= 1) {
            clearInterval(countdownIntervalRef.current!);
            clearInterval(captureIntervalRef.current!);
            cancelAnimationFrame(waveAnimRef.current);
            streamRef.current?.getTracks().forEach(t => t.stop());
            setStep('audio-intro');
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } catch (err: any) {
      setErrorMsg(`Camera error: ${err.message}`);
      setStep('error');
    }
  }, [drawWave]);

  // ── Phase 2: Audio recording ───────────────────────────────────────────────

  const startAudio = useCallback(async () => {
    setStep('audio');
    setAudioCountdown(AUDIO_DURATION);
    audioChunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioStreamRef.current = stream;

      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = e => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      recorder.start(200);

      audioIntervalRef.current = setInterval(() => {
        setAudioCountdown(prev => {
          if (prev <= 1) {
            clearInterval(audioIntervalRef.current!);
            recorder.stop();
            stream.getTracks().forEach(t => t.stop());
            processResults();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } catch (err: any) {
      // Microphone not available — skip audio, process PPG only
      processResults(true);
    }
  }, []);

  // ── Phase 3: Send to backend ───────────────────────────────────────────────

  const processResults = useCallback(async (skipAudio = false) => {
    setStep('processing');

    let transcript = '';
    let duration = AUDIO_DURATION;

    // Upload audio for Whisper transcription
    if (!skipAudio && audioChunksRef.current.length > 0) {
      try {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append('file', blob, 'respiratory.webm');

        const audioRes = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/analyze_audio`, {
          method: 'POST',
          body: formData,
        });
        const audioData = await audioRes.json();
        if (audioData.status === 'ok') {
          transcript = audioData.transcript || '';
          duration = audioData.duration || AUDIO_DURATION;
        }
      } catch (_) {
        // Silently skip audio if upload fails
      }
    }

    // Call LangGraph vitals_summary endpoint
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/vitals_summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ppg_signal: ppgSignalRef.current,
          ppg_fps: ppgFpsRef.current,
          audio_transcript: transcript,
          audio_duration: duration,
        }),
      });
      const data = await res.json();

      if (data.status === 'error') {
        setErrorMsg(data.message || 'Analysis failed');
        setStep('error');
        return;
      }

      setVitals(data);
      setStep('results');
    } catch (err: any) {
      setErrorMsg(`Backend error: ${err.message}`);
      setStep('error');
    }
  }, []);

  // ── Handle complete ────────────────────────────────────────────────────────

  const handleDone = () => {
    if (vitals) {
      onComplete(vitals.clinical_summary || 'Vital signs analysis complete.', vitals);
    }
    onClose();
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="vm-overlay" role="dialog" aria-modal="true">
      <div className="vm-modal">

        {/* Header */}
        <div className="vm-header">
          <div className="vm-header-left">
            <Activity size={20} className="vm-header-icon" />
            <span className="vm-title">Vital Signs Check</span>
          </div>
          <button className="vm-close" onClick={onClose} aria-label="Close">
            <X size={20} />
          </button>
        </div>

        {/* Step indicator */}
        <div className="vm-steps">
          {['Camera PPG', 'Breathing', 'Results'].map((label, i) => {
            const stepIdx = { intro: 0, camera: 0, 'audio-intro': 1, audio: 1, processing: 2, results: 2, error: 2 }[step] ?? 0;
            const active = i === stepIdx;
            const done = i < stepIdx;
            return (
              <div key={label} className={`vm-step ${active ? 'vm-step--active' : ''} ${done ? 'vm-step--done' : ''}`}>
                <span className="vm-step-dot">{done ? '✓' : i + 1}</span>
                <span className="vm-step-label">{label}</span>
              </div>
            );
          })}
        </div>

        {/* ─── Intro ─────────────────────────────────────────────────────── */}
        {step === 'intro' && (
          <div className="vm-body vm-body--center">
            <div className="vm-hero-icon">❤️</div>
            <h2 className="vm-h2">Ready to check your vitals?</h2>
            <p className="vm-desc">
              This 2-minute scan uses your camera and microphone to estimate heart rate, HRV, SpO₂, and respiratory health.
            </p>
            <div className="vm-checklist">
              <div className="vm-check"><span>💡</span> Sit in a well-lit room facing your camera</div>
              <div className="vm-check"><span>🤫</span> Stay still and avoid talking during the camera phase</div>
              <div className="vm-check">Results are estimates — not a clinical diagnosis</div>
            </div>
            <button className="vm-btn vm-btn--primary" onClick={startCamera}>
              Begin Scan
            </button>
          </div>
        )}

        {/* ─── Camera PPG ────────────────────────────────────────────────── */}
        {step === 'camera' && (
          <div className="vm-body">
            <div className="vm-camera-wrap">
              <video ref={videoRef} className="vm-video" playsInline muted />
              <div className="vm-camera-overlay">
                <div className="vm-face-guide" />
                <div className="vm-countdown-badge">{countdown}s</div>
              </div>
            </div>
            {/* Hidden canvas for pixel sampling */}
            <canvas ref={canvasRef} width={16} height={16} style={{ display: 'none' }} />

            <p className="vm-instruction">
              🧘 Hold still, keep your face centred, and breathe normally.
            </p>

            {/* Live PPG waveform */}
            <div className="vm-wave-wrap">
              <canvas ref={waveCanvasRef} className="vm-wave-canvas" width={340} height={60} />
            </div>

            <div className="vm-progress-bar">
              <div
                className="vm-progress-fill"
                style={{ width: `${((CAMERA_DURATION - countdown) / CAMERA_DURATION) * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* ─── Audio Intro ───────────────────────────────────────────────── */}
        {step === 'audio-intro' && (
          <div className="vm-body vm-body--center">
            <div className="vm-hero-icon"><Activity /></div>
            <h2 className="vm-h2">Camera scan complete!</h2>
            <p className="vm-desc">
              Now we'll record your breathing and cough pattern. The AI will classify your respiratory sound.
            </p>
            <div className="vm-checklist">
              <div className="vm-check"><span>1️⃣</span> Take <strong>3 deep breaths</strong> slowly</div>
              <div className="vm-check"><span>2️⃣</span> Cough naturally <strong>twice</strong></div>
              <div className="vm-check"><span>3️⃣</span> Take one more <strong>normal breath</strong></div>
              <div className="vm-check">Recording will last {AUDIO_DURATION} seconds</div>
            </div>
            <button className="vm-btn vm-btn--primary" onClick={startAudio}>
              Start Recording
            </button>
          </div>
        )}

        {/* ─── Audio Recording ───────────────────────────────────────────── */}
        {step === 'audio' && (
          <div className="vm-body vm-body--center">
            <div className="vm-record-anim">
              <span className="vm-record-dot" />
              <span className="vm-record-label">Recording…</span>
            </div>
            <p className="vm-instruction" style={{ marginTop: 8 }}>
              3 deep breaths → 2 coughs → 1 normal breath
            </p>
            <div className="vm-audio-countdown">{audioCountdown}s</div>
            <div className="vm-progress-bar" style={{ marginTop: 16 }}>
              <div
                className="vm-progress-fill vm-progress-fill--green"
                style={{ width: `${((AUDIO_DURATION - audioCountdown) / AUDIO_DURATION) * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* ─── Processing ────────────────────────────────────────────────── */}
        {step === 'processing' && (
          <div className="vm-body vm-body--center">
            <Loader2 size={48} className="vm-spinner" />
            <h2 className="vm-h2">Analysing with AI…</h2>
            <p className="vm-desc">LangGraph is processing your PPG signal and respiratory audio with Groq.</p>
          </div>
        )}

        {/* ─── Results ───────────────────────────────────────────────────── */}
        {step === 'results' && vitals && (
          <div className="vm-body">
            <div className="vm-results-title">
              <CheckCircle size={20} color="#34c759" />
              <span>Scan Complete</span>
            </div>

            {/* Gauges */}
            <div className="vm-gauges">
              {ring(vitals.heart_rate, 180, '#e74c3c', 'Heart Rate', 'BPM')}
              {ring(vitals.hrv_rmssd, 100, '#007AFF', 'HRV (RMSSD)', 'ms')}
              {ring(vitals.spo2_est, 100, '#34c759', 'Est. SpO₂', '%')}
            </div>

            {/* Signal quality badge */}
            <div className={`vm-quality-badge vm-quality--${vitals.ppg_quality}`}>
              Signal quality: {vitals.ppg_quality || '—'}
            </div>

            {/* Respiratory */}
            <div className="vm-resp-card">
              <div className="vm-resp-header">
                <Wind size={16} />
                <span>Respiratory Analysis</span>
              </div>
              <div className="vm-resp-class">
                {RESP_LABELS[vitals.respiratory_class || 'unknown'] ?? vitals.respiratory_class}
                {vitals.resp_confidence != null && (
                  <span className="vm-resp-conf">{Math.round((vitals.resp_confidence) * 100)}% confident</span>
                )}
              </div>
              {vitals.resp_note && <p className="vm-resp-note">{vitals.resp_note}</p>}
            </div>

            {/* Clinical summary */}
            {vitals.clinical_summary && (
              <div className="vm-summary-card">
                <div className="vm-summary-header">
                  <Heart size={15} />
                  <span>AI Clinical Summary</span>
                </div>
                <p className="vm-summary-text">{vitals.clinical_summary}</p>
              </div>
            )}

            <p className="vm-disclaimer">
              These are estimates from a webcam/microphone — not a medical diagnosis. Consult a doctor for clinical evaluation.
            </p>

            <button className="vm-btn vm-btn--primary" onClick={handleDone}>
              Add to Chat
            </button>
          </div>
        )}

        {/* ─── Error ─────────────────────────────────────────────────────── */}
        {step === 'error' && (
          <div className="vm-body vm-body--center">
            <AlertCircle size={48} color="#e74c3c" />
            <h2 className="vm-h2">Something went wrong</h2>
            <p className="vm-desc" style={{ color: '#e74c3c' }}>{errorMsg}</p>
            <button className="vm-btn vm-btn--secondary" onClick={onClose}>Close</button>
          </div>
        )}

      </div>
    </div>
  );
}
