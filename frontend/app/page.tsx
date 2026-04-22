'use client';
import { useState, useRef, useEffect, useCallback } from 'react';
import {
  Mic, SendHorizonal, Paperclip, X, VolumeX, Volume2,
  Layers, Stethoscope, FileText, Camera, SwitchCamera, Square,
  Download, ClipboardList, Activity, User, ChevronRight, Loader2,
  CheckCircle2, Sparkles, Globe,
} from 'lucide-react';
import { useSession, signIn, signOut } from 'next-auth/react';
import { sendMessage } from '@/lib/api';
import VitalModal from '@/components/VitalModal';

// ── Types ─────────────────────────────────────────────────────────────────────

type Mode = 'text' | 'voice';
type ActiveMode = 'diagnosis' | 'report_analysis' | null;
type Language = 'en' | 'hi' | 'mr';

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

interface Message {
  id: number;
  role: 'user' | 'assistant';
  text: string;
  time: string;
  isDashboard?: boolean;
  reportData?: any;
  downloadUrl?: string;
}

interface PatientInfo {
  name: string;
  age: string;
  weight: string;
  height: string;
  id: string;
  date: string;
}

interface InputBarProps {
  inputRef: React.RefObject<HTMLInputElement | null>;
  value: string;
  loading: boolean;
  mode: Mode;
  placeholder: string;
  selectedFile: File | null;
  previewUrl: string | null;
  isListening: boolean;
  onChange: (v: string) => void;
  onFileChange: (f: File | null) => void;
  onSubmit: () => void;
  onMicToggle: () => void;
  onCancel: () => void;
  volume: number;
  onVolumeChange: (v: number) => void;
  isAtTop?: boolean;
  activeMode: ActiveMode;
  onReportAnalysisClick: () => void;
}

// ── Markdown Formatter ────────────────────────────────────────────────────────

function formatMessageText(text: string) {
  const parts = text.split(/(\*\*.*?\*\*)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={i}>{part.slice(2, -2)}</strong>;
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

// ── Multilingual title ────────────────────────────────────────────────────────

const texts = [
  'What symptoms are you experiencing?',
  'आपको कौन से लक्षण हो रहे हैं?',
  'तुम्हाला कोणती लक्षणे जाणवत आहेत?',
];

export function MultilingualTitle() {
  const [index, setIndex] = useState(0);
  const [show, setShow] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setShow(false);
      setTimeout(() => { setIndex(p => (p + 1) % texts.length); setShow(true); }, 300);
    }, 2600);
    return () => clearInterval(interval);
  }, []);

  return (
    <h1 className="title fade-text">
      {texts.map((text, i) => (
        <span
          key={i}
          className={
            i === index
              ? 'active'
              : i === (index - 1 + texts.length) % texts.length
                ? 'exit'
                : ''
          }
        >
          {text}
        </span>
      ))}
    </h1>
  );
}

// ── Patient Info Modal ────────────────────────────────────────────────────────

function PatientInfoModal({
  onClose,
  onSubmit,
  loading,
}: {
  onClose: () => void;
  onSubmit: (info: PatientInfo) => void;
  loading: boolean;
}) {
  const today = new Date().toISOString().split('T')[0];
  const [form, setForm] = useState<PatientInfo>({
    name: '',
    age: '',
    weight: '',
    height: '',
    id: '',
    date: today,
  });
  const [error, setError] = useState('');

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    setForm(p => ({ ...p, [e.target.name]: e.target.value }));
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!form.name.trim() || !form.age.trim()) {
      setError('Please fill in at least your name and age.');
      return;
    }
    const id = form.id.trim() || `PAT-${Math.random().toString(36).substring(2, 8).toUpperCase()}`;
    onSubmit({ ...form, id });
  }

  return (
    <div className="pi-overlay" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="pi-card">
        {/* Header */}
        <div className="pi-header">
          <div className="pi-header-icon">
            <User size={22} />
          </div>
          <div>
            <h2 className="pi-title">Patient Information</h2>
            <p className="pi-subtitle">Needed to generate your medical report</p>
          </div>
          <button className="pi-close" onClick={onClose}><X size={20} /></button>
        </div>

        <form onSubmit={handleSubmit} className="pi-form">
          <div className="pi-row">
            <div className="pi-field pi-field--full">
              <label className="pi-label">Full Name *</label>
              <input className="pi-input" name="name" value={form.name} onChange={handleChange} placeholder="e.g. John Doe" autoFocus />
            </div>
          </div>

          <div className="pi-row">
            <div className="pi-field">
              <label className="pi-label">Age *</label>
              <input className="pi-input" name="age" value={form.age} onChange={handleChange} placeholder="e.g. 34" type="number" min="1" max="120" />
            </div>
            <div className="pi-field">
              <label className="pi-label">Weight</label>
              <input className="pi-input" name="weight" value={form.weight} onChange={handleChange} placeholder="e.g. 70 kg" />
            </div>
            <div className="pi-field">
              <label className="pi-label">Height</label>
              <input className="pi-input" name="height" value={form.height} onChange={handleChange} placeholder={'e.g. 5\'8"'} />
            </div>
          </div>

          <div className="pi-row">
            <div className="pi-field">
              <label className="pi-label">Patient ID <span className="pi-opt">(auto-generated if blank)</span></label>
              <input className="pi-input" name="id" value={form.id} onChange={handleChange} placeholder="e.g. PAT-001" />
            </div>
            <div className="pi-field">
              <label className="pi-label">Date</label>
              <input className="pi-input" name="date" value={form.date} onChange={handleChange} type="date" />
            </div>
          </div>

          {error && <p className="pi-error">{error}</p>}

          <div className="pi-actions">
            <button type="button" className="pi-btn pi-btn--secondary" onClick={onClose} disabled={loading}>
              Cancel
            </button>
            <button type="submit" className="pi-btn pi-btn--primary" disabled={loading}>
              {loading
                ? <><Loader2 size={16} className="pi-spinner" /> Generating…</>
                : <><Sparkles size={16} /> Generate Report</>}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ── Report Dashboard card (inline in chat) ────────────────────────────────────

function ReportDashboard({ data, downloadUrl }: { data: any; downloadUrl: string }) {
  const ps = data.patient_summary || {};
  const cs = data.clinical_support || {};

  function handleDownload() {
    window.open(downloadUrl, '_blank');
  }

  return (
    <div className="rdb">
      {/* ── Header ── */}
      <div className="rdb-header">
        <h2 className="rdb-title">{data.name || 'Patient'}'s Health Report</h2>
        <button className="rdb-dl-pill" onClick={handleDownload}>
          <Download size={14} /> Download PDF
        </button>
      </div>

      <div className="rdb-divider" />

      {/* ── Two-column grid ── */}
      <div className="rdb-grid">

        {/* LEFT — Clinical Support */}
        <div className="rdb-section">
          <p className="rdb-section-label">CLINICAL SUPPORT</p>

          {cs.indicators && (
            <div className="rdb-cs-block">
              <p className="rdb-cs-text">{cs.indicators}</p>
            </div>
          )}

          {cs.focus?.length > 0 && (
            <div className="rdb-cs-focus">
              {cs.focus.slice(0, 4).map((f: string, i: number) => (
                <div key={i} className="rdb-cs-focus-item">
                  <span className="rdb-cs-bullet">›</span>
                  <span>{f}</span>
                </div>
              ))}
            </div>
          )}

          {/* Chief complaint + duration below */}
          {ps.chief_complaint && (
            <div className="rdb-ps-row">
              <span className="rdb-ps-label">Chief Complaint:</span>
              <span className="rdb-ps-val">{ps.chief_complaint}</span>
            </div>
          )}
          {ps.duration_onset && (
            <div className="rdb-ps-row">
              <span className="rdb-ps-label">Duration:</span>
              <span className="rdb-ps-val">{ps.duration_onset}</span>
            </div>
          )}

          {/* Symptom pills */}
          {ps.symptoms_present?.length > 0 && (
            <div className="rdb-symp-box">
              <span className="rdb-symp-label">Present:</span>
              {ps.symptoms_present.slice(0, 5).join(', ')}{ps.symptoms_present.length > 5 ? '…' : ''}
            </div>
          )}
          {ps.symptoms_absent?.length > 0 && (
            <div className="rdb-symp-box rdb-symp-box--absent">
              <span className="rdb-symp-label">Absent:</span>
              {ps.symptoms_absent.slice(0, 4).join(', ')}{ps.symptoms_absent.length > 4 ? '…' : ''}
            </div>
          )}
        </div>

        {/* RIGHT — Vitals Overview */}
        <div className="rdb-section">
          <p className="rdb-section-label">VITALS OVERVIEW</p>
          <div className="rdb-vitals-grid">
            {(data.vitals?.length > 0 ? data.vitals : []).slice(0, 4).map((v: any, i: number) => {
              const isGreen = v.status?.includes('Green');
              const isAmber = v.status?.includes('Amber');
              return (
                <div
                  key={i}
                  className={`rdb-vital-card ${isGreen ? 'rdb-vital-card--green'
                    : isAmber ? 'rdb-vital-card--amber'
                      : 'rdb-vital-card--red'
                    }`}
                >
                  <span className="rdb-vital-card-name">{v.name}</span>
                  <span className="rdb-vital-card-value">{v.value}</span>
                </div>
              );
            })}
          </div>
        </div>

      </div>
    </div>
  );
}

// ── Report Upload Modal (for Report Analysis) ─────────────────────────────────

function ReportUploadModal({
  onClose,
  onAnalyzed,
}: {
  onClose: () => void;
  onAnalyzed: (summary: string) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [step, setStep] = useState<'choose' | 'uploading' | 'done'>('choose');
  const [cameraOpen, setCameraOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('environment');

  useEffect(() => {
    if (!cameraOpen) return;
    let mounted = true;
    async function start() {
      try {
        const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode } });
        if (!mounted) { s.getTracks().forEach(t => t.stop()); return; }
        streamRef.current = s;
        if (videoRef.current) videoRef.current.srcObject = s;
      } catch (err: any) {
        setError(`Camera error: ${err.message}`);
        setCameraOpen(false);
      }
    }
    start();
    return () => {
      mounted = false;
      streamRef.current?.getTracks().forEach(t => t.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
    };
  }, [cameraOpen, facingMode]);

  function capturePhoto() {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
      if (!blob) return;
      const file = new File([blob], 'report-photo.jpg', { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      setPreview(url);
      setFileName('report-photo.jpg');
      setCameraOpen(false);
      uploadFile(file);
    }, 'image/jpeg', 0.92);
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';
    const url = file.type.startsWith('image/') ? URL.createObjectURL(file) : null;
    setPreview(url);
    setFileName(file.name);
    uploadFile(file);
  }

  async function uploadFile(file: File) {
    setStep('uploading');
    setError(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/analyze_report`, {
        method: 'POST',
        body: form,
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setStep('done');
      onAnalyzed(data.summary);
    } catch (err: any) {
      setError(err.message || 'Analysis failed.');
      setStep('choose');
    }
  }

  return (
    <div className="report-modal-overlay" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      {cameraOpen && (
        <div className="report-camera-fullscreen">
          <video ref={videoRef} autoPlay playsInline className="report-camera-video-fs" />
          <div className="report-camera-controls-fs">
            <button className="report-cam-btn" onClick={() => setFacingMode(m => m === 'user' ? 'environment' : 'user')}>
              <SwitchCamera size={22} />
            </button>
            <button className="report-cam-capture-fs" onClick={capturePhoto} />
            <button className="report-cam-btn" onClick={() => setCameraOpen(false)}>
              <X size={22} />
            </button>
          </div>
        </div>
      )}

      <div className="report-modal-card" style={{ display: cameraOpen ? 'none' : undefined }}>
        <div className="report-modal-header">
          <div className="report-modal-icon"><FileText size={22} /></div>
          <h2 className="report-modal-title">Report Analysis</h2>
          <button className="report-modal-close" onClick={onClose}><X size={20} /></button>
        </div>

        {step === 'choose' && (
          <div className="report-modal-body">
            <p className="report-modal-hint">Choose how to share your report:</p>
            <div className="report-upload-options">
              <button className="report-upload-option" onClick={() => fileInputRef.current?.click()}>
                <div className="report-option-icon report-option-icon--blue"><FileText size={28} /></div>
                <span className="report-option-label">Upload PDF or Image</span>
                <span className="report-option-sub">From your device</span>
              </button>
              <button className="report-upload-option" onClick={() => setCameraOpen(true)}>
                <div className="report-option-icon report-option-icon--teal"><Camera size={28} /></div>
                <span className="report-option-label">Take a Photo</span>
                <span className="report-option-sub">Use your camera</span>
              </button>
            </div>
            {error && <p className="report-modal-error">{error}</p>}
            <input ref={fileInputRef} type="file" accept="image/*,.pdf" className="sr-only" onChange={handleFileChange} />
          </div>
        )}

        {step === 'uploading' && (
          <div className="report-modal-body report-modal-body--center">
            <div className="report-analyzing-anim">
              <div className="report-scan-ring" />
              <FileText size={36} className="report-scan-icon" />
            </div>
            {preview && <img src={preview} alt="preview" className="report-preview-thumb" />}
            {fileName && !preview && <p className="report-file-name">{fileName}</p>}
            <p className="report-analyzing-text">Analysing your report…</p>
            <p className="report-analyzing-sub">Extracting medical information and preparing a simple summary</p>
          </div>
        )}

        {step === 'done' && (
          <div className="report-modal-body report-modal-body--center">
            <div className="report-done-icon"><CheckCircle2 size={36} /></div>
            <p className="report-analyzing-text">Analysis complete!</p>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Input Bar ─────────────────────────────────────────────────────────────────

function InputBar({
  inputRef, value, loading, placeholder,
  selectedFile, previewUrl, isListening, onChange, onFileChange, onSubmit, onMicToggle, onCancel,
  volume, onVolumeChange, isAtTop,
  activeMode, onReportAnalysisClick,
}: InputBarProps) {
  const [toolsOpen, setToolsOpen] = useState(false);
  const toolsRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleInternalFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFileChange(file);
    e.target.value = '';
  };

  useEffect(() => {
    function onOutside(e: MouseEvent) {
      if (toolsRef.current && !toolsRef.current.contains(e.target as Node)) setToolsOpen(false);
    }
    document.addEventListener('mousedown', onOutside);
    return () => document.removeEventListener('mousedown', onOutside);
  }, []);

  return (
    <div className={`input-group-wrapper ${isAtTop ? 'at-top' : ''}`} style={{ width: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="input-row">
        <div className={`searchBox ${loading ? 'searchBox--loading' : ''}`}>
          {selectedFile && (
            <div className="file-preview-strip">
              <div className="preview-thumb">
                {previewUrl ? <img src={previewUrl} alt="preview" /> : <Paperclip size={14} />}
              </div>
              <span className="preview-name">{selectedFile.name}</span>
              <button className="preview-remove" onClick={() => onFileChange(null)}>
                <X size={14} />
              </button>
            </div>
          )}
          <input
            ref={inputRef}
            type="text"
            placeholder={placeholder}
            className="input"
            value={value}
            onChange={e => onChange(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && onSubmit()}
          />
          <div className="input-actions">
            {(value || selectedFile || loading) && (
              <button className="clear-btn" onClick={onCancel} title="Clear/Cancel">
                <X size={18} />
              </button>
            )}
            {loading ? (
              <span className="loader" />
            ) : (
              <button
                className="send"
                style={{ opacity: (value.trim() || selectedFile) ? 1 : 0.35 }}
                onClick={onSubmit}
                disabled={!(value.trim() || selectedFile)}
              >
                <SendHorizonal size={22} />
              </button>
            )}
          </div>
        </div>

        <div className="extra-icons">
          <button
            className={`circular-btn mic-btn ${isListening ? 'text-red-500' : ''}`}
            onClick={onMicToggle}
            title={isListening ? 'Stop listening' : 'Microphone'}
            style={{ color: isListening ? '#e74c3c' : 'inherit' }}
          >
            {isListening ? <Square size={20} /> : <Mic size={20} />}
          </button>

          <div className="stack-wrap" ref={toolsRef}>
            <button
              className={`circular-btn stack-btn ${toolsOpen ? 'stack-btn--active' : ''}`}
              onClick={() => setToolsOpen(o => !o)}
              title="Menu"
            >
              <Layers size={20} />
            </button>
            {toolsOpen && (
              <div className="stack-popover">
                <button className="stack-item" onClick={() => setToolsOpen(false)}>
                  <Stethoscope size={18} /> Diagnosis
                </button>
                <button
                  className={`stack-item ${activeMode === 'report_analysis' ? 'stack-item--active' : ''}`}
                  onClick={() => { onReportAnalysisClick(); setToolsOpen(false); }}
                >
                  <FileText size={18} /> Report Analysis
                </button>
                <div className="stack-vol">
                  <div className="stack-vol-header">
                    {volume === 0 ? <VolumeX size={18} /> : <Volume2 size={18} />}
                    <span className="vol-label">{volume}</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={volume}
                    className="vol-slider-horizontal"
                    style={{ '--val': `${volume}%` } as React.CSSProperties}
                    onChange={e => onVolumeChange(Number(e.target.value))}
                  />
                </div>
              </div>
            )}
            <input ref={fileInputRef} type="file" accept="*/*" className="sr-only" onChange={handleInternalFileChange} />
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Confidence progress bar ───────────────────────────────────────────────────

function ConfidenceBar({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  return (
    <div className="conf-bar-wrap">
      <div className="conf-bar-track">
        <div className="conf-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="conf-bar-label">AI confidence: {pct}%</span>
    </div>
  );
}

// ── Generate Report CTA ───────────────────────────────────────────────────────

function GenerateReportCTA({ onGenerate, loading }: { onGenerate: () => void; loading: boolean }) {
  return (
    <div className="gen-cta-wrap">
      <div className="gen-cta-card">
        <CheckCircle2 size={28} className="gen-cta-icon" />
        <div className="gen-cta-text">
          <p className="gen-cta-title">Assessment complete</p>
          <p className="gen-cta-sub">Your intake is done. Generate a full clinical report now.</p>
        </div>
        <button className="gen-cta-btn" onClick={onGenerate} disabled={loading}>
          {loading
            ? <><Loader2 size={16} className="gen-cta-spinner" /> Generating…</>
            : <><Sparkles size={16} /> Generate Report <ChevronRight size={16} /></>}
        </button>
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function Home() {
  const { data: session } = useSession();

  // Chat state
  const [value, setValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const sessionIdRef = useRef<string | undefined>(undefined);
  const [isDone, setIsDone] = useState(false);
  const [confidence, setConfidence] = useState(0);

  // Language state
  const [language, setLanguage] = useState<Language>('en');
  const languageRef = useRef<Language>('en');
  const [langDropdownOpen, setLangDropdownOpen] = useState(false);

  // UI state
  const [profileOpen, setProfileOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [activeMode, setActiveMode] = useState<ActiveMode>(null);
  const [reportModalOpen, setReportModalOpen] = useState(false);
  const [vitalModalOpen, setVitalModalOpen] = useState(false);
  const [patientInfoOpen, setPatientInfoOpen] = useState(false);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [reportGenerated, setReportGenerated] = useState(false);

  // Vitals — mandatory after HoD completes
  const [storedVitals, setStoredVitals] = useState<VitalsResult | null>(null);
  const [vitalsCollected, setVitalsCollected] = useState(false);
  const [showVitalsCTA, setShowVitalsCTA] = useState(false);

  // Voice
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [handsFree, setHandsFree] = useState(false);
  const [volume, setVolume] = useState(80);
  const [interimTranscript, setInterimTranscript] = useState('');
  const recognitionRef = useRef<any>(null);
  const handsFreeRef = useRef(false);
  const vitalModalOpenRef = useRef(false);

  const scrollAnchorRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const hasMessages = messages.length > 0;
  // Show Generate Report only after BOTH: HoD done AND vitals collected
  const showReportCTA = isDone && vitalsCollected && !generatingReport && !reportGenerated;

  // ── Keep refs in sync with state (avoids stale closures in voice callbacks) ──
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);
  useEffect(() => { languageRef.current = language; }, [language]);
  useEffect(() => { vitalModalOpenRef.current = vitalModalOpen; }, [vitalModalOpen]);

  useEffect(() => {
    if (!vitalModalOpen) return;
    recognitionRef.current?.abort();
    window.speechSynthesis.cancel();
    setIsListening(false);
    setIsSpeaking(false);
    setInterimTranscript('');
  }, [vitalModalOpen]);

  // ── Vitals trigger ───────────────────────────────────────────────────────────

  const VITALS_TRIGGERS = [
    'check my vital', 'check vitals', 'vital signs', 'check my vitals',
    'measure my heart', 'check my heart', 'scan my vitals', 'health scan',
    'vitals check', 'check pulse', 'check my pulse',
  ];
  function isVitalsTrigger(text: string) {
    return VITALS_TRIGGERS.some(t => text.toLowerCase().includes(t));
  }

  // ── Scroll ────────────────────────────────────────────────────────────────────

  useEffect(() => {
    scrollAnchorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, loading]);

  // ── TTS ──────────────────────────────────────────────────────────────────────

  const LANG_LABELS: Record<Language, string> = { en: 'EN', hi: 'हिं', mr: 'मर' };
  const LANG_NAMES: Record<Language, string> = { en: 'English', hi: 'हिन्दी', mr: 'मराठी' };
  const LANG_STT_CODES: Record<Language, string> = { en: 'en-IN', hi: 'hi-IN', mr: 'hi-IN' };

  function speak(text: string) {
    if (vitalModalOpenRef.current) return;
    window.speechSynthesis.cancel();
    let lang = LANG_STT_CODES[languageRef.current] || 'en-IN';
    const utterance = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();

    let matched = voices.find(v => v.lang === lang) || voices.find(v => v.lang.startsWith(lang.split('-')[0]));

    // Mac OS lacks Marathi TTS by default. The Hindi TTS engine accurately reads Marathi Devanagari text.
    if (!matched && lang.startsWith('mr')) {
      matched = voices.find(v => v.lang === 'hi-IN') || voices.find(v => v.lang.startsWith('hi'));
      if (matched) lang = matched.lang;
    }

    if (!matched) matched = voices[0];

    if (matched) utterance.voice = matched;
    utterance.lang = lang;
    utterance.rate = 1;
    utterance.volume = volume / 100;
    utterance.onstart = () => { setIsSpeaking(true); recognitionRef.current?.abort(); setIsListening(false); };
    utterance.onend = () => {
      setIsSpeaking(false);
      if (handsFreeRef.current && !vitalModalOpenRef.current) startMic();
    };
    utterance.onerror = () => {
      setIsSpeaking(false);
      if (handsFreeRef.current && !vitalModalOpenRef.current) startMic();
    };
    window.speechSynthesis.speak(utterance);
  }

  // ── STT ──────────────────────────────────────────────────────────────────────

  function startMic() {
    if (vitalModalOpenRef.current) return;
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) { alert('Speech Recognition not supported in this browser'); return; }
    const recognition = new SpeechRecognition();
    recognition.lang = LANG_STT_CODES[languageRef.current];
    recognition.continuous = false;
    recognition.interimResults = true;

    // Track whether we already processed a final result (prevents double-submit)
    let didFinalize = false;
    let pendingInterim = '';
    let silenceTimer: NodeJS.Timeout | null = null;

    async function handleFinalText(text: string) {
      if (silenceTimer) clearTimeout(silenceTimer);
      if (didFinalize || !text.trim()) return;
      didFinalize = true;
      setInterimTranscript(text);
      setValue(text);
      try { recognition.abort(); } catch (_) { }
      setIsListening(false);
      setInterimTranscript('');
      if (isVitalsTrigger(text)) {
        handsFreeRef.current = false; setHandsFree(false);
        window.speechSynthesis.cancel();
        setVitalModalOpen(true);
        return;
      }
      await submitMessage(text, null);
    }

    recognition.onstart = () => setIsListening(true);
    recognition.onresult = async (event: any) => {
      let interim = '', final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) final += t; else interim += t;
      }
      if (interim) {
        pendingInterim = interim;
        setInterimTranscript(interim);

        // Reset the silence timer on each interim result for Hindi/Marathi
        if (silenceTimer) clearTimeout(silenceTimer);
        const currentLang = languageRef.current;
        if (currentLang === 'hi' || currentLang === 'mr') {
          silenceTimer = setTimeout(() => {
            if (!didFinalize && pendingInterim.trim()) {
              try { recognition.abort(); } catch (_) { } // abort triggers onend which calls handleFinalText
            }
          }, 2500); // 2.5s silence triggers abort
        }
      }
      if (final) {
        pendingInterim = '';
        await handleFinalText(final);
      }
    };
    recognition.onerror = (event: any) => {
      if (silenceTimer) clearTimeout(silenceTimer);
      if (event.error === 'not-allowed') alert('Microphone permission denied.');
      else if (event.error === 'audio-capture') alert('No microphone found.');
      setIsListening(false);
      if (handsFreeRef.current && event.error === 'no-speech') startMic();
    };
    recognition.onend = async () => {
      if (silenceTimer) clearTimeout(silenceTimer);
      // For Hindi/Marathi: Chrome often doesn't fire isFinal — use last interim instead
      if (!didFinalize && pendingInterim.trim()) {
        await handleFinalText(pendingInterim);
        return; // handleFinalText will re-trigger mic via speak → onend if hands-free
      }
      setIsListening(false);
      if (handsFreeRef.current && !vitalModalOpenRef.current && !window.speechSynthesis.speaking) startMic();
    };
    recognitionRef.current = recognition;
    recognition.start();
  }

  function toggleMic() {
    if (handsFree) {
      handsFreeRef.current = false; setHandsFree(false);
      recognitionRef.current?.abort(); window.speechSynthesis.cancel();
      setIsListening(false); setIsSpeaking(false);
    } else {
      handsFreeRef.current = true; setHandsFree(true);
      startMic();
    }
  }

  // ── File handling ────────────────────────────────────────────────────────────

  const handleFileChange = (file: File | null) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(file.type.startsWith('image/') ? URL.createObjectURL(file) : null);
    } else {
      setSelectedFile(null);
      setPreviewUrl(null);
    }
  };

  // ── Submit message ────────────────────────────────────────────────────────────

  async function submitMessage(textToSend: string, attachedFile: File | null) {
    if (!textToSend.trim() && !attachedFile) return;
    if (loading) { abortControllerRef.current?.abort(); window.speechSynthesis.cancel(); }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const userMsg: Message = {
      id: Date.now(),
      role: 'user',
      text: textToSend.trim() || (attachedFile ? `Attached file: ${attachedFile.name}` : ''),
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    setMessages(prev => [...prev, userMsg]);
    handleFileChange(null);
    setLoading(true);

    try {
      const res = await sendMessage(userMsg.text, sessionIdRef.current, controller.signal, languageRef.current);
      if (res.sessionId) {
        setSessionId(res.sessionId);
        sessionIdRef.current = res.sessionId;
      }
      if (res.done && !isDone) {
        setIsDone(true);
        // Vitals are mandatory — auto-trigger the scan prompt
        setShowVitalsCTA(true);
      }
      setConfidence(res.confidence);
      setInterimTranscript('');
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        text: res.reply,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }]);
      speak(res.reply);
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        text: `Error: ${err.message}`,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }]);
    } finally {
      if (!controller.signal.aborted) { setLoading(false); inputRef.current?.focus(); }
    }
  }

  function handleSubmit() { submitMessage(value, selectedFile); setValue(''); }
  function handleCancel() {
    abortControllerRef.current?.abort();
    window.speechSynthesis.cancel();
    setValue('');
    handleFileChange(null);
    setLoading(false);
  }

  // ── Vitals completion (mandatory step) ───────────────────────────────────────

  function handleVitalsComplete(summary: string, vitals: VitalsResult) {
    setStoredVitals(vitals);
    setVitalsCollected(true);
    setShowVitalsCTA(false);
    const userMsg: Message = {
      id: Date.now(), role: 'user',
      text: 'Vital signs scan complete',
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    const assistantMsg: Message = {
      id: Date.now() + 1, role: 'assistant',
      text: [
        '**Biometric Scan Results**',
        `• Heart Rate: ${vitals.heart_rate ?? '—'} BPM`,
        `• HRV (RMSSD): ${vitals.hrv_rmssd ?? '—'} ms`,
        `• Est. SpO₂: ${vitals.spo2_est ?? '—'}%`,
        `• Signal Quality: ${vitals.ppg_quality ?? '—'}`,
        `• Breathing: ${(vitals.respiratory_class ?? '—').replace(/_/g, ' ')}`,
        '',
        summary,
        '',
        'Vitals collected. You can now generate your full clinical report.',
      ].join('\n'),
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    setMessages(prev => [...prev, userMsg, assistantMsg]);
    speak('Vitals collected. You can now generate your report.');
  }

  // ── Report Analysis ───────────────────────────────────────────────────────────

  function handleReportAnalyzed(summary: string) {
    setReportModalOpen(false);
    const userMsg: Message = { id: Date.now(), role: 'user', text: 'I uploaded my medical report for analysis.', time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
    const assistantMsg: Message = { id: Date.now() + 1, role: 'assistant', text: summary, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
    setMessages(prev => [...prev, userMsg, assistantMsg]);
    speak(summary);
    setActiveMode(null);
  }

  // ── Build vitals list for the report from stored VitalsResult ─────────────────

  function buildVitalsList(vitals: VitalsResult | null): any[] {
    if (!vitals) return [];
    const items = [
      { name: 'Heart Rate', value: vitals.heart_rate != null ? `${vitals.heart_rate} BPM` : '—', status: !vitals.heart_rate ? 'Unknown' : vitals.heart_rate < 60 || vitals.heart_rate > 100 ? 'Amber — Outside normal range' : 'Green — Normal' },
      { name: 'HRV (RMSSD)', value: vitals.hrv_rmssd != null ? `${vitals.hrv_rmssd} ms` : '—', status: !vitals.hrv_rmssd ? 'Unknown' : vitals.hrv_rmssd < 20 ? 'Amber — Low HRV' : 'Green — Healthy' },
      { name: 'Est. SpO₂', value: vitals.spo2_est != null ? `${vitals.spo2_est}%` : '—', status: !vitals.spo2_est ? 'Unknown' : vitals.spo2_est < 95 ? 'Amber — Below normal' : 'Green — Normal' },
      { name: 'Signal Quality', value: vitals.ppg_quality || '—', status: vitals.ppg_quality === 'good' ? 'Green — Good' : vitals.ppg_quality === 'fair' ? 'Amber — Fair' : 'Red — Poor' },
      { name: 'Breathing Pattern', value: (vitals.respiratory_class || '—').replace(/_/g, ' '), status: vitals.respiratory_class === 'normal_breathing' ? 'Green — Normal' : vitals.respiratory_class ? 'Amber — Abnormal pattern' : 'Unknown' },
    ];
    return items;
  }

  // ── Generate report flow ──────────────────────────────────────────────────────

  async function handleGenerateReport(info: PatientInfo) {
    setPatientInfoOpen(false);
    setGeneratingReport(true);
    setLoading(true);

    try {
      if (!sessionId) throw new Error('No active session — please chat first.');

      // Step 1: PredDoc — generate structured report
      const reportRes = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/session/${sessionId}/report`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ...info,
            vitals: buildVitalsList(storedVitals),
            language: languageRef.current,
          }),
        }
      );
      if (!reportRes.ok) {
        const err = await reportRes.json();
        throw new Error(err.detail || 'Report generation failed');
      }
      const finalReport = await reportRes.json();

      // Step 2: PDF generation via pdf_agent/template.html
      const pdfRes = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/generate_pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(finalReport),
      });
      if (!pdfRes.ok) throw new Error('PDF generation failed');
      const pdfData = await pdfRes.json();

      // Step 3: Show dashboard in chat
      setReportGenerated(true);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        text: 'Here is your generated clinical report based on our consultation.',
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        isDashboard: true,
        reportData: pdfData.data,
        downloadUrl: `${process.env.NEXT_PUBLIC_API_URL}${pdfData.download_url}`,
      }]);
      speak('Your clinical report has been generated. You can download it now.');
    } catch (err: any) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        text: `Report generation failed: ${err.message}`,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }]);
    } finally {
      setLoading(false);
      setGeneratingReport(false);
    }
  }

  // ── Mode handlers ─────────────────────────────────────────────────────────────

  function handleReportAnalysisClick() {
    setActiveMode('report_analysis');
    setReportModalOpen(true);
  }

  const inputBarProps: InputBarProps = {
    inputRef,
    value,
    loading,
    mode: 'text',
    placeholder: handsFree ? 'Listening...' : "Whenever you're ready, tell me how you feel…",
    selectedFile,
    previewUrl,
    isListening: handsFree,
    onChange: (v) => { setValue(v); },
    onFileChange: handleFileChange,
    onSubmit: handleSubmit,
    onMicToggle: toggleMic,
    onCancel: handleCancel,
    volume,
    onVolumeChange: setVolume,
    activeMode,
    onReportAnalysisClick: handleReportAnalysisClick,
  };

  // ─────────────────────────────────────────────────────────────────────────────

  return (
    <div className="app">

      {/* Listening ripples */}
      {isListening && (
        <div className="voice-bg-anim voice-bg-anim--listening">
          {[1, 2, 3, 4, 5].map(i => <span key={i} className="voice-ripple" style={{ animationDelay: `${(i - 1) * 0.22}s` }} />)}
        </div>
      )}

      {/* Live transcript */}
      {interimTranscript && (
        <div className="transcript-box">
          <span className="transcript-dot" />
          <span className="transcript-text">{interimTranscript}</span>
        </div>
      )}

      {/* Vital Signs Modal — auto-triggered when HoD is done, or manually */}
      {vitalModalOpen && (
        <VitalModal
          onClose={() => {
            setVitalModalOpen(false);
            // If dismissed without completing during mandatory flow, keep showVitalsCTA
          }}
          onComplete={(summary, vitals) => { setVitalModalOpen(false); handleVitalsComplete(summary, vitals); }}
        />
      )}

      {/* Report Analysis Modal */}
      {reportModalOpen && (
        <ReportUploadModal
          onClose={() => { setReportModalOpen(false); setActiveMode(null); }}
          onAnalyzed={handleReportAnalyzed}
        />
      )}

      {/* Patient Info Modal */}
      {patientInfoOpen && (
        <PatientInfoModal
          onClose={() => setPatientInfoOpen(false)}
          onSubmit={handleGenerateReport}
          loading={generatingReport}
        />
      )}

      {/* Top-left Logo */}
      <div className="brand-badge" aria-label="1stDoctor">
        <div className="brand-mark">
          <span className="brand-mark__digit">1</span>
          <span className="brand-mark__sup">st</span>
          <span className="brand-mark__word">Doctor</span>
        </div>
      </div>

      {/* Top-right language picker + profile */}
      <div className="profile-corner">
        {/* Language Picker */}
        <div className="lang-picker-wrap">
          <button
            className="lang-picker-btn"
            onClick={() => setLangDropdownOpen(o => !o)}
            title={`Language: ${LANG_NAMES[language]}`}
          >
            <Globe size={18} />
            <span className="lang-picker-label">{LANG_LABELS[language]}</span>
          </button>
          {langDropdownOpen && (
            <div className="lang-dropdown">
              {(['en', 'hi', 'mr'] as Language[]).map(lang => (
                <button
                  key={lang}
                  className={`lang-option ${language === lang ? 'lang-option--active' : ''}`}
                  onClick={() => { setLanguage(lang); setLangDropdownOpen(false); }}
                >
                  <span className="lang-option-code">{LANG_LABELS[lang]}</span>
                  <span className="lang-option-name">{LANG_NAMES[lang]}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <button
          className="profile-btn"
          onClick={() => session ? setProfileOpen(p => !p) : signIn('google')}
          title={session ? session.user?.name || 'Account' : 'Sign in'}
        >
          {session?.user?.image ? (
            <img src={session.user.image} alt={session.user.name || 'Profile'} className="profile-img" />
          ) : (
            <span className="profile-blank">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: '20px', height: '20px' }}>
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
              </svg>
            </span>
          )}
        </button>
        {profileOpen && session && (
          <div className="profile-dropdown">
            <p className="dropdown-name">{session.user?.name}</p>
            <p className="dropdown-email">{session.user?.email}</p>
            <hr className="dropdown-divider" />
            <button className="dropdown-signout" onClick={() => { signOut(); setProfileOpen(false); }}>Sign out</button>
          </div>
        )}
      </div>

      {/* Sticky header */}
      {hasMessages && (
        <header className="sticky-header">
          <div className="input-center">
            <InputBar {...inputBarProps} isAtTop={true} />
          </div>
        </header>
      )}

      {/* Main */}
      <main className={`main ${hasMessages ? 'main--chat' : ''}`}>
        <div className="content">

          {!hasMessages && (
            <>
              <MultilingualTitle />
              <InputBar {...inputBarProps} isAtTop={false} />
            </>
          )}

          {hasMessages && (
            <div className="ai-response-container">
              {messages.filter(m => m.role === 'assistant').map((msg, idx) => (
                <div
                  key={msg.id}
                  className={`ai-card ${msg.isDashboard ? 'ai-card--dashboard' : ''}`}
                  style={{ zIndex: idx + 1 } as React.CSSProperties}
                >
                  <div className="ai-card-content">
                    {msg.isDashboard && msg.reportData ? (
                      <ReportDashboard data={msg.reportData} downloadUrl={msg.downloadUrl!} />
                    ) : (
                      formatMessageText(msg.text)
                    )}
                  </div>
                  <span className="ai-card-time">{msg.time}</span>
                </div>
              ))}

              {/* Confidence bar */}
              {hasMessages && !isDone && confidence > 0 && (
                <ConfidenceBar confidence={confidence} />
              )}

              {/* ── Step 1: Mandatory Vitals CTA (appears when HoD done) ── */}
              {showVitalsCTA && !vitalsCollected && (
                <div className="gen-cta-wrap">
                  <div className="gen-cta-card gen-cta-card--vitals">
                    <Activity size={28} className="gen-cta-icon" />
                    <div className="gen-cta-text">
                      <p className="gen-cta-title">Vital signs required</p>
                      <p className="gen-cta-sub">The assessment is complete. Please take a 2-minute vitals scan before generating your report.</p>
                    </div>
                    <button className="gen-cta-btn" onClick={() => setVitalModalOpen(true)}>
                      Start Vitals Scan
                    </button>
                  </div>
                </div>
              )}

              {/* ── Step 2: Generate Report CTA (after vitals collected) ── */}
              {showReportCTA && (
                <GenerateReportCTA
                  onGenerate={() => setPatientInfoOpen(true)}
                  loading={generatingReport}
                />
              )}

              {/* ── Step 3: Start Again Floating Button ── */}
              {reportGenerated && (
                <div className="fixed bottom-8 right-8 z-[3000]" style={{ animation: 'slideUpModal 0.6s cubic-bezier(0.22, 1, 0.36, 1)' }}>
                  <button
                    onClick={() => window.location.reload()}
                    className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 text-white px-6 py-3.5 rounded-full shadow-[0_10px_40px_rgba(0,122,255,0.35)] transition-all duration-300 hover:-translate-y-1.5 hover:shadow-[0_15px_50px_rgba(0,122,255,0.45)]"
                  >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                      <path d="M3 3v5h5" />
                    </svg>
                    <span className="font-semibold text-[15px] tracking-wide">Start Again</span>
                  </button>
                </div>
              )}

              {loading && (
                <div className="ai-card ai-card--typing" style={{ zIndex: 999 }}>
                  <div className="bubble--typing">
                    <span /><span /><span />
                  </div>
                </div>
              )}
              <div ref={scrollAnchorRef} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
