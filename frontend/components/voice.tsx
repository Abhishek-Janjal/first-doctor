'use client';

import { useState, useRef } from 'react';
import { Mic, Square } from 'lucide-react';
import { sendMessage } from '@/lib/api';

export default function VoiceChat() {
    const [listening, setListening] = useState(false);
    const [text, setText] = useState('');
    const [response, setResponse] = useState('');
    const [sessionId, setSessionId] = useState<string | undefined>(undefined);
    const recognitionRef = useRef<any>(null);

    // Detect language (basic)
    function detectLanguage(text: string) {
        if (/[\u0900-\u097F]/.test(text)) {
            // Hindi or Marathi (Devanagari)
            return 'hi-IN'; // fallback (Marathi often not supported well)
        }
        return 'en-IN';
    }

    // Start Listening
    function startListening() {
        const SpeechRecognition =
            (window as any).SpeechRecognition ||
            (window as any).webkitSpeechRecognition;

        if (!SpeechRecognition) {
            alert('Speech Recognition not supported in this browser');
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.lang = 'en-IN'; // start default
        recognition.continuous = false;
        recognition.interimResults = false;

        recognition.onstart = () => setListening(true);

        recognition.onresult = async (event: any) => {
            const transcript = event.results[0][0].transcript;
            setText(transcript);

            const lang = detectLanguage(transcript);
            recognition.lang = lang;

            await handleAI(transcript, lang);
        };

        recognition.onerror = (err: any) => {
            console.error(err);
            setListening(false);
        };

        recognition.onend = () => setListening(false);

        recognitionRef.current = recognition;
        recognition.start();
    }

    // Stop Listening
    function stopListening() {
        recognitionRef.current?.stop();
        setListening(false);
    }

    // AI + Voice Response
    async function handleAI(message: string, lang: string) {
        try {
            const res = await sendMessage(message, sessionId);
            if (res.sessionId) {
                setSessionId(res.sessionId);
            }
            setResponse(res.reply);

            speak(res.reply, lang);
        } catch (err: any) {
            console.error(err);
        }
    }

    // Text-to-Speech
    function speak(text: string, lang: string) {
        const utterance = new SpeechSynthesisUtterance(text);

        // Try to match correct voice
        const voices = window.speechSynthesis.getVoices();

        let selectedVoice =
            voices.find(v => v.lang === lang) ||
            voices.find(v => v.lang.startsWith(lang.split('-')[0])) ||
            voices[0];

        if (selectedVoice) {
            utterance.voice = selectedVoice;
        }

        utterance.lang = lang;
        utterance.rate = 1;

        window.speechSynthesis.speak(utterance);
    }

    return (
        <div className="p-6 max-w-xl mx-auto text-center">
            <h1 className="text-xl font-semibold mb-4">
                Voice Assistant (EN / HI / MR)
            </h1>

            <div className="flex gap-4 justify-center mb-4">
                {!listening ? (
                    <button
                        onClick={startListening}
                        className="bg-green-500 text-white p-3 rounded-full"
                    >
                        <Mic />
                    </button>
                ) : (
                    <button
                        onClick={stopListening}
                        className="bg-red-500 text-white p-3 rounded-full"
                    >
                        <Square />
                    </button>
                )}
            </div>

            <p className="mb-2"><strong>You:</strong> {text}</p>
            <p><strong>AI:</strong> {response}</p>
        </div>
    );
}