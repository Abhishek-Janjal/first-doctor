export interface SendMessageResponse {
  reply: string;
  sessionId: string;
  done: boolean;
  confidence: number;
  turnCount: number;
  language?: string;
}

export async function sendMessage(
  message: string,
  sessionId?: string,
  signal?: AbortSignal,
  language?: string
): Promise<SendMessageResponse> {
  const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId, language: language || undefined }),
    signal,
  });

  if (!res.ok) {
    throw new Error(`Server error: ${res.status}`);
  }

  const data = await res.json();

  if (data.error) {
    throw new Error(data.error);
  }

  return {
    reply:      data.reply,
    sessionId:  data.session_id,
    done:       data.done ?? false,
    confidence: data.confidence ?? 0,
    turnCount:  data.turn_count ?? 0,
    language:   data.language,
  };
}