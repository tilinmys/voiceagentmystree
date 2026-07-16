import { AccessToken, AgentDispatchClient } from "livekit-server-sdk";
import { randomUUID } from "crypto";
import { NextRequest, NextResponse } from "next/server";

const WORKER_STATUS_URL = process.env.WORKER_STATUS_URL || "http://localhost:8080";
const AGENT_NAME = process.env.LIVEKIT_AGENT_NAME || "mystree-care";

async function fetchWorkerStatus(): Promise<{ ready: boolean; reason?: string }> {
  try {
    const res = await fetch(`${WORKER_STATUS_URL}/health`, { cache: "no-store" });
    return await res.json();
  } catch (error: any) {
    return { ready: false, reason: `Could not reach worker status server: ${error.message}` };
  }
}

export async function GET(req: NextRequest) {
  // A unique room per call guarantees a fresh agent job is dispatched every time.
  // Reusing a fixed room name meant a second call could join a stale room whose
  // agent had already greeted (or was shutting down), producing a silent call.
  const room =
    req.nextUrl.searchParams.get("room") ||
    `mystree-room-${Date.now().toString(36)}-${randomUUID().replace(/-/g, "").slice(0, 12)}`;
  const participant = `clinic-user-${randomUUID().replace(/-/g, "").slice(0, 12)}`;
  const provider = (req.nextUrl.searchParams.get("provider") || "").trim().toLowerCase();
  const voice = (req.nextUrl.searchParams.get("voice") || "").trim();
  const phone = (req.nextUrl.searchParams.get("phone") || "").trim();
  const temperature = (req.nextUrl.searchParams.get("temperature") || "").trim();
  const silenceThreshold = (req.nextUrl.searchParams.get("silenceThreshold") || "").trim();
  const ttsSpeed = (req.nextUrl.searchParams.get("ttsSpeed") || "").trim();
  const sttMinSilence = (req.nextUrl.searchParams.get("sttMinSilence") || "").trim();
  const sttMaxSilence = (req.nextUrl.searchParams.get("sttMaxSilence") || "").trim();
  const sttInterruptionDelay = (req.nextUrl.searchParams.get("sttInterruptionDelay") || "").trim();

  const apiKey = process.env.LIVEKIT_API_KEY;
  const apiSecret = process.env.LIVEKIT_API_SECRET;
  const wsUrl = process.env.LIVEKIT_URL;

  if (!apiKey || !apiSecret || !wsUrl) {
    return NextResponse.json(
      { error: "Server misconfigured: missing LiveKit credentials" },
      { status: 500 }
    );
  }

  const status = await fetchWorkerStatus();
  if (!status.ready) {
    return NextResponse.json(
      { error: "Worker not ready yet. Wait a few seconds and start the call again.", worker: status },
      { status: 503 }
    );
  }

  // Dispatch metadata is how the worker's provider_and_voice_from_metadata()
  // picks the TTS provider/voice for this specific call (see agent.py) - the
  // UI's picker only means anything if it actually reaches the worker here.
  const metadataPayload: Record<string, string> = {};
  if (provider) metadataPayload.tts_provider = provider;
  if (voice) metadataPayload.voice_id = voice;
  if (phone) metadataPayload.caller_phone = phone;
  if (temperature) metadataPayload.temperature = temperature;
  if (silenceThreshold) metadataPayload.silence_threshold = silenceThreshold;
  if (ttsSpeed) metadataPayload.tts_speed = ttsSpeed;
  if (sttMinSilence) metadataPayload.stt_min_silence = sttMinSilence;
  if (sttMaxSilence) metadataPayload.stt_max_silence = sttMaxSilence;
  if (sttInterruptionDelay) metadataPayload.stt_interruption_delay = sttInterruptionDelay;
  const metadata = Object.keys(metadataPayload).length ? JSON.stringify(metadataPayload) : undefined;

  try {
    let dispatchId: string | null = null;
    if ((process.env.LIVEKIT_EXPLICIT_DISPATCH || "true").toLowerCase() !== "false") {
      const dispatchClient = new AgentDispatchClient(wsUrl.replace(/^wss:/, "https:").replace(/^ws:/, "http:"), apiKey, apiSecret);
      const dispatch = await dispatchClient.createDispatch(room, AGENT_NAME, { metadata });
      dispatchId = dispatch.id;
    }

    const at = new AccessToken(apiKey, apiSecret, { identity: participant, name: participant });
    at.addGrant({
      roomJoin: true,
      room: room,
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
    });
    const token = await at.toJwt();

    return NextResponse.json({
      token,
      url: wsUrl,
      participant,
      provider: provider || "smallest",
      voice: voice || "default",
      dispatch_id: dispatchId,
      worker: status,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: `Failed to generate token: ${error.message}` },
      { status: 500 }
    );
  }
}
