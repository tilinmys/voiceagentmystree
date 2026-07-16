import { NextResponse } from "next/server";

const WORKER_STATUS_URL = process.env.WORKER_STATUS_URL || "http://localhost:8080";

// Proxies to status_server.py's /config route (STT/LLM provider+model summary
// and the TTS voice catalog) so the browser never needs the worker's host
// directly - same proxy pattern as /api/logs and /api/worker-status.
export async function GET() {
  try {
    const res = await fetch(`${WORKER_STATUS_URL}/config`, { cache: "no-store" });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error: any) {
    return NextResponse.json(
      { error: `Could not reach worker status server at ${WORKER_STATUS_URL}: ${error.message}` },
      { status: 502 }
    );
  }
}
