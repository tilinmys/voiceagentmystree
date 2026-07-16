import { NextResponse } from "next/server";

const WORKER_STATUS_URL = process.env.WORKER_STATUS_URL || "http://localhost:8080";

// Proxies to status_server.py's /health route - whether the worker process
// is actually registered with LiveKit Cloud right now, not just running.
export async function GET() {
  try {
    const res = await fetch(`${WORKER_STATUS_URL}/health`, { cache: "no-store" });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error: any) {
    return NextResponse.json(
      { ready: false, reason: `Could not reach worker status server at ${WORKER_STATUS_URL}: ${error.message}` },
      { status: 502 }
    );
  }
}
