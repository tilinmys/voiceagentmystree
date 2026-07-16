import { NextRequest, NextResponse } from "next/server";

const WORKER_STATUS_URL = process.env.WORKER_STATUS_URL || "http://localhost:8080";

export async function GET(req: NextRequest) {
  const since = req.nextUrl.searchParams.get("since") || "0";
  try {
    const res = await fetch(`${WORKER_STATUS_URL}/slot-events?since=${encodeURIComponent(since)}`, {
      cache: "no-store",
    });
    return NextResponse.json(await res.json(), { status: res.status });
  } catch (error: any) {
    return NextResponse.json(
      { error: `Worker unreachable: ${error.message}`, next: Number(since) || 0, events: [] },
      { status: 502 }
    );
  }
}
