import { NextRequest, NextResponse } from "next/server";

const WORKER_STATUS_URL = process.env.WORKER_STATUS_URL || "http://localhost:8080";

export async function GET(req: NextRequest) {
  const doctor = req.nextUrl.searchParams.get("doctor") || "";
  const weekStart = req.nextUrl.searchParams.get("week_start") || "";
  try {
    const qs = new URLSearchParams({ doctor });
    if (weekStart) qs.set("week_start", weekStart);
    const res = await fetch(`${WORKER_STATUS_URL}/schedule?${qs.toString()}`, { cache: "no-store" });
    return NextResponse.json(await res.json(), { status: res.status });
  } catch (error: any) {
    return NextResponse.json({ error: `Worker unreachable: ${error.message}` }, { status: 502 });
  }
}
