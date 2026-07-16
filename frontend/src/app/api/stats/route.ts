import { NextResponse } from "next/server";

const WORKER_STATUS_URL = process.env.WORKER_STATUS_URL || "http://localhost:8080";

export async function GET() {
  try {
    const res = await fetch(`${WORKER_STATUS_URL}/stats`, { cache: "no-store" });
    return NextResponse.json(await res.json(), { status: res.status });
  } catch (error: any) {
    return NextResponse.json({ error: `Worker unreachable: ${error.message}` }, { status: 502 });
  }
}
