import { AccessToken } from "livekit-server-sdk";
import { randomUUID } from "crypto";
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  // A unique room per call guarantees a fresh agent job is dispatched every time.
  // Reusing a fixed room name meant a second call could join a stale room whose
  // agent had already greeted (or was shutting down), producing a silent call.
  const room =
    req.nextUrl.searchParams.get("room") ||
    `mystree-room-${Date.now().toString(36)}-${randomUUID().replace(/-/g, "").slice(0, 12)}`;
  const participant = `clinic-user-${randomUUID().replace(/-/g, "").slice(0, 12)}`;

  const apiKey = process.env.LIVEKIT_API_KEY;
  const apiSecret = process.env.LIVEKIT_API_SECRET;
  const wsUrl = process.env.LIVEKIT_URL;

  if (!apiKey || !apiSecret || !wsUrl) {
    return NextResponse.json(
      { error: "Server misconfigured: missing LiveKit credentials" },
      { status: 500 }
    );
  }

  try {
    // Create AccessToken
    const at = new AccessToken(apiKey, apiSecret, {
      identity: participant,
      name: participant,
    });

    // Grant join, publish, and subscribe permissions
    at.addGrant({
      roomJoin: true,
      room: room,
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
    });

    const token = await at.toJwt();
    return NextResponse.json({ token, url: wsUrl, participant });
  } catch (error: any) {
    return NextResponse.json(
      { error: `Failed to generate token: ${error.message}` },
      { status: 500 }
    );
  }
}
