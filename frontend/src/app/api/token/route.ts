import { AccessToken } from "livekit-server-sdk";
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const room = req.nextUrl.searchParams.get("room") || "mystree-room";
  const participant = `clinic-user-${Math.random().toString(36).substring(2, 8)}`;

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
