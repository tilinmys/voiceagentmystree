"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  LiveKitRoom,
  RoomAudioRenderer,
  VoiceAssistantControlBar,
  useVoiceAssistant,
  BarVisualizer,
  useRoomContext
} from "@livekit/components-react";
import { Room, RoomEvent } from "livekit-client";
import "@livekit/components-styles";
import styles from "./page.module.css";

// Interface for log messages
interface LogMessage {
  time: string;
  text: string;
  category: string;
}

// Visualizer component that logs state transitions
function SimpleVisualizer({ onLog }: { onLog: (text: string, category: string) => void }) {
  const { state, audioTrack } = useVoiceAssistant();

  useEffect(() => {
    if (state) {
      onLog(`Voice Agent state: ${state}`, "agent-state");
    }
  }, [state, onLog]);

  const displayState = state.charAt(0).toUpperCase() + state.slice(1);

  return (
    <div className={styles.visualizerContainer}>
      <div className={`${styles.visualizerAnimation} ${styles[state] || ""}`}>
        <BarVisualizer trackRef={audioTrack} state={state} barCount={7} />
      </div>
      <div className={styles.stateContainer}>
        <span className={`${styles.pulseDot} ${styles[state] || ""}`}></span>
        <span className={styles.agentState}>{displayState}</span>
      </div>
    </div>
  );
}

// Room logger component that listens to WebRTC track and participant events
function RoomLogger({ onLog }: { onLog: (text: string, category: string) => void }) {
  const room = useRoomContext();

  useEffect(() => {
    if (!room) return;

    const handleParticipantConnected = (participant: any) => {
      onLog(`Participant connected: ${participant.identity}`, "event");
    };

    const handleParticipantDisconnected = (participant: any) => {
      onLog(`Participant disconnected: ${participant.identity}`, "event");
    };

    const handleTrackSubscribed = (track: any, publication: any, participant: any) => {
      onLog(`Audio stream active from participant: ${participant.identity}`, "event");
    };

    room.on(RoomEvent.ParticipantConnected, handleParticipantConnected);
    room.on(RoomEvent.ParticipantDisconnected, handleParticipantDisconnected);
    room.on(RoomEvent.TrackSubscribed, handleTrackSubscribed);

    onLog("WebRTC connection established with LiveKit room", "info");

    return () => {
      room.off(RoomEvent.ParticipantConnected, handleParticipantConnected);
      room.off(RoomEvent.ParticipantDisconnected, handleParticipantDisconnected);
      room.off(RoomEvent.TrackSubscribed, handleTrackSubscribed);
    };
  }, [room, onLog]);

  return null;
}

export default function Home() {
  const [token, setToken] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogMessage[]>([]);
  const [room, setRoom] = useState<Room | null>(null);
  const consoleEndRef = useRef<HTMLDivElement | null>(null);

  // Helper to add a log entry
  const addLog = useCallback((text: string, category = "info") => {
    const time = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit"
    });
    setLogs((prev) => [...prev, { time, text, category }]);
  }, []);

  // Scroll to bottom of log body when logs change
  useEffect(() => {
    if (consoleEndRef.current) {
      consoleEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);

  // Connection manager useEffect (handles Strict Mode double-mount cleanly)
  useEffect(() => {
    if (!token || !url) {
      setRoom(null);
      return;
    }

    addLog("Initializing LiveKit Room instance...", "info");
    const activeRoom = new Room();
    setRoom(activeRoom);

    const connectRoom = async () => {
      try {
        addLog("Connecting to LiveKit room via WebRTC...", "info");
        await activeRoom.connect(url, token);
      } catch (err: any) {
        addLog(`WebRTC connection failed: ${err.message}`, "error");
        setError(`Failed to connect: ${err.message}`);
      }
    };

    connectRoom();

    return () => {
      addLog("Disconnecting room session (cleaning up ghost connection)...", "info");
      activeRoom.disconnect();
    };
  }, [token, url, addLog]);

  const startCall = async () => {
    setConnecting(true);
    setError(null);
    setLogs([]);
    addLog("Requesting authentication token from /api/token...", "info");
    try {
      const res = await fetch("/api/token");
      const data = await res.json();
      if (data.error) {
        throw new Error(data.error);
      }
      addLog(`Token generated. Participant identity: ${data.participant}`, "info");
      addLog(`Connecting to ws: ${data.url}`, "info");
      setToken(data.token);
      setUrl(data.url);
    } catch (e: any) {
      setError(e.message || "Failed to start call");
      addLog(`Error starting call: ${e.message}`, "error");
    } finally {
      setConnecting(false);
    }
  };

  const endCall = () => {
    addLog("Voice session ended", "info");
    setToken(null);
    setUrl(null);
  };

  return (
    <main className={styles.container}>
      {token && url && room ? (
        <div className={styles.callWrapper}>
          <LiveKitRoom
            room={room}
            token={token}
            serverUrl={url}
            onDisconnected={endCall}
            data-lk-theme="default"
            className={styles.roomContainer}
          >
            <RoomLogger onLog={addLog} />
            <div className={styles.callCard}>
              <div className={styles.clinicHeader}>
                <span className={styles.statusBadge}>Live Call</span>
                <h1 className={styles.clinicName}>MyStree Clinic</h1>
                <p className={styles.subtitle}>Care Coordinator Voice Agent</p>
              </div>

              <SimpleVisualizer onLog={addLog} />

              <div className={styles.controlsWrapper}>
                <VoiceAssistantControlBar />
              </div>
            </div>
            <RoomAudioRenderer />
          </LiveKitRoom>

          {/* Side logs console panel */}
          <div className={styles.consoleCard}>
            <div className={styles.consoleHeader}>
              <h2>System Logs Console</h2>
              <span className={styles.consoleStatus}>Active</span>
            </div>
            <div className={styles.consoleBody}>
              {logs.map((log, idx) => (
                <div key={idx} className={styles.consoleLog}>
                  <span className={styles.logTimestamp}>[{log.time}]</span>
                  <span className={`${styles.logText} ${styles[log.category] || ""}`}>
                    {log.text}
                  </span>
                </div>
              ))}
              <div ref={consoleEndRef} />
            </div>
          </div>
        </div>
      ) : (
        <div className={styles.lobbyCard}>
          <div className={styles.brand}>
            <div className={styles.brandLogo}>🏥</div>
            <h1>MyStree Clinic</h1>
            <p>
              Welcome to our automated voice care center. Connect directly to schedule,
              check, or cancel your clinic appointments.
            </p>
          </div>

          <div className={styles.guidelines}>
            <h3>📋 Before you call:</h3>
            <ul>
              <li>Ensure your microphone is enabled.</li>
              <li>Say your phone number when asked.</li>
              <li>Keep your requests natural and clear.</li>
            </ul>
          </div>

          {error && <div className={styles.error}>{error}</div>}

          <button
            onClick={startCall}
            disabled={connecting}
            className={styles.startButton}
          >
            {connecting ? "Connecting..." : "Start Voice Call"}
          </button>
        </div>
      )}
    </main>
  );
}
