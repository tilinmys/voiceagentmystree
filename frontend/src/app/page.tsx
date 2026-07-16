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

interface LogMessage {
  id: number;
  time: string;
  text: string;
  category: string;
  details?: Record<string, unknown> | null;
}

interface CatalogVoice {
  voice_id: string;
  name: string;
  gender?: string;
  style?: string;
}

interface CatalogProvider {
  default: string | null;
  available: boolean;
  unavailable_reason: string | null;
  voices: CatalogVoice[];
}

type Catalog = Record<string, CatalogProvider>;

interface ProviderInfo {
  provider: string;
  model: string;
  key_count?: number;
}

interface ProvidersConfig {
  llm: { primary: ProviderInfo; fallback: ProviderInfo };
  stt: { primary: ProviderInfo; fallback: ProviderInfo };
  turn_detection: { mode: string; min_endpointing_delay_s: number; max_endpointing_delay_s: number };
}

interface ConfigResponse {
  providers: ProvidersConfig;
  tts_catalog: Catalog;
}

interface WorkerStatus {
  ready: boolean;
  reason?: string;
}

interface Doctor {
  name: string;
  speciality: string;
}

interface SlotCell {
  status: string;
  patient_name: string | null;
  booked_via: string | null;
}

interface WeekSchedule {
  doctor: string;
  week_start: string;
  days: string[];
  times: string[];
  grid: Record<string, Record<string, SlotCell>>;
}

interface SlotEvent {
  event_id: number;
  event_type: string;
  doctor_name: string;
  slot_date: string;
  slot_time: string;
  patient_name: string | null;
  via: string | null;
  note: string | null;
  created_at: string;
}

interface DashboardStats {
  bookings_today: number;
  cancellations_today: number;
  upcoming_appointments: number;
  open_slots_today: number;
  busiest_doctor: { name: string; upcoming: number } | null;
}

function providerLabel(info?: ProviderInfo): string {
  if (!info) return "unknown";
  const keys = info.key_count ? ` ×${info.key_count}` : "";
  return `${info.provider}/${info.model}${keys}`;
}

function dayLabel(iso: string): string {
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" });
}

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
    const handleTrackSubscribed = (_t: any, _p: any, participant: any) => {
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
  // --- call state ---
  const [token, setToken] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [room, setRoom] = useState<Room | null>(null);
  const roomRef = useRef<Room | null>(null);

  // --- config / picker ---
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [workerStatus, setWorkerStatus] = useState<WorkerStatus | null>(null);
  const [ttsProvider, setTtsProvider] = useState<string>("smallest");
  const [ttsVoice, setTtsVoice] = useState<string>("");
  const [temperature, setTemperature] = useState<number>(0.1);
  const [silenceThreshold, setSilenceThreshold] = useState<number>(0.5);
  const [ttsSpeed, setTtsSpeed] = useState<number>(1.05);
  const [sttMinSilence, setSttMinSilence] = useState<number>(90);
  const [sttMaxSilence, setSttMaxSilence] = useState<number>(320);
  const [sttInterruptionDelay, setSttInterruptionDelay] = useState<number>(120);

  // --- logs console ---
  const [logs, setLogs] = useState<LogMessage[]>([]);
  const consoleEndRef = useRef<HTMLDivElement | null>(null);
  const consoleBodyRef = useRef<HTMLDivElement | null>(null);
  // Only auto-scroll to the newest log line if the user was already at (or
  // near) the bottom before this update. Logs poll every ~1.2s during a
  // call - unconditionally scrolling on every update was yanking the view
  // back down the instant someone scrolled up to read history, making
  // scrolling feel completely broken.
  const stickToBottomRef = useRef(true);
  const [hasNewLogs, setHasNewLogs] = useState(false);
  const logsCursorRef = useRef<number>(-1);
  const [copyLabel, setCopyLabel] = useState("Copy logs");

  const handleConsoleScroll = useCallback(() => {
    const el = consoleBodyRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const atBottom = distanceFromBottom < 60;
    stickToBottomRef.current = atBottom;
    if (atBottom) setHasNewLogs(false);
  }, []);

  const jumpToLatestLogs = useCallback(() => {
    stickToBottomRef.current = true;
    setHasNewLogs(false);
    consoleEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  // --- schedule dashboard ---
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [selectedDoctor, setSelectedDoctor] = useState<string>("");
  const [schedule, setSchedule] = useState<WeekSchedule | null>(null);
  const [weekOffset, setWeekOffset] = useState(0);
  const [showNewDoctor, setShowNewDoctor] = useState(false);
  const [newDoctorName, setNewDoctorName] = useState("");
  const [newDoctorSpec, setNewDoctorSpec] = useState("");
  const [doctorError, setDoctorError] = useState<string | null>(null);
  const [addSlotDay, setAddSlotDay] = useState<string | null>(null);
  const [addSlotTime, setAddSlotTime] = useState<string>("");
  const [slotError, setSlotError] = useState<string | null>(null);
  const [slotEvents, setSlotEvents] = useState<SlotEvent[]>([]);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const slotEventsCursorRef = useRef<number>(-1);
  // "date|time" -> event_type, for the flash highlight on recently changed cells
  const [recentChanges, setRecentChanges] = useState<Record<string, string>>({});
  const selectedDoctorRef = useRef(selectedDoctor);
  selectedDoctorRef.current = selectedDoctor;
  const logIdRef = useRef(0);

  const addLog = useCallback((text: string, category = "info") => {
    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    setLogs((prev) => [...prev.slice(-499), { id: logIdRef.current++, time, text, category }]);
  }, []);

  useEffect(() => {
    if (logs.length === 0) return;
    if (stickToBottomRef.current) {
      consoleEndRef.current?.scrollIntoView({ behavior: "smooth" });
    } else {
      setHasNewLogs(true);
    }
  }, [logs]);

  const weekStartIso = useCallback(() => {
    const now = new Date();
    const monday = new Date(now);
    monday.setDate(now.getDate() - ((now.getDay() + 6) % 7) + weekOffset * 7);
    return monday.toISOString().slice(0, 10);
  }, [weekOffset]);

  // --- load config (models + voice catalog) ---
  useEffect(() => {
    let cancelled = false;
    fetch("/api/config")
      .then((res) => res.json())
      .then((data: ConfigResponse & { error?: string }) => {
        if (cancelled) return;
        if (data.error) {
          setConfigError(data.error);
          return;
        }
        setConfig(data);
        const providers = Object.keys(data.tts_catalog || {});
        const firstAvailable = providers.find((p) => data.tts_catalog[p].available) || providers[0];
        if (firstAvailable) {
          setTtsProvider(firstAvailable);
          setTtsVoice(data.tts_catalog[firstAvailable].default || data.tts_catalog[firstAvailable].voices[0]?.voice_id || "");
        }
      })
      .catch((e) => {
        if (!cancelled) setConfigError(`Could not load system config: ${e.message}`);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // --- worker readiness (lobby only) ---
  useEffect(() => {
    if (token) return;
    let cancelled = false;
    const check = () => {
      fetch("/api/worker-status")
        .then((res) => res.json())
        .then((data: WorkerStatus) => {
          if (!cancelled) setWorkerStatus(data);
        })
        .catch(() => {
          if (!cancelled) setWorkerStatus({ ready: false, reason: "Could not reach worker status API" });
        });
    };
    check();
    const interval = setInterval(check, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [token]);

  // --- doctors list ---
  const loadDoctors = useCallback(() => {
    fetch("/api/doctors")
      .then((res) => res.json())
      .then((data) => {
        const list: Doctor[] = data.doctors || [];
        setDoctors(list);
        setSelectedDoctor((cur) => cur || list[0]?.name || "");
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    loadDoctors();
  }, [loadDoctors]);

  // --- week schedule ---
  const loadSchedule = useCallback(() => {
    const doctor = selectedDoctorRef.current;
    if (!doctor) return;
    const qs = new URLSearchParams({ doctor, week_start: weekStartIso() });
    fetch(`/api/schedule?${qs.toString()}`)
      .then((res) => res.json())
      .then((data) => {
        if (data && data.grid) setSchedule(data);
      })
      .catch(() => {});
  }, [weekStartIso]);

  useEffect(() => {
    loadSchedule();
  }, [selectedDoctor, weekOffset, loadSchedule]);

  // --- dashboard stats: light poll (cheap single-table aggregates) ---
  useEffect(() => {
    let cancelled = false;
    const load = () => {
      fetch("/api/stats", { cache: "no-store" })
        .then((res) => res.json())
        .then((data) => {
          if (!cancelled && data && typeof data.bookings_today === "number") setStats(data);
        })
        .catch(() => {});
    };
    load();
    const interval = setInterval(load, 10000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  // --- live slot events: poll always, refresh schedule when it changes ---
  useEffect(() => {
    let cancelled = false;
    let inFlight = false;
    const poll = async () => {
      // A slow response (or a dev-mode double interval) must never overlap
      // with another in-flight poll - two concurrent fetches would both read
      // the same cursor, both get the same events back, and both prepend
      // them, producing duplicate event_id React keys and duplicate rows.
      if (inFlight) return;
      inFlight = true;
      try {
        const res = await fetch(`/api/slot-events?since=${slotEventsCursorRef.current}`, { cache: "no-store" });
        const data = await res.json();
        if (cancelled || !data || !Array.isArray(data.events)) return;
        slotEventsCursorRef.current = typeof data.next === "number" ? data.next : slotEventsCursorRef.current;
        if (data.events.length === 0) return;
        setSlotEvents((prev) => {
          const seen = new Set(prev.map((e) => e.event_id));
          const fresh = (data.events as SlotEvent[]).filter((e) => !seen.has(e.event_id)).reverse();
          return [...fresh, ...prev].slice(0, 30);
        });
        const changes: Record<string, string> = {};
        let affectsSelected = false;
        for (const ev of data.events as SlotEvent[]) {
          changes[`${ev.slot_date}|${ev.slot_time}`] = ev.event_type;
          if (ev.doctor_name.toLowerCase() === selectedDoctorRef.current.toLowerCase()) affectsSelected = true;
        }
        setRecentChanges((prev) => ({ ...prev, ...changes }));
        setTimeout(() => {
          setRecentChanges((prev) => {
            const next = { ...prev };
            for (const key of Object.keys(changes)) delete next[key];
            return next;
          });
        }, 6000);
        if (affectsSelected) loadSchedule();
        // A booking/cancellation just happened - refresh the stat tiles now
        // instead of waiting out the 10s idle poll.
        fetch("/api/stats", { cache: "no-store" })
          .then((res) => res.json())
          .then((data) => {
            if (data && typeof data.bookings_today === "number") setStats(data);
          })
          .catch(() => {});
      } catch {
        // transient poll error; next tick retries
      } finally {
        inFlight = false;
      }
    };
    poll();
    const interval = setInterval(poll, 1500);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [loadSchedule]);

  // --- backend pipeline logs: poll always ---
  useEffect(() => {
    let cancelled = false;
    let inFlight = false;
    const bootstrap = async () => {
      try {
        const res = await fetch("/api/logs?since=latest");
        const data = await res.json();
        logsCursorRef.current = typeof data.next === "number" ? data.next : 0;
      } catch {
        logsCursorRef.current = 0;
      }
    };
    const poll = async () => {
      if (logsCursorRef.current < 0 || inFlight) return;
      inFlight = true;
      try {
        const res = await fetch(`/api/logs?since=${logsCursorRef.current}`, { cache: "no-store" });
        const data = await res.json();
        if (cancelled || !data || !Array.isArray(data.events)) return;
        logsCursorRef.current = typeof data.next === "number" ? data.next : logsCursorRef.current;
        if (data.events.length === 0) return;
        const newEntries: LogMessage[] = data.events.map((ev: any) => {
          const time = ev.ts
            ? new Date(ev.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
            : new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
          return {
            id: logIdRef.current++,
            time,
            text: `[${ev.stage || "worker"}] ${ev.label || ""} — ${ev.message || ""}`,
            category: `backend-${ev.status || "info"}`,
            details: ev.details && Object.keys(ev.details).length ? ev.details : null,
          };
        });
        setLogs((prev) => [...prev.slice(-(500 - newEntries.length)), ...newEntries]);
      } catch {
        // transient poll error
      } finally {
        inFlight = false;
      }
    };
    logsCursorRef.current = -1;
    bootstrap().then(() => {
      if (!cancelled) poll();
    });
    const interval = setInterval(poll, 1200);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const startCall = async () => {
    if (connecting || roomRef.current) return;
    setConnecting(true);
    setError(null);
    addLog("Requesting authentication token from /api/token...", "info");
    try {
      const params = new URLSearchParams();
      if (ttsProvider) params.set("provider", ttsProvider);
      if (ttsVoice) params.set("voice", ttsVoice);
      params.set("temperature", temperature.toString());
      params.set("silenceThreshold", silenceThreshold.toString());
      params.set("ttsSpeed", ttsSpeed.toString());
      params.set("sttMinSilence", sttMinSilence.toString());
      params.set("sttMaxSilence", sttMaxSilence.toString());
      params.set("sttInterruptionDelay", sttInterruptionDelay.toString());
      const res = await fetch(`/api/token?${params.toString()}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      addLog(`Token generated. Participant: ${data.participant}`, "info");
      addLog(`TTS for this call: ${data.provider} / ${data.voice}`, "info");
      const activeRoom = new Room();
      roomRef.current = activeRoom;
      setRoom(activeRoom);
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
    const activeRoom = roomRef.current;
    roomRef.current = null;
    activeRoom?.disconnect();
    setRoom(null);
    setToken(null);
    setUrl(null);
  };

  const copyLogs = async () => {
    const text = logs
      .map((log) => {
        const base = `[${log.time}] ${log.text}`;
        return log.details ? `${base}\n${JSON.stringify(log.details, null, 2)}` : base;
      })
      .join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setCopyLabel("Copied!");
    } catch {
      setCopyLabel("Copy failed");
    }
    setTimeout(() => setCopyLabel("Copy logs"), 1500);
  };

  const createDoctor = async () => {
    setDoctorError(null);
    try {
      const res = await fetch("/api/doctors", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newDoctorName, speciality: newDoctorSpec }),
      });
      const data = await res.json();
      if (data.error) {
        setDoctorError(`Could not create doctor: ${data.error}`);
        return;
      }
      addLog(`Doctor created: ${data.doctor.name} (${data.doctor.speciality})`, "event");
      setNewDoctorName("");
      setNewDoctorSpec("");
      setShowNewDoctor(false);
      loadDoctors();
      setSelectedDoctor(data.doctor.name);
    } catch (e: any) {
      setDoctorError(`Could not create doctor: ${e.message}`);
    }
  };

  const addSlot = async (day: string) => {
    if (!addSlotTime || !selectedDoctor) return;
    setSlotError(null);
    try {
      const res = await fetch("/api/slots", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doctor: selectedDoctor, date: day, time: addSlotTime }),
      });
      const data = await res.json();
      if (data.error) {
        setSlotError(`Could not add slot: ${data.error}`);
        return;
      }
      addLog(`Slot opened: ${selectedDoctor} ${day} ${addSlotTime}`, "event");
      setAddSlotDay(null);
      setAddSlotTime("");
      loadSchedule();
    } catch (e: any) {
      setSlotError(`Could not add slot: ${e.message}`);
    }
  };

  const providers = config ? Object.keys(config.tts_catalog) : [];
  const selectedProviderCatalog = config?.tts_catalog[ttsProvider];

  const cellClass = (day: string, time: string): string => {
    const cell = schedule?.grid?.[day]?.[time];
    const recent = recentChanges[`${day}|${time}`];
    const classes = [styles.slotCell];
    if (!cell) classes.push(styles.slotNone);
    else if (cell.status === "available") classes.push(styles.slotAvailable);
    else if (cell.status === "booked") classes.push(styles.slotBooked);
    else classes.push(styles.slotClosed);
    if (recent === "booked") classes.push(styles.slotJustBooked);
    if (recent === "cancelled") classes.push(styles.slotJustCancelled);
    return classes.join(" ");
  };

  const missingTimes = (day: string): string[] =>
    (schedule?.times || []).filter((t) => !schedule?.grid?.[day]?.[t]);

  return (
    <main className={styles.page}>
      <div className={styles.topRow}>
        {/* ---------- left: call panel ---------- */}
        <div className={styles.callColumn}>
          {token && url && room ? (
            <LiveKitRoom
              room={room}
              token={token}
              serverUrl={url}
              // The library defaults audio to false - the caller's mic is
              // never published unless this is explicitly true, which was
              // the real cause behind "I talk and nothing happens": zero
              // VAD/STT activity in the worker logs because no microphone
              // track was ever being sent at all.
              audio={true}
              onDisconnected={endCall}
              data-lk-theme="default"
              className={styles.roomContainer}
            >
              <RoomLogger onLog={addLog} />
              <div className={styles.callCard}>
                <span className={styles.statusBadge}>Live Call</span>
                <h1 className={styles.clinicName}>MyStree Clinic</h1>
                {config && (
                  <div className={styles.modelBar}>
                    <div className={styles.modelRow}>
                      <span className={styles.modelLabel}>STT</span>
                      <span className={styles.modelValue}>{providerLabel(config.providers.stt.primary)}</span>
                    </div>
                    <div className={styles.modelRow}>
                      <span className={styles.modelLabel}>LLM</span>
                      <span className={styles.modelValue}>{providerLabel(config.providers.llm.primary)}</span>
                    </div>
                    <div className={styles.modelRow}>
                      <span className={styles.modelLabel}>TTS</span>
                      <span className={styles.modelValue}>{ttsProvider} / {ttsVoice || "default"}</span>
                    </div>
                  </div>
                )}
                <SimpleVisualizer onLog={addLog} />
                <div className={styles.controlsWrapper}>
                  <VoiceAssistantControlBar />
                </div>
              </div>
              <RoomAudioRenderer />
            </LiveKitRoom>
          ) : (
            <div className={styles.callCard}>
              <h1 className={styles.clinicName}>MyStree Clinic</h1>
              <div className={styles.workerStatusRow}>
                <span className={`${styles.workerDot} ${workerStatus?.ready ? styles.workerReady : styles.workerDown}`} />
                <span>
                  {workerStatus === null ? "Checking worker..." : workerStatus.ready ? "Worker ready" : workerStatus.reason || "Worker not ready"}
                </span>
              </div>

              {config && (
                <div className={styles.modelBar}>
                  <div className={styles.modelRow}>
                    <span className={styles.modelLabel}>STT</span>
                    <span className={styles.modelValue}>
                      {providerLabel(config.providers.stt.primary)} <span className={styles.modelArrow}>→</span> {providerLabel(config.providers.stt.fallback)}
                    </span>
                  </div>
                  <div className={styles.modelRow}>
                    <span className={styles.modelLabel}>LLM</span>
                    <span className={styles.modelValue}>
                      {providerLabel(config.providers.llm.primary)} <span className={styles.modelArrow}>→</span> {providerLabel(config.providers.llm.fallback)}
                    </span>
                  </div>
                </div>
              )}

              {config && (
                <div className={styles.pickerRow}>
                  <label className={styles.pickerLabel}>
                    TTS provider
                    <select
                      className={styles.pickerSelect}
                      value={ttsProvider}
                      onChange={(e) => {
                        const next = e.target.value;
                        setTtsProvider(next);
                        const cat = config.tts_catalog[next];
                        setTtsVoice(cat?.default || cat?.voices[0]?.voice_id || "");
                      }}
                    >
                      {providers.map((p) => (
                        <option key={p} value={p} disabled={!config.tts_catalog[p].available}>
                          {p}
                          {!config.tts_catalog[p].available ? " (unavailable)" : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className={styles.pickerLabel}>
                    Voice
                    <select
                      className={styles.pickerSelect}
                      value={ttsVoice}
                      onChange={(e) => setTtsVoice(e.target.value)}
                      disabled={!selectedProviderCatalog?.available}
                    >
                      {selectedProviderCatalog?.voices.map((v) => (
                        <option key={v.voice_id} value={v.voice_id}>
                          {v.name}
                          {v.gender ? ` (${v.gender})` : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              )}
              
              <div className={styles.pickerRow} style={{ marginTop: '1rem' }}>
                <label className={styles.pickerLabel} style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', width: '100%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Temperature (Tone Strictness)</span>
                    <span style={{ fontWeight: 'bold' }}>{temperature.toFixed(2)}</span>
                  </div>
                  <input 
                    type="range" 
                    min="0" max="1" step="0.05" 
                    value={temperature} 
                    onChange={(e) => setTemperature(parseFloat(e.target.value))} 
                    style={{ width: '100%' }}
                  />
                  <small style={{ fontSize: '0.75rem', color: '#666' }}>Lower = Strict/Anti-hallucination. Higher = Creative/Chatty.</small>
                </label>
              </div>

              <div className={styles.pickerRow} style={{ marginTop: '1rem', marginBottom: '1rem' }}>
                <label className={styles.pickerLabel} style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', width: '100%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Silence Threshold (Seconds)</span>
                    <span style={{ fontWeight: 'bold' }}>{silenceThreshold.toFixed(2)}s</span>
                  </div>
                  <input 
                    type="range" 
                    min="0.1" max="2.0" step="0.1" 
                    value={silenceThreshold} 
                    onChange={(e) => setSilenceThreshold(parseFloat(e.target.value))} 
                    style={{ width: '100%' }}
                  />
                  <small style={{ fontSize: '0.75rem', color: '#666' }}>How long the agent waits after you stop talking before it replies.</small>
                </label>
              </div>

              <div className={styles.pickerRow} style={{ marginTop: '1rem' }}>
                <label className={styles.pickerLabel} style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', width: '100%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>TTS Speech Speed</span>
                    <span style={{ fontWeight: 'bold' }}>{ttsSpeed.toFixed(2)}x</span>
                  </div>
                  <input 
                    type="range" 
                    min="0.8" max="1.5" step="0.05" 
                    value={ttsSpeed} 
                    onChange={(e) => setTtsSpeed(parseFloat(e.target.value))} 
                    style={{ width: '100%' }}
                  />
                  <small style={{ fontSize: '0.75rem', color: '#666' }}>Multiplier for how fast the agent speaks (Smallest.ai only).</small>
                </label>
              </div>

              <div className={styles.pickerRow} style={{ marginTop: '1rem' }}>
                <label className={styles.pickerLabel} style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', width: '100%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>STT Min Turn Silence (ms)</span>
                    <span style={{ fontWeight: 'bold' }}>{sttMinSilence}ms</span>
                  </div>
                  <input 
                    type="range" 
                    min="60" max="180" step="10" 
                    value={sttMinSilence} 
                    onChange={(e) => setSttMinSilence(parseInt(e.target.value, 10))} 
                    style={{ width: '100%' }}
                  />
                  <small style={{ fontSize: '0.75rem', color: '#666' }}>Min silence before STT thinks your sentence is done.</small>
                </label>
              </div>

              <div className={styles.pickerRow} style={{ marginTop: '1rem' }}>
                <label className={styles.pickerLabel} style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', width: '100%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>STT Max Turn Silence (ms)</span>
                    <span style={{ fontWeight: 'bold' }}>{sttMaxSilence}ms</span>
                  </div>
                  <input 
                    type="range" 
                    min="180" max="500" step="10" 
                    value={sttMaxSilence} 
                    onChange={(e) => setSttMaxSilence(parseInt(e.target.value, 10))} 
                    style={{ width: '100%' }}
                  />
                  <small style={{ fontSize: '0.75rem', color: '#666' }}>Absolute max silence before STT cuts off the turn.</small>
                </label>
              </div>

              <div className={styles.pickerRow} style={{ marginTop: '1rem', marginBottom: '1rem' }}>
                <label className={styles.pickerLabel} style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', width: '100%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>STT Interruption Delay (ms)</span>
                    <span style={{ fontWeight: 'bold' }}>{sttInterruptionDelay}ms</span>
                  </div>
                  <input 
                    type="range" 
                    min="80" max="250" step="10" 
                    value={sttInterruptionDelay} 
                    onChange={(e) => setSttInterruptionDelay(parseInt(e.target.value, 10))} 
                    style={{ width: '100%' }}
                  />
                  <small style={{ fontSize: '0.75rem', color: '#666' }}>Ms of speech needed to interrupt the agent.</small>
                </label>
              </div>

              {configError && <div className={styles.error}>{configError}</div>}
              {error && <div className={styles.error}>{error}</div>}

              <button
                onClick={startCall}
                disabled={connecting || (workerStatus !== null && !workerStatus.ready)}
                className={styles.startButton}
              >
                {connecting ? "Connecting..." : "Start Voice Call"}
              </button>
            </div>
          )}
        </div>

        {/* ---------- right: doctor schedule dashboard ---------- */}
        <div className={styles.scheduleCard}>
          {stats && (
            <div className={styles.statsRow}>
              <div className={styles.statTile}>
                <span className={styles.statValue}>{stats.bookings_today}</span>
                <span className={styles.statLabel}>Booked today</span>
              </div>
              <div className={styles.statTile}>
                <span className={styles.statValue}>{stats.cancellations_today}</span>
                <span className={styles.statLabel}>Cancelled today</span>
              </div>
              <div className={styles.statTile}>
                <span className={styles.statValue}>{stats.upcoming_appointments}</span>
                <span className={styles.statLabel}>Upcoming</span>
              </div>
              <div className={styles.statTile}>
                <span className={styles.statValue}>{stats.open_slots_today}</span>
                <span className={styles.statLabel}>Open slots today</span>
              </div>
              {stats.busiest_doctor && (
                <div className={`${styles.statTile} ${styles.statTileWide}`}>
                  <span className={styles.statValue}>{stats.busiest_doctor.name}</span>
                  <span className={styles.statLabel}>Busiest · {stats.busiest_doctor.upcoming} upcoming</span>
                </div>
              )}
            </div>
          )}
          <div className={styles.scheduleHeader}>
            <select
              className={styles.pickerSelect}
              value={selectedDoctor}
              onChange={(e) => setSelectedDoctor(e.target.value)}
            >
              {doctors.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name} — {d.speciality}
                </option>
              ))}
            </select>
            <button className={styles.smallButton} onClick={() => setShowNewDoctor((s) => !s)} type="button">
              {showNewDoctor ? "Cancel" : "+ New doctor"}
            </button>
            <div className={styles.weekNav}>
              <button className={styles.smallButton} onClick={() => setWeekOffset((w) => w - 1)} type="button">‹</button>
              <span className={styles.weekLabel}>{schedule ? `Week of ${dayLabel(schedule.week_start)}` : "…"}</span>
              <button className={styles.smallButton} onClick={() => setWeekOffset((w) => w + 1)} type="button">›</button>
            </div>
          </div>

          {showNewDoctor && (
            <div className={styles.newDoctorRow}>
              <input
                className={styles.textInput}
                placeholder="Doctor name (e.g. Dr. Asha Rao)"
                value={newDoctorName}
                onChange={(e) => setNewDoctorName(e.target.value)}
              />
              <input
                className={styles.textInput}
                placeholder="Speciality"
                value={newDoctorSpec}
                onChange={(e) => setNewDoctorSpec(e.target.value)}
              />
              <button className={styles.smallButtonPrimary} onClick={createDoctor} type="button">Add</button>
            </div>
          )}
          {doctorError && <div className={styles.error}>{doctorError}</div>}
          {slotError && <div className={styles.error}>{slotError}</div>}

          <div className={styles.tableScroll}>
            <table className={styles.slotTable}>
              <thead>
                <tr>
                  <th className={styles.timeHead}></th>
                  {(schedule?.days || []).map((day) => (
                    <th key={day} className={styles.dayHead}>
                      <span>{dayLabel(day)}</span>
                      <button
                        className={styles.plusButton}
                        type="button"
                        title={`Add a slot on ${dayLabel(day)}`}
                        onClick={() => {
                          setAddSlotDay(addSlotDay === day ? null : day);
                          setAddSlotTime(missingTimes(day)[0] || "");
                        }}
                      >
                        +
                      </button>
                      {addSlotDay === day && (
                        <div className={styles.addSlotPopover}>
                          <select
                            className={styles.pickerSelect}
                            value={addSlotTime}
                            onChange={(e) => setAddSlotTime(e.target.value)}
                          >
                            {missingTimes(day).map((t) => (
                              <option key={t} value={t}>{t}</option>
                            ))}
                          </select>
                          <button className={styles.smallButtonPrimary} type="button" onClick={() => addSlot(day)}>
                            Open slot
                          </button>
                        </div>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(schedule?.times || []).map((time) => (
                  <tr key={time}>
                    <td className={styles.timeCell}>{time}</td>
                    {(schedule?.days || []).map((day) => {
                      const cell = schedule?.grid?.[day]?.[time];
                      return (
                        <td key={day + time} className={cellClass(day, time)}>
                          {cell?.status === "booked" ? (
                            <span className={styles.slotPatient}>{cell.patient_name || cell.booked_via || "booked"}</span>
                          ) : cell?.status === "available" ? (
                            <span className={styles.slotOpenMark}>open</span>
                          ) : cell ? (
                            <span className={styles.slotClosedMark}>closed</span>
                          ) : null}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* live booking/cancellation feed */}
          <div className={styles.liveFeed}>
            <div className={styles.liveFeedHeader}>
              <span className={styles.liveDot} /> Live slot activity
            </div>
            {slotEvents.length === 0 && <div className={styles.feedEmpty}>No bookings or cancellations yet.</div>}
            {slotEvents.map((ev) => (
              <div key={ev.event_id} className={`${styles.feedRow} ${ev.event_type === "booked" ? styles.feedBooked : styles.feedCancelled}`}>
                <span className={styles.feedType}>{ev.event_type === "booked" ? "BOOKED" : "CANCELLED"}</span>
                <span className={styles.feedText}>
                  {ev.slot_time} · {dayLabel(ev.slot_date)} · {ev.doctor_name}
                  {ev.patient_name ? ` · ${ev.patient_name}` : ""}
                  {ev.note ? ` (${ev.note})` : ""}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ---------- bottom: full-width detailed logs ---------- */}
      <div className={styles.consoleCardWide}>
        <div className={styles.consoleHeader}>
          <h2>System Logs Console</h2>
          <div className={styles.consoleHeaderActions}>
            <button className={styles.copyButton} onClick={copyLogs} type="button">
              {copyLabel}
            </button>
            <span className={styles.consoleStatus}>Live</span>
          </div>
        </div>
        <div className={styles.consoleBodyWrapper}>
          <div className={styles.consoleBody} ref={consoleBodyRef} onScroll={handleConsoleScroll}>
            {logs.map((log) => (
              <div key={log.id} className={styles.consoleLog}>
                <span className={styles.logTimestamp}>[{log.time}]</span>
                <span className={`${styles.logText} ${styles[log.category] || ""}`}>{log.text}</span>
                {log.details && <pre className={styles.logDetails}>{JSON.stringify(log.details, null, 2)}</pre>}
              </div>
            ))}
            <div ref={consoleEndRef} />
          </div>
          {hasNewLogs && (
            <button className={styles.jumpToLatestButton} onClick={jumpToLatestLogs} type="button">
              ↓ New logs
            </button>
          )}
        </div>
      </div>
    </main>
  );
}
