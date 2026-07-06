import os
import asyncio
import logging
import sqlite3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

# Initialize database
import db_helper
db_helper.init_db(reset=os.getenv("SQLITE_RESET_ON_START", "false").lower() == "true")

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    llm,
    stt,
    tts,
    RoomInputOptions,
    AgentSession,
    Agent,
    TurnHandlingOptions,
    EndpointingOptions,
    PreemptiveGenerationOptions,
    RunContext,
    metrics,
    MetricsCollectedEvent
)
from livekit.agents.inference import TurnDetector
from livekit.plugins import openai, assemblyai, silero, noise_cancellation
from sarvam_wrappers import SarvamSTT, SarvamTTS

# Define actions as function tools
@llm.function_tool
async def lookup_doctors(ctx: RunContext) -> str:
    """Lists all available doctors at MyStree Clinic along with their specialities."""
    await ctx.session.say("Let me look that up for you...", allow_interruptions=True)
    await asyncio.sleep(0.8)
    
    try:
        doctors = db_helper.get_doctors()
        res = "Available doctors at MyStree Clinic:\n"
        for doc in doctors:
            res += f"- {doc['name']} ({doc['speciality']})\n"
        return res
    except Exception as e:
        logger.error(f"Error looking up doctors: {e}")
        raise llm.ToolError("Failed to lookup doctors list. Please try again shortly.")

@llm.function_tool
async def lookup_booking_timings(ctx: RunContext, doctor_name: str, date: str) -> str:
    """Looks up available appointment booking timings (slots) for a specific doctor on a given date (YYYY-MM-DD)."""
    await ctx.session.say("Let me look that up for you...", allow_interruptions=True)
    await asyncio.sleep(0.8)
    
    try:
        slots = db_helper.get_booking_timings(doctor_name, date)
        if not slots:
            raise llm.ToolError(f"No slots available for {doctor_name} on {date}.")
        
        res = f"Available timings for {doctor_name} on {date}:\n"
        for slot in slots:
            res += f"- {slot}\n"
        return res
    except llm.ToolError:
        raise
    except Exception as e:
        logger.error(f"Error looking up timings: {e}")
        raise llm.ToolError(f"Failed to lookup timings for {doctor_name} on {date}: {e}")
@llm.function_tool
async def lookup_appointments(ctx: RunContext, phone: str) -> str:
    """Looks up scheduled clinic appointments in the database for a patient by phone number. Always verify phone number first."""
    await ctx.session.say("Let me search for your appointments.", allow_interruptions=True)
    await asyncio.sleep(0.8)
    
    try:
        patient = db_helper.get_patient_by_phone(phone)
        if not patient:
            return f"No patient found with phone number {phone}. The patient needs to register first."
        
        appointments = db_helper.get_appointments_by_patient_id(patient["patient_id"])
        if not appointments:
            return f"Patient {patient['name']} has no scheduled appointments."
        
        res = f"Appointments for {patient['name']}:\n"
        for appt in appointments:
            res += f"- ID: {appt['appointment_id']}, with {appt['doctor_name']} on {appt['appointment_date']} at {appt['appointment_time']}\n"
        return res
    except Exception as e:
        logger.error(f"Error in lookup_appointments: {e}")
        raise llm.ToolError(f"Failed to lookup appointments: {e}")

@llm.function_tool
async def book_appointment(ctx: RunContext, phone: str, doctor_name: str, date: str, time: str) -> str:
    """Books a new appointment in the database using phone, doctor_name, date (e.g. 2026-07-08), and time (e.g. 10:00 AM). Always verify phone number first."""
    await ctx.session.say(f"Checking schedules to book with {doctor_name}...", allow_interruptions=True)
    await asyncio.sleep(0.8)
        
    try:
        patient = db_helper.get_patient_by_phone(phone)
        if not patient:
            return f"No patient found with phone number {phone}. They must be registered before booking."
        
        appointment_id = db_helper.book_appointment(patient["patient_id"], doctor_name, date, time)
        return f"Successfully booked appointment for {patient['name']} (ID: {appointment_id}) with {doctor_name} on {date} at {time}."
    except Exception as e:
        logger.error(f"Error booking appointment: {e}")
        raise llm.ToolError(f"Failed to book appointment: {e}")

@llm.function_tool
async def cancel_appointment(ctx: RunContext, appointment_id: int) -> str:
    """Cancels a scheduled appointment in the database using its unique appointment ID."""
    await ctx.session.say("Processing appointment cancellation...", allow_interruptions=True)
    await asyncio.sleep(0.8)
        
    try:
        success = db_helper.cancel_appointment(appointment_id)
        if success:
            return f"Appointment {appointment_id} has been successfully cancelled."
        else:
            return f"Appointment ID {appointment_id} was not found or has already been cancelled."
    except Exception as e:
        logger.error(f"Error cancelling appointment: {e}")
        raise llm.ToolError(f"Failed to cancel appointment: {e}")

@llm.function_tool
async def register_patient(ctx: RunContext, name: str, phone: str, dob: str) -> str:
    """Registers a new patient with their full name, phone number, and DOB (YYYY-MM-DD)."""
    await ctx.session.say(f"Registering patient {name} in our clinic system...", allow_interruptions=True)
    await asyncio.sleep(0.8)
        
    try:
        conn = sqlite3.connect(db_helper.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO patients (name, phone, dob) VALUES (?, ?, ?)", (name, phone, dob))
        conn.commit()
        conn.close()
        return f"Patient {name} with phone {phone} has been successfully registered."
    except sqlite3.IntegrityError:
        return f"A patient with phone number {phone} is already registered."
    except Exception as e:
        logger.error(f"Error registering patient: {e}")
        raise llm.ToolError(f"Failed to register patient: {e}")


async def entrypoint(ctx: JobContext):
    logger.info("Starting care coordinator agent session...")
    try:
        # 1. Setup custom Sarvam STT and fallback AssemblyAI
        sarvam_stt_key = os.getenv("SARVAM_API_KEY")
        sarvam_stt_model = os.getenv("SARVAM_STT_MODEL", "saarika:v2.5")
        sarvam_stt_lang = os.getenv("SARVAM_LANGUAGE_CODE", "en-IN")
        
        sarvam_stt = SarvamSTT(
            api_key=sarvam_stt_key,
            model=sarvam_stt_model,
            language_code=sarvam_stt_lang
        )
        
        assemblyai_stt = assemblyai.STT(
            api_key=os.getenv("ASSEMBLYAI_API_KEY"),
            model=os.getenv("ASSEMBLYAI_STT_MODEL", "universal-3-5-pro"),
            keyterms_prompt=[os.getenv("CLINIC_NAME", "MyStree Clinic")]
        )
        
        stt_fallback = stt.FallbackAdapter([
            assemblyai_stt,
            sarvam_stt
        ])
        
        # 2. Setup custom Sarvam TTS and fallback OpenAI TTS
        sarvam_tts_model = os.getenv("SARVAM_TTS_MODEL", "bulbul:v2")
        sarvam_tts_speaker = os.getenv("SARVAM_SPEAKER", "anushka")
        
        sarvam_tts = SarvamTTS(
            api_key=sarvam_stt_key,
            model=sarvam_tts_model,
            speaker=sarvam_tts_speaker,
            target_language_code=sarvam_stt_lang
        )
        
        openai_tts = openai.TTS()
        
        tts_fallback = tts.FallbackAdapter([
            sarvam_tts,
            openai_tts
        ])
        
        # 3. Setup LLM and Groq fallback
        openai_llm = openai.LLM(model=os.getenv("LLM_MODEL", "gpt-4o-mini"))
        groq_llm = openai.LLM(
            model=os.getenv("LLM_FALLBACK_MODEL", "llama-3.3-70b-versatile"),
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        llm_fallback = llm.FallbackAdapter([
            openai_llm,
            groq_llm
        ])
        
        # 4. Setup Chat Context & Instructions
        initial_ctx = llm.ChatContext()
        initial_ctx.add_message(
            role="system",
            content=f"You are an empathetic and highly professional receptionist for {os.getenv('CLINIC_NAME', 'MyStree Clinic')}. "
                    "When the call connects, you must immediately greet the user with: 'Welcome to MyStree Clinic, how can I help you today?' "
                    "Your goal is to assist patients with lookup, booking, registration, or cancelling clinic appointments. "
                    "Always verify the patient's phone number before booking, cancelling, or looking up their appointments. "
                    "Keep all replies under three sentences. Use short clauses and absolutely avoid parentheticals or markdown. "
                    "To ensure digits (like phone numbers, appointment IDs, or dates) are read naturally by the text-to-speech engine, you must format all numbers by separating individual digits with dashes (e.g., 9-8-7-6-5-4-3-2-1-0 or 1-2-3), rather than outputting them as a single large number."
        )
        
        # 5. Initialize the modern AgentSession with low-latency settings
        session = AgentSession(
            stt=stt_fallback,
            vad=silero.VAD.load(min_silence_duration=0.3),
            llm=llm_fallback,
            tts=tts_fallback,
            tools=[
                lookup_appointments,
                book_appointment,
                cancel_appointment,
                register_patient,
                lookup_doctors,
                lookup_booking_timings
            ],
            turn_handling=TurnHandlingOptions(
                turn_detection=TurnDetector(),
                endpointing=EndpointingOptions(
                    mode="fixed",
                    min_delay=0.3,
                    max_delay=0.3
                ),
                preemptive_generation=PreemptiveGenerationOptions(
                    enabled=True,
                    preemptive_tts=True
                )
            )
        )
        
        # Initialize usage collector for timing and tokens profiling
        usage_collector = metrics.UsageCollector()

        @session.on("metrics_collected")
        def _on_metrics_collected(ev: MetricsCollectedEvent):
            metrics.log_metrics(ev.metrics)
            usage_collector.collect(ev.metrics)
        
        logger.info("Connecting to LiveKit room...")
        await ctx.connect()
        
        logger.info("Starting AgentSession...")
        # Define instructions in the Agent
        agent = Agent(
            instructions="You are a helpful care coordinator receptionist for MyStree Clinic.",
            chat_ctx=initial_ctx
        )
        
        await session.start(
            agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC()
            )
        )
        
        # Initial clinic greeting
        await session.say("Welcome to MyStree Clinic, how can I help you today?", allow_interruptions=True)
        
        # Session loop
        while ctx.room.connection_state == "connected":
            await asyncio.sleep(1)
            
        logger.info("Room disconnected, ending agent task.")
        summary = usage_collector.get_summary()
        logger.info(f"Session Usage Summary: {summary}")
    except Exception as e:
        logger.error(f"Fatal error in agent entrypoint: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    except (KeyboardInterrupt, SystemExit):
        logger.info("Received termination signal. Closing Care Coordinator agent.")
