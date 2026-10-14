
"""
Ray-Ban Meta Glasses + YOLO11 ONNX + OpenAI LLM + Voice
Windows / Python 3.14

S = Describe scene
T = Type a question
V = Ask through microphone
M = Select microphone
A = Toggle automatic narration
B = Test repeated speech
Q = Quit

Epson VT6 robot movement is DISABLED.
"""

import os
import sys
import time
import base64
import subprocess
from pathlib import Path
from collections import Counter

import cv2
import mss
import numpy as np
import onnxruntime as ort
from openai import OpenAI

# ---------------------------------------------------
# MICROPHONE: PYAUDIOWPATCH
# ---------------------------------------------------

try:
    import pyaudiowpatch as pyaudio
    sys.modules["pyaudio"] = pyaudio
    AUDIO_BACKEND = "PyAudioWPatch"
except ImportError:
    try:
        import pyaudio
        AUDIO_BACKEND = "PyAudio"
    except ImportError:
        pyaudio = None
        AUDIO_BACKEND = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------

MODEL_PATH = Path(__file__).with_name("yolo11n.onnx")

IMAGE_SIZE = 640
CONFIDENCE = 0.35
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 40

LLM_MODEL = os.getenv(
    "OPENAI_VISION_MODEL",
    "gpt-4.1-mini"
)

AUTO_INTERVAL_SECONDS = 25
JPEG_QUALITY = 75

MIC_DEVICE_INDEX = None

# ---------------------------------------------------
# COCO OBJECT LABELS
# ---------------------------------------------------

COCO = (
    "person,bicycle,car,motorcycle,airplane,bus,train,"
    "truck,boat,traffic light,fire hydrant,stop sign,"
    "parking meter,bench,bird,cat,dog,horse,sheep,cow,"
    "elephant,bear,zebra,giraffe,backpack,umbrella,"
    "handbag,tie,suitcase,frisbee,skis,snowboard,"
    "sports ball,kite,baseball bat,baseball glove,"
    "skateboard,surfboard,tennis racket,bottle,"
    "wine glass,cup,fork,knife,spoon,bowl,banana,"
    "apple,sandwich,orange,broccoli,carrot,hot dog,"
    "pizza,donut,cake,chair,couch,potted plant,bed,"
    "dining table,toilet,tv,laptop,mouse,remote,"
    "keyboard,cell phone,microwave,oven,toaster,"
    "sink,refrigerator,book,clock,vase,scissors,"
    "teddy bear,hair drier,toothbrush"
).split(",")

# ---------------------------------------------------
# WINDOWS DPI
# ---------------------------------------------------

def setup_dpi():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass

# ---------------------------------------------------
# WINDOWS TEXT TO SPEECH
# FIX: NEW SPEECH PROCESS FOR EACH RESPONSE
# ---------------------------------------------------

def speak(words):
    if not words or not str(words).strip():
        return False

    print("[VOICE] Starting speech...", flush=True)

    ps_script = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech

$text = [Console]::In.ReadToEnd()

$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer

try {
    $voice.Rate = 0
    $voice.Volume = 100
    $voice.Speak($text)
}
finally {
    $voice.Dispose()
}
"""

    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                ps_script
            ],
            input=str(words),
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
            creationflags=getattr(
                subprocess,
                "CREATE_NO_WINDOW",
                0
            )
        )

        if result.returncode == 0:
            print("[VOICE] Speech completed.")
            return True

        print("[VOICE ERROR]", result.stderr.strip())
        return False

    except subprocess.TimeoutExpired:
        print("[VOICE ERROR] Speech timed out.")
        return False

    except Exception as e:
        print("[VOICE ERROR]", e)
        return False

# ---------------------------------------------------
# MICROPHONE SELECTION
# ---------------------------------------------------

def select_microphone():
    global MIC_DEVICE_INDEX

    if sr is None or pyaudio is None:
        print("[MIC] Install SpeechRecognition and PyAudioWPatch.")
        return

    try:
        microphones = sr.Microphone.list_microphone_names()

        print("\nAVAILABLE MICROPHONES")

        for i, name in enumerate(microphones):
            print(f"{i}: {name}")

        choice = input(
            "Select microphone index (Enter = default): "
        ).strip()

        if not choice:
            MIC_DEVICE_INDEX = None
            print("[MIC] Windows default microphone selected.")
            return

        index = int(choice)

        if index < 0 or index >= len(microphones):
            print("[MIC] Invalid microphone index.")
            return

        MIC_DEVICE_INDEX = index
        print(f"[MIC] Selected: {microphones[index]}")

    except Exception as e:
        print("[MIC ERROR]", e)

# ---------------------------------------------------
# SPEECH RECOGNITION
# ---------------------------------------------------

def listen_question():
    if sr is None or pyaudio is None:
        print("[MIC] Speech recognition is not available.")
        return None

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone(
            device_index=MIC_DEVICE_INDEX
        ) as source:

            print("[MIC] Adjusting background noise...")

            recognizer.adjust_for_ambient_noise(
                source,
                duration=0.5
            )

            print("[MIC] Listening... Speak now.")

            audio = recognizer.listen(
                source,
                timeout=8,
                phrase_time_limit=12
            )

        print("[MIC] Recognizing speech...")

        question = recognizer.recognize_google(audio)

        print(f"[MIC] You said: {question}")

        return question

    except sr.WaitTimeoutError:
        print("[MIC] No speech detected.")

    except sr.UnknownValueError:
        print("[MIC] Could not understand your speech.")

    except sr.RequestError as e:
        print("[MIC] Speech recognition service error:", e)

    except Exception as e:
        print("[MIC ERROR]", e)

    return None

# ---------------------------------------------------
# LOAD YOLO11 ONNX RUNTIME
# ---------------------------------------------------

def init_yolo():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing YOLO model: {MODEL_PATH}\n"
            "Put yolo11n.onnx beside your Python file."
        )

    print("[YOLO] Loading model...")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    inp = session.get_inputs()[0]

    if inp.type != "tensor(float)":
        raise ValueError(
            f"Unsupported model input type: {inp.type}"
        )

    shape = inp.shape

    height = (
        shape[2]
        if isinstance(shape[2], int)
        else IMAGE_SIZE
    )

    width = (
        shape[3]
        if isinstance(shape[3], int)
        else IMAGE_SIZE
    )

    if height <= 0 or width <= 0:
        raise ValueError("Invalid YOLO model input size.")

    print("[YOLO] Model loaded.")
    print("[YOLO] Input:", shape)
    print(
        "[YOLO] Output:",
        [(o.name, o.shape) for o in session.get_outputs()]
    )

    return session, inp.name, (width, height)

# ---------------------------------------------------
# LETTERBOX PREPROCESSING
# ---------------------------------------------------

def letterbox(image, size):
    target_w, target_h = size

    h, w = image.shape[:2]

    scale = min(
        target_w / w,
        target_h / h
    )

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h)
    )

    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2

    canvas = np.full(
        (target_h, target_w, 3),
        114,
        dtype=np.uint8
    )

    canvas[
        dy:dy + new_h,
        dx:dx + new_w
    ] = resized

    return canvas, scale, dx, dy

# ---------------------------------------------------
# YOLO DETECTION
# ---------------------------------------------------

def yolo_detect(frame, session, input_name, size):
    padded, scale, dx, dy = letterbox(frame, size)

    rgb = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB
    )

    blob = np.ascontiguousarray(
        rgb.transpose(2, 0, 1)[None],
        dtype=np.float32
    ) / 255.0

    raw = session.run(
        None,
        {input_name: blob}
    )[0]

    output = np.squeeze(raw)

    if output.ndim != 2:
        raise ValueError(
            f"Unsupported YOLO output shape: {raw.shape}"
        )

    if (
        output.shape[0] in (84, 85)
        and output.shape[1] > output.shape[0]
    ):
        output = output.T

    if output.shape[1] not in (84, 85):
        raise ValueError(
            f"Unsupported YOLO output: {output.shape}"
        )

    if output.shape[1] == 84:
        class_scores = output[:, 4:]
        class_ids = np.argmax(
            class_scores,
            axis=1
        )
        scores = class_scores[
            np.arange(len(output)),
            class_ids
        ]
    else:
        class_scores = output[:, 5:]
        class_ids = np.argmax(
            class_scores,
            axis=1
        )
        scores = output[:, 4] * class_scores[
            np.arange(len(output)),
            class_ids
        ]

    mask = scores >= CONFIDENCE

    candidates = output[mask, :4]
    scores = scores[mask]
    class_ids = class_ids[mask]

    boxes = []
    classes = []
    confidences = []

    img_h, img_w = frame.shape[:2]

    for box, score, cls in zip(
        candidates,
        scores,
        class_ids
    ):
        cx, cy, bw, bh = box

        x1 = int(np.clip(
            (cx - bw / 2 - dx) / scale,
            0, img_w - 1
        ))

        y1 = int(np.clip(
            (cy - bh / 2 - dy) / scale,
            0, img_h - 1
        ))

        x2 = int(np.clip(
            (cx + bw / 2 - dx) / scale,
            0, img_w - 1
        ))

        y2 = int(np.clip(
            (cy + bh / 2 - dy) / scale,
            0, img_h - 1
        ))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ])

        classes.append(int(cls))
        confidences.append(float(score))

    selected = []

    # Apply non-maximum suppression per class.
    for cls in sorted(set(classes)):
        ids = [
            i for i, c in enumerate(classes)
            if c == cls
        ]

        class_boxes = [boxes[i] for i in ids]
        class_scores = [
            confidences[i] for i in ids
        ]

        kept = cv2.dnn.NMSBoxes(
            class_boxes,
            class_scores,
            CONFIDENCE,
            IOU_THRESHOLD
        )

        if len(kept):
            for j in np.asarray(kept).reshape(-1):
                selected.append(ids[int(j)])

    selected.sort(
        key=lambda i: confidences[i],
        reverse=True
    )

    selected = selected[:MAX_DETECTIONS]

    annotated = frame.copy()
    detections = []

    for i in selected:
        x, y, w, h = boxes[i]

        cls = classes[i]

        label = (
            COCO[cls]
            if 0 <= cls < len(COCO)
            else f"class_{cls}"
        )

        confidence = confidences[i]

        detections.append({
            "label": label,
            "confidence": round(confidence, 2),
            "box": [x, y, w, h]
        })

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 200, 0),
            2
        )

        cv2.putText(
            annotated,
            f"{label} {confidence:.2f}",
            (x, max(20, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 240, 0),
            2
        )

    return detections, annotated

# ---------------------------------------------------
# OPENAI VISION LLM
# ---------------------------------------------------

def ask_llm(client, frame, detections, question):
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )

    if not success:
        print("[LLM ERROR] Image encoding failed.")
        return None

    image_base64 = base64.b64encode(
        encoded.tobytes()
    ).decode("ascii")

    counts = Counter(
        d["label"] for d in detections
    )

    description = ", ".join(
        f"{count} {label}"
        for label, count in counts.items()
    ) or "none"

    print("[LLM] YOLO detections:", description)
    print("[LLM] Asking OpenAI...")

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistive AI vision "
                        "companion using Ray-Ban glasses. "
                        "You receive a captured image and "
                        "YOLO object detections. "
                        "Answer the user's question "
                        "clearly and naturally. "
                        "Use the image rather than "
                        "relying only on YOLO labels. "
                        "If something is uncertain, "
                        "say so. "
                        "Do not invent objects or distances. "
                        "Do not claim to see continuous video. "
                        "Do not issue robot commands. "
                        "Keep your response brief, "
                        "usually one to three sentences."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Question: {question}\n"
                                f"YOLO detections: {description}\n"
                                "Please examine this "
                                "current camera image."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/jpeg;base64,"
                                    + image_base64
                                ),
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=180
        )

        answer = (
            response.choices[0].message.content
            or "No response returned."
        )

        print("\n[AI]", answer, "\n")

        return answer

    except Exception as e:
        print("[LLM ERROR]", e)
        return None

# ---------------------------------------------------
# SCREEN CAPTURE REGION
# ---------------------------------------------------

def select_capture_region(sct):
    monitors = sct.monitors[1:]

    print("\nAVAILABLE MONITORS")

    for i, monitor in enumerate(monitors, 1):
        print(i, monitor)

    choice = input(
        "Monitor containing Ray-Ban video [1]: "
    ).strip()

    index = int(choice) if choice else 1

    if not 1 <= index <= len(monitors):
        raise ValueError("Invalid monitor number.")

    monitor = monitors[index - 1]

    screenshot = np.asarray(
        sct.grab(monitor)
    )

    screen = cv2.cvtColor(
        screenshot,
        cv2.COLOR_BGRA2BGR
    )

    window_name = "SELECT RAY-BAN VIDEO REGION"

    x, y, w, h = cv2.selectROI(
        window_name,
        screen,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(window_name)

    if w < 2 or h < 2:
        raise RuntimeError(
            "No capture region selected."
        )

    return {
        "left": monitor["left"] + int(x),
        "top": monitor["top"] + int(y),
        "width": int(w),
        "height": int(h)
    }

# ---------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------

def main():
    setup_dpi()

    print("=" * 55)
    print("RAY-BAN META + YOLO11 + OPENAI LLM + VOICE")
    print("=" * 55)

    print("[SAFETY] Epson VT6 movement DISABLED")

    session, input_name, size = init_yolo()

    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        client = OpenAI(api_key=api_key)
        print("[LLM] OpenAI client initialized.")
    else:
        client = None
        print("[LLM] OPENAI_API_KEY missing.")

    print("[MIC] Backend:", AUDIO_BACKEND)

    print("""
KEYBOARD CONTROLS

S = Describe scene
T = Type a question
V = Ask with microphone
M = Select microphone
A = Toggle automatic narration
B = Test repeated speech
Q = Quit
""")

    automatic = False
    last_auto = time.monotonic()
    last_log = 0

    window_name = "Ray-Ban YOLO + OpenAI + Voice"

    with mss.mss() as sct:
        region = select_capture_region(sct)

        print("[CAPTURE]", region)

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL
        )

        try:
            while True:
                screenshot = np.asarray(
                    sct.grab(region)
                )

                frame = cv2.cvtColor(
                    screenshot,
                    cv2.COLOR_BGRA2BGR
                )

                try:
                    detections, annotated = yolo_detect(
                        frame,
                        session,
                        input_name,
                        size
                    )
                except Exception as e:
                    print("[YOLO ERROR]", e)
                    break

                cv2.putText(
                    annotated,
                    "S:Scene T:Type V:Voice M:Mic "
                    "A:Auto B:Test Q:Quit",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                if automatic:
                    cv2.putText(
                        annotated,
                        "AUTO ON",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                h, w = annotated.shape[:2]

                scale = min(
                    1.0,
                    1100 / w,
                    750 / h
                )

                if scale < 1:
                    preview = cv2.resize(
                        annotated,
                        None,
                        fx=scale,
                        fy=scale
                    )
                else:
                    preview = annotated

                cv2.imshow(
                    window_name,
                    preview
                )

                now = time.monotonic()

                if now - last_log >= 5:
                    counts = Counter(
                        d["label"] for d in detections
                    )

                    print(
                        "[YOLO]",
                        dict(counts)
                        if counts
                        else "No objects detected"
                    )

                    last_log = now

                pressed = cv2.waitKey(1) & 0xFF

                question = None

                if pressed in (ord("q"), 27):
                    break

                elif pressed == ord("b"):
                    speak(
                        "Hello. This is a speech test. "
                        "I can speak every time "
                        "you press the B key."
                    )

                elif pressed == ord("s"):
                    question = (
                        "What objects do you see, "
                        "and where are they?"
                    )

                elif pressed == ord("t"):
                    question = input(
                        "\nYour question: "
                    ).strip()

                elif pressed == ord("v"):
                    question = listen_question()

                elif pressed == ord("m"):
                    select_microphone()

                elif pressed == ord("a"):
                    automatic = not automatic

                    last_auto = time.monotonic()

                    print(
                        "[AUTO]",
                        "ON" if automatic else "OFF"
                    )

                if (
                    automatic
                    and time.monotonic() - last_auto
                    >= AUTO_INTERVAL_SECONDS
                ):
                    if not question:
                        question = (
                            "Describe the current "
                            "scene briefly."
                        )

                    last_auto = time.monotonic()

                if question:
                    if client is None:
                        print(
                            "[LLM] Configure "
                            "OPENAI_API_KEY first."
                        )
                    else:
                        # Capture a fresh frame after
                        # listening or typing.
                        fresh = np.asarray(
                            sct.grab(region)
                        )

                        latest_frame = cv2.cvtColor(
                            fresh,
                            cv2.COLOR_BGRA2BGR
                        )

                        latest_detections, _ = yolo_detect(
                            latest_frame,
                            session,
                            input_name,
                            size
                        )

                        answer = ask_llm(
                            client,
                            latest_frame,
                            latest_detections,
                            question
                        )

                        if answer:
                            print(
                                "[VOICE] Sending AI "
                                "answer to speaker..."
                            )

                            speak(answer)

        finally:
            cv2.destroyAllWindows()

    print("[INFO] Program stopped.")
    print("[SAFETY] No robot movement was executed.")


if __name__ == "__main__":
    main()
