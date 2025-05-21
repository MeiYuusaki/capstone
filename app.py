import io
import cv2
import json
import time
import base64
import requests
import textwrap
import threading
import numpy as np
import tkinter as tk
import firebase_admin
from deepface import DeepFace
from PIL import Image, ImageTk
from google.cloud.firestore_v1 import DELETE_FIELD
from firebase_admin import credentials, storage, firestore

API_URL = ""
BEACON_URL = ""
TIMEOUT = 15
DATA = {
        "timeout": TIMEOUT,
        "mode": "static-enroll-high",
        "modality": "face-iris",
        "captureVolume": {
            "origin": {
            "x": -0.6,
            "y": 0,
            "z": 0.5
            },
            "size": {
            "width": 0.9,
            "height": 2.2,
            "depth": 1
            }
        },
        "trackingVolume": {
            "origin": {
            "x": -0.6,
            "y": 0,
            "z": 0
            },
            "size": {
            "width": 0.9,
            "height": 2.2,
            "depth": 1.51
            }
        },
        "antiSpoofingLevel": "none"
    }
RUN_LIVE_VIDEO = True
TRY_AGAIN_COUNT = 0
AGE_LIMIT = 7
CUSTOM_THRESHOLD = 0.55
MIN_MARGIN = 0.06
PASSPORT_INFO = {}
CAPTURE = 0
SUCCESS = []
ORI_TILT = False
HIGH_TILT = False

# Face Detection for images, used for drawing bounding boxes during live video
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Firebase Initialization
cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'capstone-sal.firebasestorage.app'
})
bucket = storage.bucket()
db = firestore.client()

avatar_path = "./avatar.png"
avatar_icon = None
avatar_success_path = "./avatar_success.png"
avatar_success_icon = None
has_shown_pass_page = False
loading_video = None
plane_video = None
start_time = None
first_checked = False
force_exit = False

class VideoPlayer:
    def __init__(self, parent, video_path, size=None):
        self.parent = parent
        self.video_path = video_path
        self.cap = None
        self.target_size = size
        self.playing = False

        self.video_label = tk.Label(parent, bg="#77AFD8")
        self.video_label.pack(expand=True)
        self.video_label.place(relx=0.5, rely=0.45, anchor="center")

    def start(self):
        if self.playing:
            return  # Already playing
        self.cap = cv2.VideoCapture(self.video_path)
        self.playing = True
        self.play_video()

    def stop(self):
        self.playing = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.video_label.config(image="")
        self.video_label.image = None

    def play_video(self):
        if not self.playing:
            return

        ret, frame = self.cap.read()
        if ret:
            if self.target_size:
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=image)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.parent.after(33, self.play_video)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.play_video()

# Check if people are entering the gate
def poll_open_front_flap():
    global force_exit
    try:
        doc_ref = db.collection("gates").document("sal")
        doc = doc_ref.get()
        if doc.exists:
            status = doc.to_dict().get("status")
            if status == "entering":
                print("Gate opening detected.")
                show_main_page()
                get_passport_info()
                print("Passports:", PASSPORT_INFO.keys())
                return
    except Exception as e:
        print("Error polling Firestore (open flap):", e)
        pass

    root.after(1000, poll_open_front_flap)

# Check if people have successfully entered the gate
def poll_close_front_flap():
    global PASSPORT_INFO, force_exit
    try:
        doc_ref = db.collection("gates").document("sal")
        doc = doc_ref.get()
        if doc.exists:
            status = doc.to_dict().get("status")
            if status == "entered" and PASSPORT_INFO:
                show_countdown_page()
                return
    except Exception as e:
        print("Error polling Firestore (close flap):", e)
        pass

    root.after(1000, poll_close_front_flap)

# Check if status == "exiting"
def poll_exiting():
    global has_shown_pass_page, force_exit
    try:
        doc_ref = db.collection("gates").document("sal")
        doc = doc_ref.get()
        if doc.exists:
            status = doc.to_dict().get("status")
            if status == "exiting" and not has_shown_pass_page:
                has_shown_pass_page = True
                force_exit = True
                print("Status is exiting — showing pass-through page")
                show_pass_through_page()
                return
    except Exception as e:
        print("Error polling Firestore (exiting):", e)
        pass

    root.after(1000, poll_exiting)

# Check if status == "exited"

def poll_exited():
    global plane_video, start_time, force_exit

    if start_time is None:
        start_time = time.time()  # Initialize when polling starts

    try:
        doc_ref = db.collection("gates").document("sal")
        doc = doc_ref.get()
        if doc.exists:
            status = doc.to_dict().get("status")
            if status == "exited" or (time.time() - start_time) >= 20:
                if plane_video is not None:
                    plane_video.stop()
                show_start_page()
                doc_ref.update({
                    "tilt_mode": "low",
                    "to_tilt": True
                })
                return
    except Exception as e:
        print("Error polling Firestore (exited):", e)
        pass

    root.after(1000, poll_exited)

def show_start_page(event=None):
    global PASSPORT_INFO, has_shown_pass_page, plane_video, loading_video, start_time, first_checked
    try:
        press_button_page.pack_forget()
        countdown_page.pack_forget()
        live_video_page.pack_forget()
        try_again_page.pack_forget()
        tilting_page.pack_forget()
        manual_auth_page.pack_forget()
        authenticated_page.pack_forget()
        pass_through_page.pack_forget()
    except:
        pass
    start_page.pack(fill="both", expand=True)
    PASSPORT_INFO.clear()
    has_shown_pass_page = False
    start_time = None
    first_checked = False
    users_ref = db.collection("gates").document("sal").collection("users")
    docs = users_ref.stream()
    for doc in docs:
        doc.reference.delete()
    user_doc = db.collection("gates").document("sal")
    user_doc.update({
        "alerts": "",
        "back_flap": False,
        "front_flap": False,
        "height_sensor": 0,
        "status": "",
        "timestamp": DELETE_FIELD
    })
    poll_open_front_flap()
    poll_exiting()

def show_main_page(event=None):
    start_page.pack_forget()
    press_button_page.pack(fill="both", expand=True)
    poll_close_front_flap()

def show_countdown_page(event=None):
    global SUCCESS, first_checked
    try_again_page.pack_forget()
    tilting_page.pack_forget()
    manual_auth_page.pack_forget()
    pass_through_page.pack_forget()
    press_button_page.pack_forget()
    authenticated_page.pack_forget()
    SUCCESS.clear()
    user_names = [info['name'] for info in PASSPORT_INFO.values()]
    draw_user_names_on_canvas(canvas, user_names)
    countdown_page.pack(fill="both", expand=True)
    if not first_checked:
        threading.Thread(target=first_check, daemon=True).start()
        threading.Thread(target=update_countdown, args=(5,), daemon=True).start()
    else:
        update_countdown(3)

def first_check():
    global ORI_TILT, HIGH_TILT, first_checked
    try:
        capture_url = f"{API_URL}/captures"
        response = requests.post(capture_url, json=DATA)
        token = json.loads(response.text)['token']
        time.sleep(2)
        response = requests.put(f"{API_URL}/captures/cancellation")
        fetch_url = f"{API_URL}/captures/{token}/result"
        data = requests.get(fetch_url).json()
        if len(data['persons']) == 0:
            user_doc = db.collection("gates").document("sal")
            user_doc.update({
                "to_tilt": True
            })
            tilt_mode = user_doc.get().to_dict().get('tilt_mode')
            if tilt_mode == 'original' and not ORI_TILT:
                ORI_TILT = True
            elif tilt_mode == 'high' and not HIGH_TILT:
                HIGH_TILT = True
    except Exception as e:
        print("First Check Failed!")
        pass
    first_checked = True
    return

def update_countdown(count):
    if has_shown_pass_page:
        return
    if count > 0:
        canvas.itemconfig(countdown_text, text=str(count))
        root.after(1000, update_countdown, count - 1)
    else:
        root.after(500, show_live_video_page)

def show_live_video_page():
    if has_shown_pass_page:
        return
    countdown_page.pack_forget()
    live_video_page.pack(fill="both", expand=True)
    call_captures()

def show_loading_page():
    global RUN_LIVE_VIDEO, loading_video
    RUN_LIVE_VIDEO = False
    if has_shown_pass_page:
        return
    live_video_page.pack_forget()
    if loading_video is None:
        loading_video = VideoPlayer(loading_page, "./loading.gif")
    loading_video.start()
    loading_page.pack(fill="both", expand=True)

def show_authenticated_page():
    global SUCCESS, ORI_TILT, HIGH_TILT, CAPTURE, loading_video
    if has_shown_pass_page:
        return
    loading_page.pack_forget()
    if loading_video is not None:
        loading_video.stop()
    authenticated_page.pack(fill="both", expand=True)
    draw_user_names_on_canvas(auth_canvas, SUCCESS, icon_type="success")
    if all(age['age'] < AGE_LIMIT for age in PASSPORT_INFO.values()):
        print("Check for kids!")
        temp_str = "[" + ", ".join([passport_no for passport_no in PASSPORT_INFO.keys()]) + "]"
        user_doc = db.collection("gates").document("sal")
        user_doc.update({
            "alerts": f"[error] manual authentication for {temp_str}"
        })
        root.after(0, show_manual_auth_page)
    else:
        if len(SUCCESS) == CAPTURE:
            user_doc = db.collection("gates").document("sal")
            tilt_mode = user_doc.get().to_dict().get('tilt_mode')
            if tilt_mode == 'original' and not ORI_TILT:
                ORI_TILT = True
                user_doc.update({
                    "to_tilt": True
                })
                tilting_page.pack(fill="both", expand=True)
            elif tilt_mode == 'high' and not HIGH_TILT:
                HIGH_TILT = True
                user_doc.update({
                    "to_tilt": True
                })
                tilting_page.pack(fill="both", expand=True)
        root.after(1000, show_countdown_page)

def show_try_again_page():
    global RUN_LIVE_VIDEO, TRY_AGAIN_COUNT, PASSPORT_INFO, ORI_TILT, HIGH_TILT, CAPTURE, loading_video
    RUN_LIVE_VIDEO = False
    if has_shown_pass_page:
        return
    live_video_page.pack_forget()
    loading_page.pack_forget()
    if loading_video is not None:
        loading_video.stop()
    user_doc = db.collection("gates").document("sal")
    if TRY_AGAIN_COUNT > 3:
        print("Tried 3 times!")
        user_doc.update({
            "alerts": "[error] redirect to manual counter"
        })
        root.after(0, show_manual_auth_page)
    elif all(age['age'] < AGE_LIMIT for age in PASSPORT_INFO.values()):
        print("Check for kids!")
        temp_str = "[" + ", ".join([passport_no for passport_no in PASSPORT_INFO.keys()]) + "]"
        user_doc.update({
            "alerts": f"[error] manual authentication for {temp_str}"
        })
        root.after(0, show_manual_auth_page)
    else:
        if len(SUCCESS) == CAPTURE:
            tilt_mode = user_doc.get().to_dict().get('tilt_mode')
            if tilt_mode == 'original' and not ORI_TILT:
                ORI_TILT = True
                user_doc.update({
                    "to_tilt": True
                })
                tilting_page.pack(fill="both", expand=True)
            elif tilt_mode == 'high' and not HIGH_TILT:
                HIGH_TILT = True
                user_doc.update({
                    "to_tilt": True
                })
                tilting_page.pack(fill="both", expand=True)
        else:
            TRY_AGAIN_COUNT += 1
            try_again_page.pack(fill="both", expand=True)
        root.after(1000, show_countdown_page)
    
def show_manual_auth_page():
    global TRY_AGAIN_COUNT, SUCCESS
    TRY_AGAIN_COUNT = 0
    SUCCESS.clear()
    if has_shown_pass_page:
        return
    data = {
        "periodicColors": [
            {
                "color": "#FF0000",
                "intensity": 100,
                "duration": 3
            }
        ]
    }
    requests.put(BEACON_URL, json=data)
    live_video_page.pack_forget()
    try_again_page.pack_forget()
    authenticated_page.pack_forget()
    manual_auth_page.pack(fill="both", expand=True)

def show_pass_through_page():
    global TRY_AGAIN_COUNT, ORI_TILT, HIGH_TILT, plane_video, loading_video, force_exit
    TRY_AGAIN_COUNT = 0
    ORI_TILT = False
    HIGH_TILT = False
    data = {
        "periodicColors": [
            {
                "color": "#000000",
                "intensity": 100,
                "duration": 3
            }
        ]
    }
    requests.put(BEACON_URL, json=data)
    start_page.pack_forget()
    press_button_page.pack_forget()
    countdown_page.pack_forget()
    loading_page.pack_forget()
    if loading_video is not None:
        loading_video.stop()
    live_video_page.pack_forget()
    try_again_page.pack_forget()
    tilting_page.pack_forget()
    manual_auth_page.pack_forget()
    authenticated_page.pack_forget()
    force_exit = False
    if plane_video is None:
        plane_video = VideoPlayer(pass_through_page, "./pass_through.gif", size=(1024, 500))
    plane_video.start()
    pass_through_page.pack(fill="both", expand=True)
    user_doc = db.collection("gates").document("sal")
    user_doc.update({
        "back_flap": True,
        "status": "exiting"
    })
    poll_exited()

# Main facial recognition process
def call_captures():
    capture_url = f"{API_URL}/captures"
    response = requests.post(capture_url, json=DATA)
    response.raise_for_status()
    token = json.loads(response.text)['token']

    # Concurrently call fetch_data to authenticate, and fetch_live_video for live video feed in GUI
    threading.Thread(target=fetch_data, args=(token,), daemon=True).start()
    threading.Thread(target=fetch_live_video, daemon=True).start()

# Helper function to extract passport images from Firestore database
def get_passport_info():
    global PASSPORT_INFO
    user_docs = db.collection("gates").document("sal").collection("users").get()
    for doc in user_docs:
        info = doc.to_dict()
        passport_url = info.get("passport_image")
        age = info.get("age")
        name = info.get("name")
        response = requests.get(passport_url)
        if response.status_code == 200:
            img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            PASSPORT_INFO[doc.id] = {"image": img, 
                                       "age": age,
                                      "name": name}

# Main process to extract live images and authenticate them
def fetch_data(token):
    global PASSPORT_INFO, CAPTURE
    print(f"Token: {token}")
    fetch_url = f"{API_URL}/captures/{token}/result"
    start_time = time.time()
    
    while time.time() - start_time < TIMEOUT:
        first_data = requests.get(fetch_url).json()
        time.sleep(2)
        second_data = requests.get(fetch_url).json()
        if len(second_data['persons']) == 0:
            root.after(0, show_try_again_page)
            response = requests.put(f"{API_URL}/captures/cancellation")
            response.raise_for_status()
            return
        else:
            if len(first_data['persons']) == len(second_data['persons']):
                response = requests.put(f"{API_URL}/captures/cancellation")
                response.raise_for_status()
                print("Number of people checked!")
                CAPTURE = len(second_data['persons'])
                root.after(0, show_loading_page)
                authenticated = authenticate_users(second_data)
                print("Authentication Done!")
                if authenticated:
                    print("All authenticated! Please pass through!")
                    root.after(0, show_pass_through_page)
                else:
                    print("All not authenticated!")
                    if not SUCCESS:
                        root.after(0, show_try_again_page)
                    else:
                        root.after(0, show_authenticated_page)
                return

# Main process to show live video with bounding boxes in GUI
def fetch_live_video():
    global RUN_LIVE_VIDEO
    data = {
        "periodicColors": [
            {
                "color": "#0000FF",
                "intensity": 100,
                "duration": 3
            }
        ]
    }
    requests.put(BEACON_URL, json=data)
    start_time = time.time()
    RUN_LIVE_VIDEO = True
    while RUN_LIVE_VIDEO and time.time() - start_time < TIMEOUT:
        try:
            response = requests.get(f"{API_URL}/live/image", stream=True)
            if response.status_code == 200:
                image_data = base64.b64decode(response.json().get('data', ''))
                image = Image.open(io.BytesIO(image_data))
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                image_cv = cv2.flip(image_cv, 1)
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(81, 81),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                valid_faces = filter_faces_by_size(faces)
                valid_faces = sorted(valid_faces, key=lambda box: box[2] * box[3], reverse=True)[:4]
                for (x, y, w, h) in valid_faces:
                    cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
                image_with_box = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
                video_frame = ImageTk.PhotoImage(image_with_box)
                live_video_label.config(image=video_frame)
                live_video_label.image = video_frame
        except Exception as e:
            pass

    RUN_LIVE_VIDEO = False
    data = {
        "periodicColors": [
            {
                "color": "#000000",
                "intensity": 100,
                "duration": 3
            }
        ]
    }
    requests.put(BEACON_URL, json=data)
    return

# Main authentication process
def authenticate_users(data):
    global PASSPORT_INFO, CUSTOM_THRESHOLD, MIN_MARGIN, SUCCESS
    try:
        for person in data['persons']:
            face_image_base64 = person['person']['face']['image'][0]['content']['data']
            left_iris_image_base64 = person['person']['iris']['left']['image'][0]['content']['data']
            right_iris_image_base64 = person['person']['iris']['right']['image'][0]['content']['data']
            face_image = base64_to_numpy(face_image_base64)

            match_scores = []

            for passport_no, passport_info in PASSPORT_INFO.items():
                if passport_info["age"] <= AGE_LIMIT:
                    print("Child Detected! Pass")
                    continue

                result = DeepFace.verify(face_image, passport_info["image"], model_name="ArcFace", enforce_detection=False)
                distance = result["distance"]
                match_scores.append((passport_no, distance, passport_info))

            if not match_scores:
                continue

            # Sort matches by distance
            match_scores.sort(key=lambda x: x[1])
            best = match_scores[0]
            second_best = match_scores[1] if len(match_scores) > 1 else (None, float('inf'), None)

            is_confident = (best[1] < CUSTOM_THRESHOLD) and ((second_best[1] - best[1]) > MIN_MARGIN)

            if is_confident:
                passport_no = best[0]
                passport_info = best[2]
                upload_base64(face_image_base64, passport_no, "current_image")
                upload_base64(left_iris_image_base64, passport_no, "left_iris")
                upload_base64(right_iris_image_base64, passport_no, "right_iris")
                user_doc = db.collection("gates").document("sal").collection("users").document(passport_no)
                user_doc.update({
                    "scan_status": "success"
                })
                SUCCESS.append(PASSPORT_INFO[passport_no]["name"])
                del PASSPORT_INFO[passport_no]

    except Exception as e:
        print("Error:", e)
        pass

    return not PASSPORT_INFO

# Helper functions
def upload_base64(image_base64, user_passport_no, image_type):
    image_data = base64.b64decode(image_base64)
    blob = bucket.blob(f"images/{user_passport_no}_{image_type}.bmp")
    blob.upload_from_string(image_data, content_type="image/bmp")
    user_doc = db.collection("gates").document("sal").collection("users").document(user_passport_no)
    user_doc.update({image_type: blob.public_url})
def base64_to_numpy(base64_string, max_retries=20, retry_interval=0.1):
    for attempt in range(max_retries):
        try:
            if not base64_string or len(base64_string) < 100:
                time.sleep(retry_interval)
                continue
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        except Exception as e:
            # Optionally print only on last attempt
            if attempt == max_retries - 1:
                print("base64_to_numpy failed after retries:", e)
            time.sleep(retry_interval)
    return None
def draw_user_names_on_canvas(canvas, names, icon_type="default"):
    global avatar_icon, avatar_success_icon

    if icon_type == "default":
        if avatar_icon is None:
            raw_image = Image.open(avatar_path).convert("RGBA")
            bbox = raw_image.getbbox()
            cropped = raw_image.crop(bbox)
            avatar_image = cropped.resize((60, 60))
            avatar_icon = ImageTk.PhotoImage(avatar_image)
        icon = avatar_icon

    elif icon_type == "success":
        if avatar_success_icon is None:
            raw_image = Image.open(avatar_success_path).convert("RGBA")
            bbox = raw_image.getbbox()
            cropped = raw_image.crop(bbox)
            avatar_image = cropped.resize((60, 60))
            avatar_success_icon = ImageTk.PhotoImage(avatar_image)
        icon = avatar_success_icon

    canvas.delete("user_text")
    num_users = len(names)
    if num_users == 0:
        return
    
    center_x = 512
    top_y = 150
    gap_x = 220
    vertical_spacing = 200
    positions = []

    if num_users == 1:
        positions = [(center_x, top_y + 100)]
    elif num_users == 2:
        positions = [
            (center_x - 220, top_y + 100),
            (center_x + 220, top_y + 100)
        ]
    elif num_users == 3:
        positions = [
            (center_x, top_y),
            (center_x - gap_x, top_y + vertical_spacing),
            (center_x + gap_x, top_y + vertical_spacing)
        ]
    else:
        grid_cols = 2
        for i in range(min(4, num_users)):
            col = i % grid_cols
            row = i // grid_cols
            x = 300 + col * 420
            y = top_y + row * vertical_spacing
            positions.append((x, y))

    for i, name in enumerate(names):
        if i >= len(positions):
            break
        x, y = positions[i]
        wrapped_name = wrap_name(name)

        canvas.create_image(x, y, image=icon, anchor="center", tags="user_text")
        canvas.create_text(
            x, y + 50,
            text=wrapped_name,
            fill="black",
            font=("Nunito", 25),
            anchor="n",
            justify="center",
            tags="user_text"
        )
def filter_faces_by_size(faces):
    valid_faces = []
    for (x, y, w, h) in faces:
        aspect_ratio = w / h
        if 0.75 < aspect_ratio < 1.3:
            valid_faces.append((x, y, w, h))
    return valid_faces
def wrap_name(name, width=21):
    return '\n'.join(textwrap.wrap(name, width=width, break_long_words=False))

# GUI Setup
root = tk.Tk()
# root.attributes('-fullscreen', True)
root.title("Biometric Authentication")
root.geometry("1024x600")
root.resizable(False, False)
# root.overrideredirect(True)

# ICA Logo Page
start_page = tk.Frame(root, bg="#77AFD8")
# start_image_path = "./ICA Logo.png"
start_image_path = "./sal2.png"
original_start_image = Image.open(start_image_path).resize((1024, 600))
welcome_image = ImageTk.PhotoImage(original_start_image)
welcome_label = tk.Label(start_page, image=welcome_image, bg="#77AFD8")
welcome_label.place(relx=0.5, rely=0.5, anchor="center")
user_doc = db.collection("gates").document("sal")
user_doc.update({
    "tilt_mode": "low",
    "to_tilt": True
})
data = {
    "periodicColors": [
        {
            "color": "#000000",
            "intensity": 100,
            "duration": 3
        }
    ]
}
beacon_response = requests.put(BEACON_URL, json=data)
show_start_page()

# Press Button Page
press_button_page = tk.Frame(root, bg="#77AFD8")
press_button_image_path = "./press_button.png"
original_bg_image = Image.open(press_button_image_path).resize((1024, 500))
bg_image = ImageTk.PhotoImage(original_bg_image)
image_label = tk.Label(press_button_page, image=bg_image, bg="#77AFD8")
image_label.place(x=0, y=0, relwidth=1, height=500)
bottom_frame = tk.Frame(press_button_page, bg="#000146", height=100, width=1024)
bottom_frame.place(x=0, y=500)
bottom_label = tk.Label(bottom_frame, text="Please Press Button to Close Entry Flapper", 
                         font=("Nunito", 20), bg="#000146", fg="white")
bottom_label.place(relx=0.5, rely=0.5, anchor="center")

# Countdown Page
countdown_page = tk.Frame(root, bg="#2E2D2D")
canvas = tk.Canvas(countdown_page, width=1024, height=600, bg="#77AFD8", highlightthickness=0)
canvas.pack(fill="both", expand=True)
canvas.create_rectangle(0, 0, 1024, 60, fill="#77AFD8", outline="")
countdown_text = canvas.create_text(512, 500, text="5", font=("Nunito", 50), fill="black", anchor="center")
canvas.create_text(512, 60, text="Please Look at the Camera", font=("Nunito", 40), fill="black")
user_names = [info['name'] for info in PASSPORT_INFO.values()]
draw_user_names_on_canvas(canvas, user_names)

# Live Video Page
live_video_page = tk.Frame(root, bg="#000000")
live_video_label = tk.Label(live_video_page, text="Live Video Page", font=("Nunito", 30), fg="white", bg="#000000")
live_video_label.pack(expand=True)

# Loading Page
loading_page = tk.Frame(root, bg="#77AFD8")
loading_label = tk.Label(loading_page, text="Loading...", font=("Nunito", 50), fg="black", bg="#77AFD8", bd=5)
loading_label.pack(expand=True)
loading_label.place(relx=0.5, rely=0.8, anchor="center")

# Authenticated Page
authenticated_page = tk.Frame(root, bg="#2E2D2D")
auth_canvas = tk.Canvas(authenticated_page, width=1024, height=600, bg="#77AFD8", highlightthickness=0)
auth_canvas.pack(fill="both", expand=True)
auth_canvas.create_rectangle(0, 0, 1024, 60, fill="#77AFD8", outline="")
auth_canvas.create_text(512, 60, text="Scan Success", font=("Nunito", 40), fill="black", anchor="center")

# Try Again Page
try_again_page = tk.Frame(root, bg="#FF0000")
try_again_label = tk.Label(try_again_page, text="Try Again", font=("Nunito", 30), fg="white", bg="#FF0000")
try_again_label.pack(expand=True)

# Tilting Page
tilting_page = tk.Frame(root, bg="#77AFD8")
tilting_label = tk.Label(tilting_page, text="Tilting...", font=("Nunito", 50), fg="black", bg="#77AFD8", bd=5)
tilting_label.pack(expand=True)
tilting_label.place(relx=0.5, rely=0.5, anchor="center")

# Manual Auth Page
manual_auth_page = tk.Frame(root, bg="#FFA500")
manual_auth_label = tk.Label(manual_auth_page, text="Please wait for assistance...", font=("Nunito", 30), fg="black", bg="#FFA500")
manual_auth_label.pack(expand=True)

# Pass Through Page
pass_through_page = tk.Frame(root, bg="#77AFD8")
pass_through_label = tk.Label(pass_through_page, text="Please Go Through The Gate", font=("Nunito", 30), fg="white", bg="#008000")
pass_through_label.pack(expand=True)
pass_bottom_frame = tk.Frame(pass_through_page, bg="#000146", height=100, width=1024)
pass_bottom_frame.place(x=0, y=500)
pass_bottom_label = tk.Label(pass_bottom_frame, text="Have a pleasant journey!", 
                         font=("Nunito", 20), bg="#000146", fg="white")
pass_bottom_label.place(relx=0.5, rely=0.5, anchor="center")

root.mainloop()