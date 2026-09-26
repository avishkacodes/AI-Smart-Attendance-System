# 🤖 AI-Powered Face Recognition & Anti-Spoof Attendance System

A real-time AI-based attendance system built with **Python, OpenCV, FaceNet, and anti-spoofing technology**. The system detects and verifies a person's face through a webcam, checks for spoofing attempts, records attendance automatically, and provides voice-based confirmation.

---

## 📌 Project Overview

Traditional attendance systems can be vulnerable to proxy attendance using photographs, screenshots, or other forms of presentation attacks.

This project was built to explore how **Computer Vision and AI** can be used to make attendance verification more automated and secure.

The system combines:

* Real-time face detection
* Face recognition using FaceNet embeddings
* Anti-spoofing verification
* Attendance automation
* Voice-based feedback
* Duplicate attendance prevention
* CSV-based attendance records
* Full-screen interactive UI

---

## ✨ Features

### 👤 Face Recognition

Recognizes registered individuals using **FaceNet-based facial embeddings** and cosine distance comparison.

### 🛡️ Anti-Spoofing

Performs an anti-spoofing check to help distinguish a live face from presentation attacks such as photographs or displayed images.

### 📸 Real-Time Processing

Processes webcam frames in real time using OpenCV.

### 📝 Automatic Attendance

Automatically records:

* Student name
* Date
* Time
* Attendance status

### 🚫 Duplicate Prevention

Prevents the same person from being marked multiple times during the same attendance session.

### 🔊 Voice Confirmation

Provides voice-based feedback after successful recognition and attendance marking.

### 🖥️ Interactive Interface

Displays recognition status, verification progress, notifications, and attendance results through a full-screen OpenCV interface.

---

## 🧠 How It Works

```text
              ┌───────────────┐
              │    Webcam     │
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │ Face Detection│
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │ Anti-Spoofing │
              │    Check      │
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │ Face Embedding │
              │   (FaceNet)    │
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │ Face Matching │
              │ Cosine Distance│
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │ Verification  │
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │   Attendance  │
              │    Marked     │
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │ Voice Feedback│
              └───────────────┘
```

---

## 🛠️ Tech Stack

| Technology             | Purpose                               |
| ---------------------- | ------------------------------------- |
| **Python**             | Core programming language             |
| **OpenCV**             | Computer vision and webcam processing |
| **DeepFace / FaceNet** | Face representation and recognition   |
| **SciPy**              | Cosine distance calculation           |
| **NumPy**              | Numerical processing                  |
| **pyttsx3**            | Voice feedback                        |
| **MiniFASNet**         | Anti-spoofing                         |
| **CSV**                | Attendance data storage               |

---

## 📂 Project Structure

```text
AI-Smart-Attendance-System/
│
├── attendance_system.py       # Main application
├── capture_faces.py           # Capture face images
├── encode_faces.py            # Generate face embeddings
├── anti_spoof_test.py         # Anti-spoofing testing
├── attendance.py               # Attendance-related functionality
├── recognize_faces.py         # Face recognition testing
├── test_cam.py                # Camera testing
│
├── anti_spoofing/
│   ├── src/
│   └── detection_model/
│
├── models/
│
├── screenshots/
│
├── requirements.txt
├── .gitignore
└── README.md
```

> Personal face datasets, biometric embeddings, attendance records, virtual environments, and local model files are excluded from the public repository.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/avishkacodes/AI-Smart-Attendance-System.git
```

### 2. Open the project folder

```bash
cd AI-Smart-Attendance-System
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

After installing the required dependencies and preparing the required face database and anti-spoofing model files:

```bash
python attendance_system.py
```

The system will access the webcam and begin the face detection, verification, recognition, and attendance workflow.

---

## 📋 Attendance Format

Attendance records are stored in CSV format.

Example:

```csv
Student Name,Date,Time,Status
Sample Student,01-01-2026,09:00:00,Present
```

The public repository does **not** contain real student attendance records.

---

## 🔐 Privacy & Security

This project works with biometric face data during enrollment and recognition.

For privacy reasons, the public repository does not include:

* Real face images
* Personal face embeddings
* Real attendance records
* Personal student information

Only the code and required project structure are shared publicly.

---

## 🚀 Future Scope

Some possible improvements for future versions include:

* 🌐 Web-based admin dashboard
* 🗄️ Database integration using SQL
* 📊 Attendance analytics and reports
* ☁️ Cloud-based data storage
* 👥 Multiple-camera support
* 🔐 Role-based access control
* 🗣️ Multilingual voice feedback
* 🧠 Improved liveness detection
* 📱 Web/mobile interface for attendance monitoring

---

## 📚 What I Learned

Building this project helped me understand how different AI and software components can work together in a real-time application.

Through the development process, I worked with:

* Computer Vision
* Face embeddings
* Face recognition
* Anti-spoofing
* Real-time webcam processing
* Data handling
* Python libraries and model integration
* Debugging and system-level problem solving

More importantly, I learned that building a working AI system involves much more than writing the initial code — testing, debugging, and solving unexpected problems are a major part of the process.

---

## 👩‍💻 Author

**Avishka Gaykar**

Computer Engineering Student | AI & Computer Vision Enthusiast

GitHub: [@avishkacodes](https://github.com/avishkacodes)

---

## ⭐ If you found this project interesting

Feel free to explore the code, raise an issue, or suggest improvements!
