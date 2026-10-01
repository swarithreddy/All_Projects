# Face Recognition Attendance Prototype

A Flask-based attendance prototype that combines ORB feature matching, browser-provided geolocation, and QR-code detection before recording an attendance row in an Excel workbook.

> This project is an experimental demonstration, not a secure identity or attendance system. Do not expose it to the public internet or use it to make high-stakes decisions.

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run the Application](#run-the-application)
- [Attendance Flow](#attendance-flow)
- [Data and Configuration](#data-and-configuration)
- [Important Limitations](#important-limitations)

## Features

- Select a subject from a timetable page
- Compare a camera image against reference face descriptors using OpenCV ORB
- Require the browser to provide location within 20 meters of the configured classroom coordinates
- Detect a QR code before recording attendance
- Append the matched name, subject, and timestamp to `attendance.xlsx`
- Limit failed face-verification attempts to three per session

## Technology Stack

| Area | Technology |
|------|------------|
| Web application | Flask |
| Face detection and QR detection | OpenCV |
| Face descriptor matching | ORB and Hamming-distance matching |
| Location distance | geopy |
| Attendance storage | pandas and Excel (`openpyxl`) |
| UI | HTML templates and CSS |

## Project Structure

```text
Face Recognition Attendance System/
├── app.py                    # Flask routes and verification workflow
├── known_faces/              # Add reference .jpg or .png images here
├── attendance.xlsx           # Created or updated after successful scans
├── static/
│   └── style.css             # Page styling
└── templates1/
    ├── login_timetable.html  # Timetable and subject selection
    ├── verify.html           # Face and browser location step
    ├── scan_qr.html          # QR scan step
    ├── success.html          # Successful attendance result
    ├── failure.html          # Location or QR failure result
    └── unsuccessfull.html    # Face verification retry page
```

## Requirements

- Python 3
- A webcam connected to the machine running Flask
- Browser geolocation permission for the verification page
- Python packages: Flask, geopy, opencv-python, pandas, and openpyxl

## Setup

From the project directory, create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install Flask geopy opencv-python pandas openpyxl
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install Flask geopy opencv-python pandas openpyxl
```

Create a `known_faces/` directory beside `app.py` and place one reference image per person in it. The image filename (without its extension) becomes the name saved to the attendance workbook. Use `.jpg` or `.png` images with a clearly visible face. The directory must exist before the application starts.

## Run the Application

Start the app from the project directory:

```bash
python app.py
```

Open the local Flask URL shown in the terminal, normally [http://127.0.0.1:5000](http://127.0.0.1:5000). The default Flask development server is intended only for local testing.

## Attendance Flow

1. The home page displays a client-side login form and timetable. The login form does not authenticate credentials.
2. Select a subject and continue to verification.
3. Allow browser geolocation, then submit the face-verification form.
4. The Flask host captures frames from camera device `0` and compares detected faces against the loaded reference descriptors.
5. If a face matches and the submitted location is within the configured distance, proceed to QR scanning.
6. The host camera scans for any decodable QR code. If one is detected, append the matched name, subject, and local timestamp to `attendance.xlsx`.

## Data and Configuration

- Reference images are loaded from `known_faces/` when `app.py` starts. Add or change images and restart the server to reload them.
- The classroom coordinates and allowed radius are constants in `app.py`; update them for the intended test location.
- Attendance is stored in `attendance.xlsx` in the current working directory. The app creates the workbook on the first successful scan.
- The app uses a hard-coded Flask secret key and starts with `debug=True`; both are development-only settings.

## Important Limitations

- The login page is only a visual step; it accepts any non-empty username and password and does not authenticate users.
- ORB matching is a basic computer-vision demonstration, not robust biometric recognition or liveness detection.
- The QR step accepts any QR code with decodable content; it does not validate a signed or expected attendance token.
- Camera capture occurs on the Flask server machine, not directly in the visitor's browser. Remote clients cannot use their own webcam through this implementation.
- Geolocation can be unavailable, denied, or spoofed; it should not be treated as proof of presence.
- The default secret key and Flask debug mode are unsafe for deployment. Do not run this prototype as a production attendance service.
- Face images and attendance records are sensitive personal data. Obtain consent and secure/delete the data appropriately.
