# Young Counsel - Local Desktop App Deployment Plan
## Architecture and Packaging Guide for Client-Side Deployment (Streamlit Edition)
**Date:** 2026-02-28

---

## Why Local Deployment?

Legal software requires attorney-client privilege compliance — client data must never leave the client's machine. The current stack is already designed for this:

- **Ollama** — local LLM inference (no API calls to external servers)
- **FAISS** — local vector store (documents indexed on-device)
- **Tesseract** — local OCR engine (Arabic + English support)
- **Sentence Transformers** — local multilingual embeddings
- **FastAPI** — local backend server
- **Streamlit** — local frontend (pure Python, already built)

---

## Why Streamlit Instead of Next.js?

The original plan used Next.js, which requires:
- A separate `next build && next export` step
- Node.js installed or bundled on the client machine
- FastAPI to serve static files from a separate build output

**Streamlit eliminates all of this.** Since `frontend/app.py` is already built and
`streamlit` is already in `requirements.txt`, the entire app — backend and frontend —
is pure Python. PyInstaller bundles everything in one pass with no additional
build tools or runtimes required on the client machine.

---

## Why Not Docker for End Clients?

Docker requires WSL2 setup, BIOS virtualization, terminal usage, and 10-20 min image
pulls on first run. Non-technical clients (lawyers) find this too complex.
Docker is suitable for IT-managed on-premise deployments, not individual end users.

---

## Recommended Approach: Installer + System Tray App

Model this after LM Studio, Jan, and Msty — local AI apps that ship as a normal `.exe`
installer and run a system tray process that manages backend services.

### Architecture Flow

1. Client installs `YoungCounsel_Setup.exe`
2. Installer sets up: Ollama, Tesseract + Arabic pack, Python bundle (FastAPI + Streamlit)
3. Client double-clicks desktop icon
4. System tray app starts:
   - Ollama service
   - FastAPI backend (`localhost:8000`)
   - Streamlit frontend (`localhost:8501`)
5. Browser opens to `localhost:8501`
6. Client uses the app like a normal web app — fully local, no internet required after install

### Port Map

| Service   | Port  | Process         |
|-----------|-------|-----------------|
| FastAPI   | 8000  | uvicorn         |
| Streamlit | 8501  | streamlit run   |
| Ollama    | 11434 | ollama serve    |

---

## Build Steps

### Step 1 — Streamlit Configuration (0.5 days)

Add a `.streamlit/config.toml` so Streamlit does not try to open its own browser window
(the system tray launcher handles that) and binds to the correct port:

```toml
[server]
headless = true
port = 8501
address = "localhost"

[browser]
gatherUsageStats = false
```

This file must be included in the PyInstaller bundle at the correct path.

### Step 2 — Bundle Everything with PyInstaller (1-2 days)

A single PyInstaller bundle covers FastAPI, Streamlit, FAISS, sentence-transformers,
and Tesseract bindings. No separate Node.js build step. No static file export.

```
pyinstaller --onedir launcher/tray_app.py --name YoungCounsel \
    --add-data "frontend/app.py;frontend" \
    --add-data ".streamlit/config.toml;.streamlit" \
    --add-data "backend;backend"
```

**Common hidden imports to add to the `.spec` file:**

```python
hiddenimports=[
    "streamlit",
    "streamlit.runtime",
    "streamlit.web.cli",
    "faiss",
    "sentence_transformers",
    "torch",
    "uvicorn",
    "fastapi",
]
```

Note: `torch` and `faiss` binaries often need manual `binaries=[]` entries in the spec.
Test on a clean machine early to catch these.

### Step 3 — System Tray Launcher (1 day)

`launcher/tray_app.py` uses `pystray` to manage all three service processes.
It provides a right-click tray icon to open or quit the app.

```python
import subprocess, sys, os, webbrowser, pystray
from PIL import Image

BASE = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(__file__)

def start_services():
    # 1. Ollama
    subprocess.Popen(["ollama", "serve"])

    # 2. FastAPI via uvicorn
    subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "backend.main:app", "--host", "127.0.0.1", "--port", "8000"
    ], cwd=BASE)

    # 3. Streamlit
    subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        os.path.join(BASE, "frontend", "app.py"),
        "--server.headless", "true",
        "--server.port", "8501",
    ], cwd=BASE)

    # 4. Open browser after services are ready
    import time; time.sleep(3)
    webbrowser.open("http://localhost:8501")

def build_tray():
    icon_path = os.path.join(BASE, "assets", "icon.png")
    image = Image.open(icon_path)
    menu = pystray.Menu(
        pystray.MenuItem("Open Young Counsel",
                         lambda: webbrowser.open("http://localhost:8501")),
        pystray.MenuItem("Quit", lambda icon, _: icon.stop()),
    )
    return pystray.Icon("YoungCounsel", image, "Young Counsel", menu)

if __name__ == "__main__":
    start_services()
    build_tray().run()
```

### Step 4 — Inno Setup Installer (1 day)

Inno Setup wraps everything into `YoungCounsel_Setup.exe`. The Streamlit version is
simpler than Next.js — there is no separate build output directory to include.

```iss
[Setup]
AppName=Young Counsel
AppVersion=1.0
DefaultDirName={autopf}\YoungCounsel
DefaultGroupName=YoungCounsel

[Files]
; Main Python bundle
Source: "dist\YoungCounsel\*"; DestDir: "{app}"; Flags: recursesubdirs

; Ollama installer (runs silently during install)
Source: "vendors\OllamaSetup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

; Tesseract installer
Source: "vendors\tesseract-ocr-w64-setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Run]
Filename: "{tmp}\OllamaSetup.exe"; Parameters: "/S"; StatusMsg: "Installing Ollama..."
Filename: "{tmp}\tesseract-ocr-w64-setup.exe"; Parameters: "/S"; StatusMsg: "Installing Tesseract OCR..."
Filename: "{app}\YoungCounsel.exe"; Description: "Launch Young Counsel"; Flags: postinstall
```

### Step 5 — Test on a Clean Windows Machine (1-2 days)

Always test on a machine with no Python, no Node.js, and no development tools.
This catches missing DLLs, hidden import errors, and path issues.

Key things to verify:
- Streamlit UI loads at `localhost:8501`
- FastAPI responds at `localhost:8000/docs`
- Ollama model is available (pull on first run or bundle a small model)
- Arabic OCR works (Tesseract `ara.traineddata` present)
- System tray icon appears and quit works cleanly

---

## Handling the Ollama Dependency

Ollama has its own Windows installer (`OllamaSetup.exe`). Two strategies:

| Strategy | Approach | Recommended? |
|---|---|---|
| Bundle Ollama | Include `OllamaSetup.exe` in installer, run silently | Yes — seamless UX |
| Require pre-install | Tell client to install Ollama first | OK for tech-savvy clients |

---

## Arabic OCR on Client Machine

Tesseract installs with only English (`eng`) by default. The Arabic language pack
(`ara.traineddata`, ~15 MB) must be present for Arabic PDF support.

**Solution:** A startup check in `backend/utils/tessdata_setup.py` downloads
`ara.traineddata` from the Tesseract GitHub repo if missing, saving it to the
Tesseract `tessdata` directory automatically. No manual client action required.

---

## Estimated Timeline

| Task | Effort |
|---|---|
| Streamlit config (`config.toml`, headless mode) | 0.5 days |
| PyInstaller bundle (fixing hidden imports, spec file) | 1-2 days |
| System tray launcher | 1 day |
| Inno Setup installer script | 1 day |
| Testing on clean Windows machine | 1-2 days |
| **Total** | **~1 week** |

**Savings vs. Next.js plan:** ~3-5 days eliminated (no frontend build step,
no Node.js runtime, simpler PyInstaller spec).

---

## Deployment Strategy by Client Type

| Client Type | Best Approach |
|---|---|
| Non-technical individual users (solo lawyers) | Installer (.exe) + system tray |
| Law firms with IT department | Docker Compose (IT manages, not end user) |
| Cloud / SaaS deployment | Hosted web app (developer manages everything) |
| Privacy-first, fully offline | Installer + bundled Ollama model (no internet needed after install) |

---

*Young Counsel — Internal Planning Document — 2026-02-28*
