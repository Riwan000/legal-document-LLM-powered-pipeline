# Logging Implementation — Explained

## The Core Problem

The app runs **offline on a user's machine**. Logs are written to `logs/chat.log`. But as the developer, you can't see that file — it only exists on their machine.

The question becomes: **how do you get that file from their machine to you?**

---

## Option A — "Send Logs" Button

```python
if st.button("Send usage logs"):
    data = Path("logs/chat.log").read_bytes()
    requests.post(LOG_ENDPOINT, files={"log": data}, timeout=10)
```

**What's happening:**

1. `Path("logs/chat.log").read_bytes()` — reads the entire log file as raw bytes
2. `requests.post(...)` — makes an HTTP POST request to your server, attaching the file
3. `files={"log": data}` — sends it as a multipart form upload (same as a browser uploading a file)
4. `timeout=10` — if no internet, don't hang forever — give up after 10 seconds

**Key insight:** The app works offline. Internet is only needed at the moment the user clicks this button. If they're offline, it fails gracefully and tells them.

---

## Option B — Background Auto-Sync

Three concepts at work:

### Concept 1: Only send what's new (the marker file)

```python
SENT_MARKER = Path("logs/.last_sent")

last = int(SENT_MARKER.read_text()) if SENT_MARKER.exists() else 0
unsent = lines[last:]
```

`logs/chat.log` grows over time — new lines appended at the bottom. You don't want to re-send everything every time.

So you keep a **bookmark**: `logs/.last_sent` stores a single number — the line count already sent last time.

```
logs/chat.log     (100 lines total)
logs/.last_sent   (contains "80")
```

Next sync: read lines 80–100, send only those 20 new lines, update marker to "100".

This is called **log tailing** — the same concept `tail -f` uses.

### Concept 2: Run it in a background thread

```python
threading.Thread(target=_sync, daemon=True).start()
```

- `threading.Thread(target=_sync)` — runs `_sync()` in a separate thread, so it doesn't block the app from starting
- `daemon=True` — a daemon thread is automatically killed when the main program exits. Without this, the app would hang on close waiting for the network request to finish.

### Concept 3: Fail silently

```python
except Exception:
    pass  # offline or failed — silently skip
```

The user's experience should never be degraded because log syncing failed. No error popups. No slowdowns. If the network isn't available, skip it and try again next startup.

---

## Option C — Your Receiver Server

```python
@app.post("/logs")
async def receive_logs(lines: list[str]):
    out = Path(f"received/{datetime.utcnow().date()}.jsonl")
    with out.open("a") as f:
        f.write("\n".join(lines) + "\n")
```

**What's happening:**

- `datetime.utcnow().date()` — creates a new file per day, e.g. `received/2026-03-03.jsonl`
- `out.open("a")` — opens in **append** mode, so logs from multiple users accumulate in the same daily file rather than overwriting each other
- `.jsonl` — JSON Lines format: one JSON object per line, easy to stream and parse

---

## How the Three Options Connect

```
[User's machine]                     [Your server]

logs/chat.log  ──Option A──►  POST /logs  ──► received/2026-03-03.jsonl
               ──Option B──►  (background)
```

Option A and B are **different triggers** for the same action — sending to the same endpoint (Option C).

- Option A: user-initiated, explicit
- Option B: automatic, on startup, incremental

---

## The Privacy Concern

The logs contain `original_query` fields like:

```json
{"original_query": "what are the termination clauses in this NDA..."}
```

That's a user's **legal question about a real document**. Auto-uploading without telling them is a privacy violation — and in legal software, potentially a professional conduct issue (lawyer-client privilege, etc.).

**Fix:** show a consent screen on first run. Store their choice. Only sync if they said yes.

---

## Summary

| Concept | What it solves |
|---|---|
| Marker file | Avoid re-sending already-sent logs |
| Daemon thread | Don't block app startup or shutdown |
| Silent exception catch | Don't break UX when offline |
| Append-mode file on server | Handle multiple users writing simultaneously |
| Per-day filenames | Natural log rotation without extra tooling |
| Consent gate | Respect user privacy, especially in legal context |
