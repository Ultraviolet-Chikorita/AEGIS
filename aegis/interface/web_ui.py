from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from aegis.interface.text_ui import FrameState


logger = logging.getLogger(__name__)


class WebInterface:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, log_level: str = "INFO") -> None:
        self.host = host
        self.port = port
        self.log_level = log_level.lower()
        self.narrative_buffer: list[str] = []
        self.frame_state: FrameState | None = None
        self._command_queue: asyncio.Queue[str] = asyncio.Queue()
        self._pending_commands: deque[str] = deque()
        self._clients: set[WebSocket] = set()
        self._event_log: list[dict[str, Any]] = []
        self._event_counter = 0
        self._minimap_state: dict[str, Any] = {"width": 100, "height": 100, "entities": []}

        self.app = FastAPI()
        self.app.get("/")(self._index)
        self.app.websocket("/ws")(self._ws)
        logger.info("WebInterface initialized at http://%s:%s", self.host, self.port)

    async def start(self) -> None:
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level=self.log_level)
        server = uvicorn.Server(config)
        logger.info("Web server starting on http://%s:%s (uvicorn log level=%s)", self.host, self.port, self.log_level)
        await server.serve()

    def render_frame(self, frame_state: FrameState) -> None:
        self.frame_state = frame_state
        payload = {"type": "frame", "frame": asdict(frame_state)}
        self._schedule_broadcast(payload)

    def stream_narrative(self, text: str, char_delay: float = 0.0) -> None:
        del char_delay
        self.narrative_buffer.append(text)
        if len(self.narrative_buffer) > 300:
            self.narrative_buffer = self.narrative_buffer[-300:]
        self._schedule_broadcast({"type": "narrative", "text": text})

    async def wait_for_command(self) -> str:
        return (await self._command_queue.get()).strip()

    def pop_command(self) -> str | None:
        if not self._pending_commands:
            return None
        return self._pending_commands.popleft()

    def update_minimap(self, payload: dict[str, Any]) -> None:
        self._minimap_state = payload
        self._schedule_broadcast({"type": "minimap", "minimap": payload})

    def log_event(self, summary: str, details: list[str], category: str) -> None:
        self._event_counter += 1
        event = {
            "id": self._event_counter,
            "summary": summary,
            "details": details,
            "category": category,
        }
        self._event_log.append(event)
        if len(self._event_log) > 300:
            self._event_log = self._event_log[-300:]
        self._schedule_broadcast({"type": "event", "event": event})

    def _command_ready(self) -> bool:
        if self.frame_state is None:
            return False
        return self.frame_state.cooldown_remaining_game_hours <= 0

    def _schedule_broadcast(self, payload: dict[str, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._broadcast(payload))
        except RuntimeError:
            return

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        if not self._clients:
            return

        dead: list[WebSocket] = []
        serialized = json.dumps(payload)
        for client in self._clients:
            try:
                await client.send_text(serialized)
            except Exception:
                logger.exception("Websocket broadcast failed; dropping client")
                dead.append(client)

        for client in dead:
            self._clients.discard(client)

    async def _index(self) -> HTMLResponse:
        return HTMLResponse(_HTML_PAGE)

    async def _ws(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        logger.info("Websocket client connected (clients=%s)", len(self._clients))

        await websocket.send_text(
            json.dumps(
                {
                    "type": "snapshot",
                    "frame": asdict(self.frame_state) if self.frame_state else None,
                    "narrative": self.narrative_buffer[-200:],
                    "events": self._event_log[-100:],
                    "minimap": self._minimap_state,
                }
            )
        )

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "command":
                    text = str(data.get("text", "")).strip()
                    if text:
                        if not self._command_ready():
                            logger.debug("Rejected command before cooldown completion")
                            self.stream_narrative("Command not ready yet. Wait for cooldown to complete.")
                            continue

                        if text.startswith("#"):
                            command = text[1:].strip()
                            if command:
                                logger.info("Accepted command from websocket: %s", command)
                                self._pending_commands.append(command)
                                await self._command_queue.put(command)
                            else:
                                self.stream_narrative("Push-to-talk activated, but no command followed #.")
                        else:
                            logger.debug("Rejected command missing # prefix")
                            self.stream_narrative("Press # before the command (e.g. # ask npc name).")
                elif data.get("type") == "ptt_command":
                    transcript = str(data.get("transcript", "")).strip()
                    if not transcript:
                        continue
                    if not self._command_ready():
                        logger.debug("Rejected ptt command before cooldown completion")
                        self.stream_narrative("Command not ready yet. Wait for cooldown to complete.")
                        continue
                    logger.info("Accepted voice command from websocket: %s", transcript)
                    self._pending_commands.append(transcript)
                    await self._command_queue.put(transcript)
                elif data.get("type") == "ptt":
                    if self._command_ready():
                        logger.debug("Push-to-talk armed by websocket client")
                        self.stream_narrative("Push-to-talk primed. Hold # and speak.")
                    else:
                        logger.debug("Push-to-talk denied due to cooldown")
                        self.stream_narrative("Push-to-talk unavailable until cooldown completes.")
        except WebSocketDisconnect:
            self._clients.discard(websocket)
            logger.info("Websocket client disconnected (clients=%s)", len(self._clients))
        except Exception:
            self._clients.discard(websocket)
            logger.exception("Websocket handler crashed (clients=%s)", len(self._clients))


_HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>AEGIS Live Interface</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --accent: #22c55e;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --danger: #f97316;
      --border: #1f2937;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: radial-gradient(1200px 800px at 15% -10%, #1e293b 0%, var(--bg) 55%);
      color: var(--text);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .app {
      width: min(980px, 100%);
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }
    .card {
      background: linear-gradient(180deg, rgba(17,24,39,0.95), rgba(2,6,23,0.95));
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
    }
    .status {
      display: grid;
      grid-template-columns: repeat(2, minmax(0,1fr));
      gap: 10px;
    }
    .label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
    .value { margin-top: 6px; font-size: 14px; }
    .bar {
      margin-top: 6px;
      width: 100%;
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: #1f2937;
    }
    .fill { height: 100%; background: linear-gradient(90deg, #16a34a, var(--accent)); transition: width 180ms linear; }
    #narrative {
      height: 52vh;
      overflow: auto;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: rgba(2,6,23,0.7);
      white-space: pre-wrap;
      line-height: 1.35;
      font-size: 14px;
    }
    .grid-2 { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 12px; }
    .command-row { display: flex; gap: 8px; align-items: center; }
    input {
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #0b1220;
      color: var(--text);
      padding: 10px 12px;
      font-size: 14px;
    }
    input { flex: 1; }
    #pttHint {
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
    }
    #connection { font-size: 12px; color: var(--muted); }
    #pttState { font-size: 12px; color: var(--muted); margin-top: 6px; }
    .warn { color: var(--danger); }
    #minimap {
      width: 100%;
      max-width: 420px;
      aspect-ratio: 10/7;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #020617;
      cursor: grab;
      touch-action: none;
    }
    #minimap.dragging {
      cursor: grabbing;
    }
    .minimap-toolbar {
      margin-top: 8px;
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .mini-btn {
      border: 1px solid var(--border);
      background: #0b1220;
      color: var(--text);
      border-radius: 8px;
      padding: 4px 8px;
      font-size: 12px;
      cursor: pointer;
    }
    .mini-btn:hover {
      border-color: #334155;
    }
    .mini-meta {
      color: var(--muted);
      font-size: 12px;
    }
    #events {
      max-height: 50vh;
      overflow: auto;
      display: grid;
      gap: 8px;
    }
    details.event {
      border: 1px solid var(--border);
      border-radius: 10px;
      background: rgba(2,6,23,0.7);
      padding: 8px;
    }
    details.event summary {
      cursor: pointer;
      font-weight: 600;
      font-size: 13px;
    }
    .event-tag {
      display: inline-block;
      margin-left: 8px;
      font-size: 11px;
      color: #0b1220;
      background: #22c55e;
      padding: 2px 6px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .event-details {
      margin: 8px 0 0 0;
      padding-left: 18px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.3;
    }
    .event-empty {
      color: var(--muted);
      font-size: 12px;
      border: 1px dashed var(--border);
      border-radius: 10px;
      padding: 10px;
    }
    @media (max-width: 720px) {
      .status { grid-template-columns: 1fr; }
      #narrative { height: 46vh; }
      .grid-2 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="card">
      <div class="status">
        <div>
          <div class="label">Possession Cooldown</div>
          <div id="cooldownText" class="value">--</div>
          <div class="bar"><div id="cooldownFill" class="fill" style="width:0%"></div></div>
        </div>
        <div>
          <div class="label">Host Resistance</div>
          <div id="resistanceText" class="value">--</div>
          <div class="bar"><div id="resistanceFill" class="fill" style="width:0%"></div></div>
        </div>
      </div>
      <div style="margin-top:10px" class="label">Host Willpower</div>
      <div id="willpowerText" class="value">--</div>
      <div class="bar"><div id="willpowerFill" class="fill" style="width:0%"></div></div>
      <div style="margin-top:10px" class="label">Possessed Host</div>
      <div id="hostSummary" class="value">--</div>
      <div id="connection" style="margin-top:10px">Connecting...</div>
      <div id="pttState">Hold # to push-to-talk when cooldown is ready.</div>
      <div id="pttHint">Microphone input while # is held becomes the command.</div>
    </section>

    <section class="grid-2">
      <section class="card">
        <div id="narrative"></div>
      </section>
      <section class="card">
        <div class="label">Minimap</div>
        <canvas id="minimap" width="420" height="294"></canvas>
        <div class="minimap-toolbar">
          <button id="zoomOut" class="mini-btn" type="button">-</button>
          <button id="zoomIn" class="mini-btn" type="button">+</button>
          <button id="zoomReset" class="mini-btn" type="button">Reset</button>
          <span id="zoomLabel" class="mini-meta">Zoom 1.00x</span>
          <span id="minimapMeta" class="mini-meta">Map feed waiting...</span>
          <span class="mini-meta">Wheel to zoom, drag to pan.</span>
        </div>
        <div class="label" style="margin-top:12px">Live Event Log</div>
        <div id="events"></div>
      </section>
    </section>

    <section class="card command-row">
      <input id="transcriptPreview" placeholder="Voice transcript will appear here..." readonly />
    </section>
  </main>

  <script>
    const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
    const ws = new WebSocket(wsUrl);

    const el = {
      narrative: document.getElementById('narrative'),
      transcriptPreview: document.getElementById('transcriptPreview'),
      connection: document.getElementById('connection'),
      pttState: document.getElementById('pttState'),
      cooldownText: document.getElementById('cooldownText'),
      cooldownFill: document.getElementById('cooldownFill'),
      resistanceText: document.getElementById('resistanceText'),
      resistanceFill: document.getElementById('resistanceFill'),
      willpowerText: document.getElementById('willpowerText'),
      willpowerFill: document.getElementById('willpowerFill'),
      hostSummary: document.getElementById('hostSummary'),
      minimap: document.getElementById('minimap'),
      zoomIn: document.getElementById('zoomIn'),
      zoomOut: document.getElementById('zoomOut'),
      zoomReset: document.getElementById('zoomReset'),
      zoomLabel: document.getElementById('zoomLabel'),
      minimapMeta: document.getElementById('minimapMeta'),
      events: document.getElementById('events'),
    };
    const ctx = el.minimap.getContext('2d');
    let commandReady = false;
    let pttHeld = false;
    let pendingTranscript = '';
    let currentMinimap = null;
    let hostLabel = 'host';
    let lastMinimapTick = null;
    let lastMinimapDrawAtMs = 0;
    const minimapView = {
      zoom: 1,
      minZoom: 1,
      maxZoom: 4,
      offsetX: 0,
      offsetY: 0,
      dragging: false,
      dragLastX: 0,
      dragLastY: 0,
    };
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = SpeechRecognition ? new SpeechRecognition() : null;

    if (recognition) {
      recognition.lang = 'en-US';
      recognition.continuous = true;
      recognition.interimResults = true;

      recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          transcript += event.results[i][0].transcript;
        }
        pendingTranscript = transcript.trim();
        el.transcriptPreview.value = pendingTranscript;
      };

      recognition.onerror = (event) => {
        el.pttState.textContent = `Speech capture error: ${event.error || 'unknown'}`;
        el.pttState.classList.add('warn');
      };

      recognition.onend = () => {
        const finalTranscript = pendingTranscript.trim();
        if (!pttHeld && finalTranscript && ws.readyState === WebSocket.OPEN && commandReady) {
          ws.send(JSON.stringify({ type: 'ptt_command', transcript: finalTranscript }));
          el.pttState.textContent = `Voice command sent: ${finalTranscript}`;
          el.pttState.classList.remove('warn');
        }
        pendingTranscript = '';
      };
    }

    function appendNarrative(text) {
      el.narrative.textContent += (el.narrative.textContent ? '\\n' : '') + text;
      el.narrative.scrollTop = el.narrative.scrollHeight;
    }

    function applyFrame(frame) {
      if (!frame) return;
      const total = Math.max(0.001, frame.cooldown_total_game_hours || 0.001);
      const ratio = Math.max(0, Math.min(1, 1 - ((frame.cooldown_remaining_game_hours || 0) / total)));
      const resistance = Math.max(0, Math.min(1, frame.host_resistance || 0));
      const willpower = Math.max(0, Math.min(1, frame.host_willpower || 0));
      commandReady = (frame.cooldown_remaining_game_hours || 0) <= 0;

      el.cooldownText.textContent = `${(frame.cooldown_remaining_game_hours || 0).toFixed(2)} game hours remaining`;
      el.cooldownFill.style.width = `${Math.round(ratio * 100)}%`;

      el.resistanceText.textContent = `${Math.round(resistance * 100)}%`;
      el.resistanceFill.style.width = `${Math.round(resistance * 100)}%`;

      el.willpowerText.textContent = `${Math.round(willpower * 100)}%`;
      el.willpowerFill.style.width = `${Math.round(willpower * 100)}%`;

      if (commandReady) {
        if (!pttHeld) {
          el.pttState.textContent = 'Command ready. Hold # to speak command.';
        }
        el.pttState.classList.remove('warn');
      } else {
        el.pttState.textContent = 'Cooldown active. Push-to-talk is locked.';
        el.pttState.classList.remove('warn');
      }
    }

    function drawMinimap(minimap) {
      if (!ctx || !minimap) return;
      currentMinimap = minimap;
      lastMinimapDrawAtMs = Date.now();
      const incomingTick = Number(minimap.tick);
      if (Number.isFinite(incomingTick)) {
        lastMinimapTick = incomingTick;
      }

      const host = minimap.host || null;
      if (host && el.hostSummary) {
        hostLabel = String(host.label || host.id || hostLabel);
        const hx = Number(host.x || 0).toFixed(1);
        const hy = Number(host.y || 0).toFixed(1);
        const hp = Number(host.health || 0).toFixed(0);
        const stamina = Number(host.stamina || 0).toFixed(0);
        const region = String(host.region || 'Unknown region');
        el.hostSummary.textContent = `${hostLabel} | ${region} | x:${hx} y:${hy} | HP ${hp} | STA ${stamina}`;
      }
      if (el.minimapMeta) {
        const tickText = Number.isFinite(lastMinimapTick) ? `tick ${lastMinimapTick}` : 'tick --';
        const asyncState = minimap.async_state || {};
        const commandState = asyncState.command_processing ? 'cmd:busy' : 'cmd:idle';
        const socialScanState = asyncState.social_scan_running ? 'scan:busy' : 'scan:idle';
        const socialPlans = Number(asyncState.pending_social_plans || 0);
        const lockedCount = Number(asyncState.locked_entity_count || 0);
        el.minimapMeta.textContent = `Map feed: ${tickText} | ${commandState} | ${socialScanState} | plans:${socialPlans} | locks:${lockedCount}`;
        el.minimapMeta.classList.remove('warn');
      }

      const width = minimap.width || 100;
      const height = minimap.height || 100;
      const kmWidth = Math.max(1, Number(minimap.km_width) || 10);
      const kmHeight = Math.max(1, Number(minimap.km_height) || 10);
      const factionColors = {
        guild: '#f59e0b',
        watch: '#60a5fa',
        locals: '#34d399',
      };

      const toCanvasX = (x) => {
        const base = (Math.max(0, Math.min(width, Number(x) || 0)) / width) * el.minimap.width;
        return (base * minimapView.zoom) + minimapView.offsetX;
      };
      const toCanvasY = (y) => {
        const base = (Math.max(0, Math.min(height, Number(y) || 0)) / height) * el.minimap.height;
        return (base * minimapView.zoom) + minimapView.offsetY;
      };

      ctx.clearRect(0, 0, el.minimap.width, el.minimap.height);
      ctx.fillStyle = '#020617';
      ctx.fillRect(0, 0, el.minimap.width, el.minimap.height);
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 2;
      ctx.strokeRect(1, 1, el.minimap.width - 2, el.minimap.height - 2);

      (minimap.regions || []).forEach((region) => {
        const points = Array.isArray(region.vertices) ? region.vertices : [];
        if (points.length >= 3) {
          ctx.beginPath();
          points.forEach((point, index) => {
            const px = toCanvasX(point.x);
            const py = toCanvasY(point.y);
            if (index === 0) {
              ctx.moveTo(px, py);
            } else {
              ctx.lineTo(px, py);
            }
          });
          ctx.closePath();
          ctx.fillStyle = region.color || '#1f2937';
          ctx.globalAlpha = 0.55;
          ctx.fill();
          ctx.globalAlpha = 1.0;
          ctx.strokeStyle = '#334155';
          ctx.lineWidth = 1.2;
          ctx.stroke();

          const cx = points.reduce((sum, point) => sum + (Number(point.x) || 0), 0) / points.length;
          const cy = points.reduce((sum, point) => sum + (Number(point.y) || 0), 0) / points.length;
          ctx.fillStyle = '#cbd5e1';
          ctx.font = '10px Segoe UI';
          ctx.fillText(region.label || region.id || 'zone', toCanvasX(cx) - 18, toCanvasY(cy));
          return;
        }

        const rx = toCanvasX(region.x || 0);
        const ry = toCanvasY(region.y || 0);
        const rw = ((region.w || 0) / width) * el.minimap.width;
        const rh = ((region.h || 0) / height) * el.minimap.height;

        ctx.fillStyle = region.color || '#1f2937';
        ctx.globalAlpha = 0.5;
        ctx.fillRect(rx, ry, rw, rh);
        ctx.globalAlpha = 1.0;

        ctx.strokeStyle = '#334155';
        ctx.strokeRect(rx, ry, rw, rh);

        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px Segoe UI';
        ctx.fillText(region.label || region.id || 'zone', rx + 6, ry + 14);
      });

      (minimap.roads || []).forEach((road) => {
        const waypoints = Array.isArray(road.waypoints) ? road.waypoints : [];
        if (waypoints.length < 2) return;
        ctx.strokeStyle = '#f1f5f9';
        ctx.globalAlpha = 0.4;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        waypoints.forEach((waypoint, index) => {
          const px = toCanvasX(waypoint.x);
          const py = toCanvasY(waypoint.y);
          if (index === 0) {
            ctx.moveTo(px, py);
          } else {
            ctx.lineTo(px, py);
          }
        });
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      });

      (minimap.buildings || []).forEach((building) => {
        const points = Array.isArray(building.footprint) ? building.footprint : [];
        if (points.length < 3) return;
        ctx.beginPath();
        points.forEach((point, index) => {
          const px = toCanvasX(point.x);
          const py = toCanvasY(point.y);
          if (index === 0) {
            ctx.moveTo(px, py);
          } else {
            ctx.lineTo(px, py);
          }
        });
        ctx.closePath();
        ctx.fillStyle = '#111827';
        ctx.globalAlpha = 0.78;
        ctx.fill();
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = '#93c5fd';
        ctx.lineWidth = 1;
        ctx.stroke();

        const cx = points.reduce((sum, point) => sum + (Number(point.x) || 0), 0) / points.length;
        const cy = points.reduce((sum, point) => sum + (Number(point.y) || 0), 0) / points.length;
        ctx.fillStyle = '#dbeafe';
        ctx.font = '9px Segoe UI';
        const label = String(building.name || building.id || 'building');
        ctx.fillText(label.slice(0, 18), toCanvasX(cx) + 4, toCanvasY(cy) - 4);
      });

      (minimap.hunting_areas || []).forEach((area) => {
        const x = toCanvasX(area.x || 0);
        const y = toCanvasY(area.y || 0);
        const radius = ((Number(area.radius) || 90) / width) * el.minimap.width;
        ctx.beginPath();
        ctx.arc(x, y, Math.max(10, radius), 0, Math.PI * 2);
        ctx.fillStyle = '#14532d';
        ctx.globalAlpha = 0.2;
        ctx.fill();
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = '#4ade80';
        ctx.setLineDash([5, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#bbf7d0';
        ctx.font = '10px Segoe UI';
        ctx.fillText(String(area.name || 'Hunt'), x + 8, y + 4);
      });

      (minimap.markets || []).forEach((market) => {
        const x = toCanvasX(market.x || 0);
        const y = toCanvasY(market.y || 0);
        ctx.beginPath();
        ctx.fillStyle = '#22d3ee';
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#a5f3fc';
        ctx.stroke();
      });

      (minimap.black_markets || []).forEach((market) => {
        const x = toCanvasX(market.x || 0);
        const y = toCanvasY(market.y || 0);
        ctx.beginPath();
        ctx.fillStyle = '#ef4444';
        ctx.rect(x - 4, y - 4, 8, 8);
        ctx.fill();
        ctx.strokeStyle = '#fecaca';
        ctx.strokeRect(x - 5, y - 5, 10, 10);
      });

      (minimap.combat_encounters || []).forEach((encounter) => {
        const x = toCanvasX(encounter.x || 0);
        const y = toCanvasY(encounter.y || 0);
        const rounds = Math.max(0, Number(encounter.rounds) || 0);

        ctx.beginPath();
        ctx.fillStyle = 'rgba(239,68,68,0.22)';
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#fecaca';
        ctx.font = '10px Segoe UI';
        ctx.fillText(`combat r${rounds}`, x + 12, y - 8);
      });

      ctx.strokeStyle = '#1f2937';
      for (let i = 1; i < 10; i += 1) {
        const x = (el.minimap.width / 10) * i;
        const y = (el.minimap.height / 10) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, el.minimap.height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(el.minimap.width, y);
        ctx.stroke();

        ctx.fillStyle = '#64748b';
        ctx.font = '9px Segoe UI';
        const xKm = Math.round((i / 10) * kmWidth);
        const yKm = Math.round((i / 10) * kmHeight);
        ctx.fillText(`${xKm}km`, x + 2, 10);
        ctx.fillText(`${yKm}km`, 2, y - 2);
      }

      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px Segoe UI';
      ctx.fillText(`Scale: ${kmWidth.toFixed(0)}km x ${kmHeight.toFixed(0)}km`, 8, el.minimap.height - 8);

      (minimap.entities || []).forEach((entity) => {
        const x = toCanvasX(entity.x || 0);
        const y = toCanvasY(entity.y || 0);
        ctx.beginPath();
        const isHost = entity.kind === 'host';
        const factionColor = factionColors[(entity.faction || '').toLowerCase()] || '#22c55e';
        ctx.fillStyle = isHost ? '#f97316' : factionColor;
        ctx.arc(x, y, isHost ? 6 : 4, 0, Math.PI * 2);
        ctx.fill();

        if (isHost) {
          ctx.strokeStyle = '#facc15';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(x, y, 9, 0, Math.PI * 2);
          ctx.stroke();
          ctx.lineWidth = 1;
        }

        ctx.fillStyle = '#e5e7eb';
        ctx.font = '11px Segoe UI';
        ctx.fillText(entity.label || entity.id || '?', x + 7, y - 7);
      });

      updateZoomLabel();
    }

    function clampMinimapView() {
      const maxPanX = Math.max(0, el.minimap.width * (minimapView.zoom - 1));
      const maxPanY = Math.max(0, el.minimap.height * (minimapView.zoom - 1));
      minimapView.offsetX = Math.max(-maxPanX, Math.min(0, minimapView.offsetX));
      minimapView.offsetY = Math.max(-maxPanY, Math.min(0, minimapView.offsetY));
    }

    function updateZoomLabel() {
      if (!el.zoomLabel) return;
      el.zoomLabel.textContent = `Zoom ${minimapView.zoom.toFixed(2)}x`;
    }

    function setMinimapZoom(nextZoom, anchorX = el.minimap.width / 2, anchorY = el.minimap.height / 2) {
      const clamped = Math.max(minimapView.minZoom, Math.min(minimapView.maxZoom, nextZoom));
      const prev = minimapView.zoom;
      if (Math.abs(clamped - prev) < 0.0001) return;

      const worldX = (anchorX - minimapView.offsetX) / prev;
      const worldY = (anchorY - minimapView.offsetY) / prev;

      minimapView.zoom = clamped;
      minimapView.offsetX = anchorX - (worldX * clamped);
      minimapView.offsetY = anchorY - (worldY * clamped);
      clampMinimapView();
      if (currentMinimap) drawMinimap(currentMinimap);
    }

    function resetMinimapView() {
      minimapView.zoom = 1;
      minimapView.offsetX = 0;
      minimapView.offsetY = 0;
      if (currentMinimap) drawMinimap(currentMinimap);
    }

    el.zoomIn?.addEventListener('click', () => setMinimapZoom(minimapView.zoom * 1.2));
    el.zoomOut?.addEventListener('click', () => setMinimapZoom(minimapView.zoom / 1.2));
    el.zoomReset?.addEventListener('click', resetMinimapView);

    el.minimap.addEventListener('wheel', (event) => {
      event.preventDefault();
      const rect = el.minimap.getBoundingClientRect();
      const scaleX = el.minimap.width / rect.width;
      const scaleY = el.minimap.height / rect.height;
      const anchorX = (event.clientX - rect.left) * scaleX;
      const anchorY = (event.clientY - rect.top) * scaleY;
      const factor = event.deltaY < 0 ? 1.14 : (1 / 1.14);
      setMinimapZoom(minimapView.zoom * factor, anchorX, anchorY);
    }, { passive: false });

    el.minimap.addEventListener('mousedown', (event) => {
      if (event.button !== 0) return;
      minimapView.dragging = true;
      minimapView.dragLastX = event.clientX;
      minimapView.dragLastY = event.clientY;
      el.minimap.classList.add('dragging');
    });

    window.addEventListener('mousemove', (event) => {
      if (!minimapView.dragging) return;
      const rect = el.minimap.getBoundingClientRect();
      const scaleX = el.minimap.width / rect.width;
      const scaleY = el.minimap.height / rect.height;
      const dx = (event.clientX - minimapView.dragLastX) * scaleX;
      const dy = (event.clientY - minimapView.dragLastY) * scaleY;
      minimapView.dragLastX = event.clientX;
      minimapView.dragLastY = event.clientY;
      minimapView.offsetX += dx;
      minimapView.offsetY += dy;
      clampMinimapView();
      if (currentMinimap) drawMinimap(currentMinimap);
    });

    window.addEventListener('mouseup', () => {
      if (!minimapView.dragging) return;
      minimapView.dragging = false;
      el.minimap.classList.remove('dragging');
    });

    function renderEvent(eventObj, prepend = true) {
      if (!eventObj) return;
      const category = (eventObj.category || 'general').toLowerCase();
      if (category === 'movement') return;

      const details = Array.isArray(eventObj.details) ? eventObj.details : [];
      const summaryText = String(eventObj.summary || '');
      const combined = `${summaryText}\\n${details.join('\\n')}`.toLowerCase();
      if (category === 'autonomy' && summaryText.toLowerCase().includes('host autonomous action')) {
        if (combined.includes('action: travel') || combined.includes('action: move')) return;
      }

      const item = document.createElement('details');
      item.className = 'event';
      item.open = true;

      const empty = el.events.querySelector('.event-empty');
      if (empty) empty.remove();

      const summary = document.createElement('summary');
      summary.textContent = eventObj.summary || 'Event';

      const tag = document.createElement('span');
      tag.className = 'event-tag';
      tag.textContent = category;
      summary.appendChild(tag);

      const list = document.createElement('ul');
      list.className = 'event-details';
      details.forEach((line) => {
        const li = document.createElement('li');
        li.textContent = line;
        list.appendChild(li);
      });

      item.appendChild(summary);
      item.appendChild(list);

      if (prepend) {
        el.events.prepend(item);
      } else {
        el.events.appendChild(item);
      }

      while (el.events.children.length > 180) {
        el.events.removeChild(el.events.lastChild);
      }
    }

    ws.addEventListener('open', () => {
      el.connection.textContent = 'Live connection established';
      el.connection.classList.remove('warn');
      el.pttState.textContent = 'Hold # to push-to-talk when cooldown is ready.';

      if (!recognition) {
        el.pttState.textContent = 'SpeechRecognition API unavailable in this browser. Use Chrome/Edge.';
        el.pttState.classList.add('warn');
      }
    });

    ws.addEventListener('close', () => {
      el.connection.textContent = 'Connection lost. Refresh to reconnect.';
      el.connection.classList.add('warn');
      el.pttState.textContent = 'Disconnected';
      el.pttState.classList.add('warn');
    });

    ws.addEventListener('message', (event) => {
      let data;
      try { data = JSON.parse(event.data); } catch { return; }
      if (data.type === 'snapshot') {
        (data.narrative || []).forEach(appendNarrative);
        applyFrame(data.frame);
        drawMinimap(data.minimap);
        (data.events || []).forEach((eventObj) => renderEvent(eventObj, false));
        if (el.events.children.length === 0) {
          const empty = document.createElement('div');
          empty.className = 'event-empty';
          empty.textContent = 'No interactions yet. The world is active and events will appear here.';
          el.events.appendChild(empty);
        }
      } else if (data.type === 'narrative') {
        appendNarrative(data.text || '');
      } else if (data.type === 'frame') {
        applyFrame(data.frame);
      } else if (data.type === 'minimap') {
        drawMinimap(data.minimap);
      } else if (data.type === 'event') {
        renderEvent(data.event, true);
        const summary = String((data.event && data.event.summary) || '').toLowerCase();
        if (summary.includes('command received')) {
          el.pttState.textContent = 'Command accepted. Host is processing...';
          el.pttState.classList.remove('warn');
        } else if (summary.includes('compulsion executed')) {
          el.pttState.textContent = 'Command executed. Waiting for cooldown.';
          el.pttState.classList.remove('warn');
        } else if (summary.includes('command processing failed')) {
          el.pttState.textContent = 'Command failed during processing. Retry with a clearer command.';
          el.pttState.classList.add('warn');
        }
      }
    });

    window.setInterval(() => {
      if (!el.minimapMeta) return;
      if (!currentMinimap) return;
      const staleMs = Date.now() - lastMinimapDrawAtMs;
      if (staleMs > 3000) {
        const tickText = Number.isFinite(lastMinimapTick) ? `tick ${lastMinimapTick}` : 'tick --';
        el.minimapMeta.textContent = `Map feed stale (${tickText}, ${Math.round(staleMs / 1000)}s old)`;
        el.minimapMeta.classList.add('warn');
      }
    }, 1000);

    window.addEventListener('keydown', (e) => {
      if (e.key !== '#' || e.repeat) return;
      if (!commandReady) {
        el.pttState.textContent = 'Cooldown active. Push-to-talk is locked.';
        el.pttState.classList.add('warn');
        e.preventDefault();
        return;
      }
      if (!recognition) {
        e.preventDefault();
        return;
      }

      pttHeld = true;
      pendingTranscript = '';
      el.transcriptPreview.value = '';
      el.pttState.textContent = 'Recording while # is held... release # to send voice command.';
      el.pttState.classList.remove('warn');

      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ptt', state: 'start' }));
      }

      try {
        recognition.start();
      } catch {
        // Ignore start race if browser still considers recognizer active.
      }

      e.preventDefault();
    });

    window.addEventListener('keyup', (e) => {
      if (e.key !== '#') return;
      if (!pttHeld) return;

      pttHeld = false;
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ptt', state: 'end' }));
      }

      if (recognition) {
        try {
          recognition.stop();
        } catch {
          // Recognizer may already be stopped.
        }
      }

      e.preventDefault();
    });
  </script>
</body>
</html>
"""
