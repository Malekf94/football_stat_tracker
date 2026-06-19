"""
Football Stat Tracker — GUI
Open via  "Open Football Tracker.bat"  in this folder.
"""

import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _DND = True
except ImportError:
    _DND = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON     = os.path.join(SCRIPT_DIR, "venv", "Scripts", "python.exe")
ANALYZE    = os.path.join(SCRIPT_DIR, "analyze.py")
SETUP_GOALS = os.path.join(SCRIPT_DIR, "setup_goals.py")
CORRECT    = os.path.join(SCRIPT_DIR, "correct.py")


class App(TkinterDnD.Tk if _DND else tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Football Stat Tracker")
        self.geometry("860x580")
        self.minsize(760, 500)
        self._video_path = tk.StringVar()
        self._running = False
        self._build_ui()
        if _DND:
            self._drop_label.drop_target_register(DND_FILES)
            self._drop_label.dnd_bind("<<Drop>>", self._on_drop)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = dict(padx=12, pady=5)

        # ── Video selection ───────────────────────────────────────────
        video_frame = ttk.LabelFrame(self, text="  Video  ", padding=10)
        video_frame.pack(fill="x", **pad)

        hint = "Drag a video file here" if _DND else "Click Browse to select a video"
        self._drop_label = tk.Label(
            video_frame,
            text=hint,
            relief="groove",
            bg="#1a1a2e",
            fg="#7a8899",
            height=3,
            font=("Segoe UI", 11),
            cursor="hand2",
        )
        self._drop_label.pack(fill="x", pady=(0, 8))
        self._drop_label.bind("<Button-1>", lambda _: self._browse())

        path_row = ttk.Frame(video_frame)
        path_row.pack(fill="x")
        ttk.Button(path_row, text="Browse…", command=self._browse).pack(side="left")
        self._path_lbl = ttk.Label(path_row, textvariable=self._video_path,
                                   foreground="#4fc3f7", wraplength=500)
        self._path_lbl.pack(side="left", padx=10)

        # ── Options ──────────────────────────────────────────────────
        opts_frame = ttk.LabelFrame(self, text="  Options  ", padding=10)
        opts_frame.pack(fill="x", **pad)

        ttk.Label(opts_frame, text="Speed / accuracy:").grid(row=0, column=0, sticky="w")
        self._skip = tk.IntVar(value=3)
        radio_row = ttk.Frame(opts_frame)
        radio_row.grid(row=0, column=1, sticky="w", padx=10)
        for val, lbl in [(2, "More accurate (slower)"), (3, "Balanced"), (5, "Faster")]:
            ttk.Radiobutton(radio_row, text=lbl, variable=self._skip,
                            value=val).pack(side="left", padx=6)

        self._make_video = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opts_frame,
            text="Also generate annotated video  (shows player IDs — takes much longer)",
            variable=self._make_video,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self._goals_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opts_frame,
            text="Enable goal detection  (unreliable with panning camera — set up zones first)",
            variable=self._goals_enabled,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # ── Buttons ──────────────────────────────────────────────────
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", **pad)

        self._goals_btn = ttk.Button(
            btn_frame, text="1.  Setup Goal Zones",
            command=self._setup_goals, width=22,
        )
        self._goals_btn.pack(side="left", padx=(0, 10))

        self._run_btn = ttk.Button(
            btn_frame, text="2.  Run Analysis",
            command=self._run_analysis, width=18,
        )
        self._run_btn.pack(side="left")

        self._correct_btn = ttk.Button(
            btn_frame, text="3.  Correct Results",
            command=self._correct, width=18,
        )
        self._correct_btn.pack(side="left", padx=(10, 0))

        ttk.Button(
            btn_frame, text="Open Output Folder",
            command=self._open_output,
        ).pack(side="right")

        # ── Progress bar ─────────────────────────────────────────────
        self._progress = ttk.Progressbar(self, mode="indeterminate")
        self._progress.pack(fill="x", padx=12, pady=(0, 4))

        # ── Log output ───────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text="  Output  ", padding=6)
        log_frame.pack(fill="both", expand=True, **pad)

        self._log = scrolledtext.ScrolledText(
            log_frame,
            state="disabled",
            font=("Consolas", 9),
            bg="#0d1117",
            fg="#c9d1d9",
            insertbackground="white",
            wrap="word",
        )
        self._log.pack(fill="both", expand=True)

        # ── Status bar ───────────────────────────────────────────────
        self._status = tk.StringVar(value="Ready — select a video to begin.")
        ttk.Label(self, textvariable=self._status, relief="sunken",
                  anchor="w", padding=(6, 2)).pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_drop(self, event):
        path = event.data.strip().strip("{}")
        self._set_video(path)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select a football video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._set_video(path)

    def _set_video(self, path: str):
        path = path.strip()
        self._video_path.set(path)
        self._drop_label.config(text=os.path.basename(path), fg="#ffffff")
        self._log_write(f"Video: {path}\n")
        self._status.set("Video selected. Run 'Setup Goal Zones' first if you haven't already.")

    def _setup_goals(self):
        path = self._video_path.get()
        if not path:
            self._status.set("Select a video first.")
            return
        cmd = [PYTHON, SETUP_GOALS, path]
        threading.Thread(target=self._stream, args=(cmd, "Goal zone setup"), daemon=True).start()

    def _run_analysis(self):
        path = self._video_path.get()
        if not path:
            self._status.set("Select a video first.")
            return
        if self._running:
            return
        cmd = [PYTHON, ANALYZE, path, "--skip", str(self._skip.get())]
        if not self._make_video.get():
            cmd.append("--no-video")
        if not self._goals_enabled.get():
            cmd.append("--no-goals")
        threading.Thread(target=self._stream, args=(cmd, "Analysis"), daemon=True).start()

    def _correct(self):
        path = self._video_path.get()
        if not path:
            self._status.set("Select a video first.")
            return
        if self._running:
            return
        base = os.path.splitext(os.path.basename(path))[0]
        meta = os.path.join(SCRIPT_DIR, "output", f"{base}_tracks_meta.json")
        if not os.path.exists(meta):
            self._status.set("No tracking log yet — run analysis on this video first.")
            self._log_write("[!] Run 'Run Analysis' before correcting (it creates the tracking log).\n")
            return
        cmd = [PYTHON, CORRECT, path]
        threading.Thread(target=self._stream, args=(cmd, "Correction tool"), daemon=True).start()

    def _stream(self, cmd: list, label: str):
        self._running = True
        self.after(0, lambda: self._run_btn.config(state="disabled"))
        self.after(0, lambda: self._goals_btn.config(state="disabled"))
        self.after(0, lambda: self._correct_btn.config(state="disabled"))
        self.after(0, self._progress.start)
        self._status.set(f"{label} running…")
        self._log_write(f"\n{'─'*60}\n")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=SCRIPT_DIR,
            )
            for line in proc.stdout:
                self._log_write(line)
            proc.wait()
            done_msg = f"{label} complete." if proc.returncode == 0 else \
                       f"{label} finished with errors (exit {proc.returncode})."
        except Exception as exc:
            done_msg = f"Error launching process: {exc}"

        self._running = False
        self._status.set(done_msg)
        self._log_write(f"\n{done_msg}\n")
        self.after(0, self._progress.stop)
        self.after(0, lambda: self._run_btn.config(state="normal"))
        self.after(0, lambda: self._goals_btn.config(state="normal"))
        self.after(0, lambda: self._correct_btn.config(state="normal"))

    def _open_output(self):
        out = os.path.join(SCRIPT_DIR, "output")
        os.makedirs(out, exist_ok=True)
        os.startfile(out)

    def _log_write(self, text: str):
        def _do():
            self._log.config(state="normal")
            self._log.insert("end", text)
            self._log.see("end")
            self._log.config(state="disabled")
        self.after(0, _do)


if __name__ == "__main__":
    app = App()
    app.mainloop()
