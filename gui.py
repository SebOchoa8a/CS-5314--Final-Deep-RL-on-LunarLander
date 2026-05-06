"""
gui.py — Interactive training GUI for Deep RL on LunarLander-v3.

Lets you pick a variant (vanilla / double / per / double+per),
toggle ICM and reward shaping, tune hyperparameters, watch live
training charts, view episode logs, and optionally preview the
simulation (requires Pillow).

Run with:
    python gui.py
"""

import multiprocessing as mp
import queue
import sys
import threading
from collections import deque

import numpy as np
import torch
import gymnasium as gym

import tkinter as tk
from tkinter import ttk, scrolledtext

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from agent import DQNAgent
from replay import UniformReplayBuffer, PrioritizedReplayBuffer
from icm import ICM
from reward_shaping import PotentialShaping

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False

MACOS = sys.platform == "darwin"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANTS = {
    "vanilla":    (False, False),
    "double":     (True,  False),
    "per":        (False, True),
    "double_per": (True,  True),
}
VARIANT_LABELS = {
    "vanilla":    "Vanilla DQN",
    "double":     "Double DQN",
    "per":        "PER only (ablation)",
    "double_per": "Double DQN + PER",
}

PLOT_EVERY = 5   # redraw charts every N episodes

# Chart colors
C_RAW    = "#aec6e8"
C_AVG    = "#e8954c"
C_SOLVED = "#5cb85c"
C_LOSS   = "#9b59b6"
C_EPS    = "#e74c3c"
C_ICM    = "#1abc9c"
C_SHAPE  = "#f39c12"


# ---------------------------------------------------------------------------
# Training Worker  (background thread)
# ---------------------------------------------------------------------------

class TrainingWorker:
    """
    Runs the full DQN training loop (+ optional ICM / reward shaping) in a
    daemon thread and pushes one dict per episode into `data_q`.

    Queue item keys
    ---------------
    episode, reward, moving_avg, epsilon, loss,
    icm_loss (float|None), shaping (float|None),
    frame (ndarray|None), solved (bool), done (bool), error (str|None)
    """

    def __init__(self, cfg: dict, data_q, stop_event):
        self.cfg    = cfg
        self.data_q = data_q
        self._stop  = stop_event   # mp.Event — visible across process boundary

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        use_double, use_per = VARIANTS[cfg["variant"]]

        # Resolve the render mode requested by the GUI.
        # "human"     → pygame window opened by gymnasium (blocking render per step)
        # "rgb_array" → we capture frames and show them in the in-app viewer
        # "none"      → no rendering at all
        rm = cfg["render_mode"]
        if rm == "rgb_array" and not PIL_OK:
            rm = "none"          # fall back silently if Pillow is missing
        actual_render_mode = None if rm == "none" else rm

        try:
            env = gym.make("LunarLander-v3", render_mode=actual_render_mode)
        except Exception as exc:
            self.data_q.put({"error": str(exc), "done": True})
            return

        state_dim  = env.observation_space.shape[0]
        action_dim = env.action_space.n

        np.random.seed(0)
        torch.manual_seed(0)
        env.reset(seed=0)

        agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim,
            hidden=cfg["hidden"], lr=cfg["lr"],
            gamma=cfg["gamma"],  tau=cfg["tau"],
            double=use_double,
        )

        if use_per:
            buffer = PrioritizedReplayBuffer(
                capacity=cfg["buffer_size"], state_dim=state_dim,
                alpha=0.6, beta_start=0.4, beta_frames=100_000,
            )
        else:
            buffer = UniformReplayBuffer(
                capacity=cfg["buffer_size"], state_dim=state_dim,
            )

        icm    = ICM(state_dim=state_dim, action_dim=action_dim) \
                     if cfg["use_icm"]     else None
        shaper = PotentialShaping(gamma=cfg["gamma"]) \
                     if cfg["use_shaping"] else None

        epsilon      = cfg["eps_start"]
        eps_step     = (cfg["eps_start"] - cfg["eps_end"]) / max(cfg["eps_decay"], 1)
        reward_win   = deque(maxlen=100)
        total_steps  = 0

        for ep in range(1, cfg["episodes"] + 1):
            if self._stop.is_set():
                break

            state, _ = env.reset()
            ep_ext   = 0.0
            ep_loss  = 0.0
            ep_icm   = 0.0
            ep_shape = 0.0
            n_learn  = 0
            ep_steps = 0
            frame    = None
            done     = False

            while not done:
                if self._stop.is_set():
                    break

                action = agent.act(state, epsilon)
                nxt, ext_r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                r = ext_r
                if icm:
                    r += icm.intrinsic_reward(state, action, nxt)
                if shaper:
                    bonus = shaper.shaping_bonus(state, nxt)
                    r        += bonus
                    ep_shape += bonus

                buffer.add(state, action, r, nxt, terminated)
                state    = nxt
                ep_ext  += ext_r
                ep_steps += 1
                total_steps += 1

                if (len(buffer) >= cfg["batch_size"]
                        and total_steps % cfg["learn_every"] == 0):
                    batch = buffer.sample(cfg["batch_size"])
                    td_errors, loss = agent.learn(batch)
                    ep_loss += loss
                    n_learn += 1
                    if use_per:
                        buffer.update_priorities(batch[6], td_errors)
                    if icm:
                        ep_icm = icm.update(batch[0], batch[1], batch[3])

                # Only capture frames for the in-app viewer.
                # "human" mode renders automatically inside env.step().
                if actual_render_mode == "rgb_array":
                    frame = env.render()

            # ε decay (episode-level, same convention as train.py)
            epsilon = max(cfg["eps_end"], epsilon - eps_step)

            reward_win.append(ep_ext)
            avg    = float(np.mean(reward_win))
            solved = (len(reward_win) == 100 and avg >= 200.0)

            self.data_q.put({
                "episode":    ep,
                "reward":     ep_ext,
                "moving_avg": avg,
                "epsilon":    epsilon,
                "loss":       ep_loss / max(n_learn, 1),
                "icm_loss":   ep_icm                         if cfg["use_icm"]     else None,
                "shaping":    ep_shape / max(ep_steps, 1)    if cfg["use_shaping"] else None,
                "frame":      frame,
                "solved":     solved,
                "done":       False,
                "error":      None,
            })

            if solved:
                break

        env.close()
        self.data_q.put({"done": True, "error": None})


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("LunarLander — Deep RL Trainer")
        root.minsize(960, 650)

        # --- data stores (one entry per episode) ---
        self.eps_list   : list[int]   = []
        self.rewards    : list[float] = []
        self.avgs       : list[float] = []
        self.losses     : list[float] = []
        self.epsilons   : list[float] = []
        self.icm_losses : list[float] = []
        self.shapings   : list[float] = []

        self._worker         : TrainingWorker | None = None
        self._proc           : mp.Process | None     = None
        self._data_q         : mp.Queue              = mp.Queue()
        self._after_id                               = None
        self._training_done  : bool                  = True
        self._sim_photo                              = None   # keep ref alive

        self._build_ui()

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self):
        pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        left  = ttk.Frame(pw, width=240, padding=(10, 8))
        right = ttk.Frame(pw, padding=(4, 4))
        pw.add(left,  weight=0)
        pw.add(right, weight=1)

        self._build_controls(left)
        self._build_workspace(right)
        self._build_statusbar(self.root)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_controls(self, f):
        # Title
        ttk.Label(f, text="Deep RL — LunarLander-v3",
                  font=("", 11, "bold")).pack(anchor="w", pady=(0, 8))

        # VARIANT
        self._section(f, "VARIANT")
        self.variant_var = tk.StringVar(value="vanilla")
        for key, lbl in VARIANT_LABELS.items():
            ttk.Radiobutton(f, text=lbl, value=key,
                            variable=self.variant_var).pack(anchor="w", pady=1)

        self._sep(f)

        # ADD-ONS
        self._section(f, "ADD-ONS")
        self.icm_var     = tk.BooleanVar(value=False)
        self.shaping_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Intrinsic Curiosity Module (ICM)",
                        variable=self.icm_var).pack(anchor="w")
        ttk.Checkbutton(f, text="Reward Shaping  (Ng et al. 1999)",
                        variable=self.shaping_var).pack(anchor="w")

        self._sep(f)

        # HYPERPARAMETERS
        self._section(f, "HYPERPARAMETERS")
        self._hp: dict[str, ttk.Entry] = {}
        params = [
            ("Episodes",    "episodes",    "1000"),
            ("Learn rate",  "lr",          "0.0005"),
            ("Gamma  (γ)",  "gamma",       "0.99"),
            ("Tau    (τ)",  "tau",         "0.001"),
            ("Batch size",  "batch_size",  "64"),
            ("Buffer size", "buffer_size", "100000"),
            ("ε  start",    "eps_start",   "1.0"),
            ("ε  end",      "eps_end",     "0.01"),
            ("ε  decay ep", "eps_decay",   "500"),
            ("Hidden units","hidden",      "64"),
            ("Learn every", "learn_every", "4"),
        ]
        for label, key, default in params:
            row = ttk.Frame(f)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=label, width=13, anchor="w").pack(side="left")
            ent = ttk.Entry(row, width=9)
            ent.insert(0, default)
            ent.pack(side="right")
            self._hp[key] = ent

        self._sep(f)

        # OPTIONS — render mode
        self._section(f, "RENDER MODE")
        self.render_mode_var = tk.StringVar(value="none")

        ttk.Radiobutton(f, text="None  (fastest)",
                        value="none",
                        variable=self.render_mode_var).pack(anchor="w", pady=1)

        rb_inapp = ttk.Radiobutton(f, text="In-app viewer  (rgb_array)",
                                   value="rgb_array",
                                   variable=self.render_mode_var)
        rb_inapp.pack(anchor="w", pady=1)
        if not PIL_OK:
            rb_inapp.configure(state="disabled")
            ttk.Label(f, text="  ↳ needs: pip install Pillow",
                      font=("", 8), foreground="#888").pack(anchor="w")

        ttk.Radiobutton(f, text="External window  (human)",
                        value="human",
                        variable=self.render_mode_var).pack(anchor="w", pady=1)

        self.no_stop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Disable early stop at 200",
                        variable=self.no_stop_var).pack(anchor="w", pady=(4, 0))

        self._sep(f)

        # BUTTONS
        self.btn_start = ttk.Button(f, text="▶  Start", command=self._start)
        self.btn_stop  = ttk.Button(f, text="■  Stop",  command=self._stop,
                                    state="disabled")
        self.btn_reset = ttk.Button(f, text="↺  Reset", command=self._reset)
        for btn in (self.btn_start, self.btn_stop, self.btn_reset):
            btn.pack(fill="x", pady=2)

    def _section(self, parent, text):
        ttk.Label(parent, text=text, font=("", 8, "bold"),
                  foreground="#555").pack(anchor="w", pady=(4, 2))

    def _sep(self, parent):
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_workspace(self, f):
        nb = ttk.Notebook(f)
        nb.pack(fill=tk.BOTH, expand=True)

        # --- Charts tab ---
        charts_tab = ttk.Frame(nb)
        nb.add(charts_tab, text="  Charts  ")
        self._build_charts(charts_tab)

        # --- Log tab ---
        log_tab = ttk.Frame(nb)
        nb.add(log_tab, text="  Episode Log  ")
        self.log_box = scrolledtext.ScrolledText(
            log_tab, state="disabled", font=("Courier", 9),
            wrap="none", height=10)
        self.log_box.pack(fill=tk.BOTH, expand=True)

        # --- Simulation tab ---
        sim_tab = ttk.Frame(nb)
        nb.add(sim_tab, text="  Simulation  ")
        self._build_sim_tab(sim_tab)

        self._nb = nb

    def _build_charts(self, f):
        """Four stacked matplotlib subplots embedded in tkinter."""
        self._fig = Figure(figsize=(8, 9), tight_layout=True)

        self._ax_r   = self._fig.add_subplot(411)   # Reward
        self._ax_l   = self._fig.add_subplot(412)   # Loss
        self._ax_e   = self._fig.add_subplot(413)   # Epsilon
        self._ax_x   = self._fig.add_subplot(414)   # ICM / Shaping / idle

        self._style_axes()

        self._canvas = FigureCanvasTkAgg(self._fig, master=f)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_empty_charts()
        self._canvas.draw()

    def _style_axes(self):
        titles = [
            (self._ax_r, "Episode Reward",         "Reward"),
            (self._ax_l, "Training Loss",           "Loss"),
            (self._ax_e, "Exploration  (ε)",        "ε"),
            (self._ax_x, "ICM Loss / Shaping Bonus","Value"),
        ]
        for ax, title, ylabel in titles:
            ax.set_facecolor("#f5f5f5")
            ax.grid(True, alpha=0.35, linewidth=0.7)
            ax.set_title(title, fontsize=9, pad=3)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
        self._ax_x.set_xlabel("Episode", fontsize=8)

    def _draw_empty_charts(self):
        for ax in (self._ax_r, self._ax_l, self._ax_e, self._ax_x):
            ax.clear()
        self._style_axes()
        self._ax_x.text(0.5, 0.5, "Enable ICM or Reward Shaping\nto see add-on metrics",
                        ha="center", va="center", transform=self._ax_x.transAxes,
                        fontsize=9, color="#999")

    def _build_sim_tab(self, f):
        if PIL_OK:
            self._sim_canvas = tk.Canvas(f, bg="#111111")
            self._sim_canvas.pack(fill=tk.BOTH, expand=True)
            self._sim_img_id = None
            # Overlay label shown when human-mode is active
            self._sim_overlay = ttk.Label(
                f,
                text="Simulation is running in an external pygame window.\n"
                     "Switch to  \"In-app viewer\"  to see frames here.",
                font=("", 11), anchor="center", justify="center",
                foreground="#888", background="#111111")
        else:
            self._sim_canvas = None
            self._sim_overlay = None
            ttk.Label(f,
                      text="Simulation viewer requires Pillow.\n\npip install Pillow",
                      font=("", 11), anchor="center", justify="center",
                      foreground="#888").pack(expand=True)

    def _build_statusbar(self, root):
        bar = ttk.Frame(root, relief="sunken", padding=(6, 2))
        bar.pack(fill="x", side="bottom")
        self._status = tk.StringVar(value="Ready  —  choose a variant and press ▶ Start")
        ttk.Label(bar, textvariable=self._status,
                  font=("Courier", 9), anchor="w").pack(fill="x")

    # =========================================================================
    # Button actions
    # =========================================================================

    def _read_config(self) -> dict:
        def _f(k): return float(self._hp[k].get())
        def _i(k): return int(float(self._hp[k].get()))
        return {
            "variant":      self.variant_var.get(),
            "use_icm":      self.icm_var.get(),
            "use_shaping":  self.shaping_var.get(),
            "render_mode":  self.render_mode_var.get(),
            "no_early_stop":self.no_stop_var.get(),
            "episodes":     _i("episodes"),
            "lr":           _f("lr"),
            "gamma":        _f("gamma"),
            "tau":          _f("tau"),
            "batch_size":   _i("batch_size"),
            "buffer_size":  _i("buffer_size"),
            "eps_start":    _f("eps_start"),
            "eps_end":      _f("eps_end"),
            "eps_decay":    _i("eps_decay"),
            "hidden":       _i("hidden"),
            "learn_every":  _i("learn_every"),
        }

    def _start(self):
        try:
            cfg = self._read_config()
        except ValueError as exc:
            self._status.set(f"Config error: {exc}")
            return

        self.btn_start.configure(state="disabled")
        self.btn_stop .configure(state="normal")
        self._training_done = False

        # clear data
        for lst in (self.eps_list, self.rewards, self.avgs,
                    self.losses, self.epsilons, self.icm_losses, self.shapings):
            lst.clear()

        addons = []
        if cfg["use_icm"]:     addons.append("ICM")
        if cfg["use_shaping"]: addons.append("Shaping")
        addon_str = " + ".join(addons) if addons else "none"
        self._log(f"{'='*60}\n"
                  f"Variant : {VARIANT_LABELS[cfg['variant']]}\n"
                  f"Add-ons : {addon_str}\n"
                  f"Episodes: {cfg['episodes']}  |  LR: {cfg['lr']}"
                  f"  |  γ: {cfg['gamma']}\n"
                  f"{'='*60}\n")

        self._draw_empty_charts()
        self._canvas.draw_idle()

        # Update simulation tab to reflect chosen render mode
        if self._sim_canvas and self._sim_overlay:
            if cfg["render_mode"] == "human":
                self._sim_canvas.place_forget()
                self._sim_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
            else:
                self._sim_overlay.place_forget()
                self._sim_canvas.pack(fill=tk.BOTH, expand=True)
                self._sim_img_id = None   # reset so first frame creates a new image item

        self._data_q = mp.Queue()
        stop_ev      = mp.Event()
        self._worker = TrainingWorker(cfg, self._data_q, stop_ev)
        self._proc   = mp.Process(target=self._worker.run, daemon=True)
        self._proc.start()
        self._poll()

    def _stop(self):
        if self._worker:
            self._worker.stop()
        self.btn_stop.configure(state="disabled")
        self._status.set("Stopping…")

    def _reset(self):
        self._stop()
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self._training_done = True

        for lst in (self.eps_list, self.rewards, self.avgs,
                    self.losses, self.epsilons, self.icm_losses, self.shapings):
            lst.clear()

        self._draw_empty_charts()
        self._canvas.draw_idle()

        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        self._status.set("Reset  —  ready")
        self.btn_start.configure(state="normal")
        self.btn_stop .configure(state="disabled")

    # =========================================================================
    # Queue polling
    # =========================================================================

    def _poll(self):
        """Drain the data queue on the main thread; reschedule if still running."""
        try:
            while True:
                self._handle(self._data_q.get_nowait())
        except queue.Empty:
            pass

        if not self._training_done:
            self._after_id = self.root.after(150, self._poll)

    def _handle(self, item: dict):
        if item.get("error"):
            self._status.set(f"Error: {item['error']}")
            self._log(f"\nERROR: {item['error']}\n")
            self._finish()
            return

        if item.get("done"):
            self._finish()
            return

        ep  = item["episode"]
        r   = item["reward"]
        avg = item["moving_avg"]
        eps = item["epsilon"]
        lss = item["loss"]
        icm = item["icm_loss"]
        shp = item["shaping"]

        # Append to data stores
        self.eps_list.append(ep)
        self.rewards .append(r)
        self.avgs    .append(avg)
        self.epsilons.append(eps)
        self.losses  .append(lss)
        if icm is not None: self.icm_losses.append(icm)
        if shp is not None: self.shapings  .append(shp)

        # Status bar
        parts = [f"Ep {ep:4d}", f"R {r:8.1f}", f"Avg100 {avg:7.1f}",
                 f"ε {eps:.3f}", f"Loss {lss:.4f}"]
        if icm is not None: parts.append(f"ICM {icm:.4f}")
        if shp is not None: parts.append(f"Shape {shp:+.3f}")
        self._status.set("  │  ".join(parts))

        # Log every 10 episodes or on solve
        if ep % 10 == 0 or item.get("solved"):
            self._log("  │  ".join(parts) + "\n")

        # Charts every PLOT_EVERY episodes
        if ep % PLOT_EVERY == 0:
            self._update_charts()

        # Simulation frame
        if item.get("frame") is not None:
            self._show_frame(item["frame"])

        # Solved banner
        if item.get("solved"):
            msg = f"\n✓ SOLVED at episode {ep}  (avg100 = {avg:.1f})\n\n"
            self._log(msg)
            self._status.set(f"SOLVED at episode {ep}  —  avg100 = {avg:.1f}")

    def _finish(self):
        self._training_done = True
        self._update_charts()
        self.btn_start.configure(state="normal")
        self.btn_stop .configure(state="disabled")
        self._log("\n— Training finished —\n")

    # =========================================================================
    # Chart updates
    # =========================================================================

    def _update_charts(self):
        if not self.eps_list:
            return
        x = self.eps_list

        # ── Reward ────────────────────────────────────────────────────────
        ax = self._ax_r
        ax.clear()
        ax.set_facecolor("#f5f5f5")
        ax.grid(True, alpha=0.35, linewidth=0.7)
        ax.plot(x, self.rewards, color=C_RAW,  alpha=0.35, linewidth=0.7)
        ax.plot(x, self.avgs,    color=C_AVG,  linewidth=1.8, label="100-ep avg")
        ax.axhline(200, color=C_SOLVED, linestyle="--", alpha=0.75,
                   linewidth=1.2, label="Solved  (200)")
        ax.set_title("Episode Reward", fontsize=9, pad=3)
        ax.set_ylabel("Reward", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.7)

        # ── Loss ─────────────────────────────────────────────────────────
        ax = self._ax_l
        ax.clear()
        ax.set_facecolor("#f5f5f5")
        ax.grid(True, alpha=0.35, linewidth=0.7)
        ax.plot(x, self.losses, color=C_LOSS, linewidth=1.2, label="DQN loss")
        ax.set_title("Training Loss", fontsize=9, pad=3)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

        # ── Epsilon ───────────────────────────────────────────────────────
        ax = self._ax_e
        ax.clear()
        ax.set_facecolor("#f5f5f5")
        ax.grid(True, alpha=0.35, linewidth=0.7)
        ax.plot(x, self.epsilons, color=C_EPS, linewidth=1.5, label="ε")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Exploration  (ε)", fontsize=9, pad=3)
        ax.set_ylabel("ε", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

        # ── ICM / Shaping ────────────────────────────────────────────────
        ax = self._ax_x
        ax.clear()
        ax.set_facecolor("#f5f5f5")
        ax.grid(True, alpha=0.35, linewidth=0.7)
        ax.set_xlabel("Episode", fontsize=8)
        ax.tick_params(labelsize=7)

        plotted = False
        if self.icm_losses and len(self.icm_losses) == len(x):
            ax.plot(x, self.icm_losses, color=C_ICM, linewidth=1.2,
                    label="ICM loss")
            plotted = True
        if self.shapings and len(self.shapings) == len(x):
            ax2 = ax.twinx()
            ax2.plot(x, self.shapings, color=C_SHAPE, linewidth=1.2,
                     linestyle="--", label="Avg shaping bonus")
            ax2.set_ylabel("Shaping bonus", fontsize=7, color=C_SHAPE)
            ax2.tick_params(axis="y", labelsize=7, labelcolor=C_SHAPE)
            plotted = True
        if plotted:
            ax.set_title("ICM Loss / Shaping Bonus", fontsize=9, pad=3)
            ax.set_ylabel("ICM loss", fontsize=8)
            lines1, labs1 = ax.get_legend_handles_labels()
            ax.legend(lines1, labs1, loc="upper right", fontsize=7, framealpha=0.7)
        else:
            ax.set_title("ICM Loss / Shaping Bonus", fontsize=9, pad=3)
            ax.set_ylabel("Value", fontsize=8)
            ax.text(0.5, 0.5,
                    "Enable ICM or Reward Shaping\nto see add-on metrics here",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#aaa")

        self._canvas.draw_idle()

    # =========================================================================
    # Simulation viewer
    # =========================================================================

    def _show_frame(self, frame: np.ndarray):
        if not PIL_OK or self._sim_canvas is None:
            return
        try:
            img = Image.fromarray(frame)
            cw  = max(self._sim_canvas.winfo_width(),  600)
            ch  = max(self._sim_canvas.winfo_height(), 400)
            # Fit inside canvas while keeping aspect ratio
            scale = min(cw / img.width, ch / img.height)
            nw, nh = int(img.width * scale), int(img.height * scale)
            img = img.resize((nw, nh), Image.LANCZOS)
            self._sim_photo = ImageTk.PhotoImage(img)   # keep ref
            if self._sim_img_id:
                self._sim_canvas.itemconfig(self._sim_img_id,
                                            image=self._sim_photo)
            else:
                self._sim_img_id = self._sim_canvas.create_image(
                    cw // 2, ch // 2, anchor="center",
                    image=self._sim_photo)
        except Exception:
            pass   # never crash the UI over a display error

    # =========================================================================
    # Episode log
    # =========================================================================

    def _log(self, text: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mp.freeze_support()   # no-op on macOS/Linux; needed if ever packaged on Windows
    root = tk.Tk()
    root.geometry("1150x780")
    try:
        # Use a modern ttk theme where available
        style = ttk.Style(root)
        for theme in ("clam", "alt", "default"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
