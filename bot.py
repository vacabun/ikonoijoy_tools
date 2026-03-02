import cv2
import numpy as np
import subprocess
import time
import os
import threading
from datetime import datetime

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from detector import CircleDetector


class ADBGameBot:
    """ADB automation bot for matching puzzle games."""

    def __init__(self, config: dict):
        self.config = config
        device_cfg = config.get('device', {})
        bot_cfg = config.get('bot', {})

        self.device_id = device_cfg.get('id')
        self.running = False
        self.debug_mode = bot_cfg.get('debug_mode', True)
        self.save_images = bot_cfg.get('save_images', True)
        self.show_preview = bot_cfg.get('show_preview', True)
        self.interval = bot_cfg.get('interval', 10)
        self.swipe_duration = bot_cfg.get('swipe_duration', 200)
        self.output_dir = bot_cfg.get('output_dir', 'game_bot_output')

        os.makedirs(self.output_dir, exist_ok=True)

        self.detector = None
        self.bot_thread = None
        self.preview_label = None
        self.status_text = None

    def _adb_cmd(self):
        """Return base adb command with optional device selector."""
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        return cmd

    def connect_device(self, device_id=None):
        """Connect to an ADB device, auto-selecting if no ID is given."""
        try:
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            if device_id:
                self.device_id = device_id
            else:
                lines = result.stdout.strip().split('\n')[1:]
                if lines and lines[0].strip():
                    self.device_id = lines[0].split()[0]
                else:
                    raise Exception("No ADB device found")
            print(f"+ Connected to device: {self.device_id}")
            return True
        except Exception as e:
            print(f"- Failed to connect device: {e}")
            return False

    def capture_screen(self):
        """Capture device screen, pull to local, and return the image."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_path = f"/sdcard/screenshot_{timestamp}.png"
            local_path = os.path.join(self.output_dir, f"screen_{timestamp}.png")

            base = self._adb_cmd()
            subprocess.run(base + ['shell', 'screencap', '-p', temp_path], check=True)
            subprocess.run(base + ['pull', temp_path, local_path], check=True, capture_output=True)
            subprocess.run(base + ['shell', 'rm', temp_path], check=True)

            image = cv2.imread(local_path)
            if not self.save_images:
                os.remove(local_path)

            if self.debug_mode:
                print(f"+ Screenshot captured: {image.shape}")
            return image, timestamp
        except Exception as e:
            print(f"- Screenshot failed: {e}")
            return None, None

    def swipe_path(self, path_coords):
        """Simulate a continuous swipe gesture through the given coordinates."""
        if len(path_coords) < 2:
            return False

        duration = self.swipe_duration
        try:
            base = self._adb_cmd() + ['shell']
            total_points = len(path_coords)
            time_per_point = duration / total_points / 1000.0

            start = path_coords[0]
            subprocess.run(
                base + ['input', 'motionevent', 'DOWN', str(start[0]), str(start[1])],
                check=False, capture_output=True
            )

            if self.debug_mode:
                print(f"+ Swipe start: {total_points} points, duration {duration}ms")

            for i, (x, y) in enumerate(path_coords[1:], 1):
                subprocess.run(
                    base + ['input', 'motionevent', 'MOVE', str(x), str(y)],
                    check=False, capture_output=True
                )
                time.sleep(time_per_point)
                if self.debug_mode and i % 2 == 1:
                    print(f"  -> point {i}/{total_points}: ({x}, {y})")

            end = path_coords[-1]
            subprocess.run(
                base + ['input', 'motionevent', 'UP', str(end[0]), str(end[1])],
                check=False, capture_output=True
            )

            if self.debug_mode:
                print(f"+ Swipe complete: {start} -> {end}")
            return True

        except Exception as e:
            if self.debug_mode:
                print(f"motionevent failed, using fallback: {e}")
            try:
                base = self._adb_cmd()
                start, end = path_coords[0], path_coords[-1]
                subprocess.run(
                    base + ['shell', 'input', 'swipe',
                             str(start[0]), str(start[1]),
                             str(end[0]), str(end[1]), str(duration)],
                    check=True
                )
                if self.debug_mode:
                    print(f"+ Fallback swipe complete: {start} -> {end}")
                return True
            except Exception as e2:
                print(f"- Swipe failed: {e2}")
                return False

    def find_bottom_paths(self, paths):
        """Filter paths by minimum length and sort bottom-first by average Y."""
        min_length = self.config['detection']['min_path_length']
        valid_paths = []
        for path in paths:
            if len(path['nodes']) >= min_length:
                avg_y = np.mean([
                    self.detector.color_data[n]['center'][1]
                    for n in path['nodes']
                ])
                path['avg_y'] = avg_y
                valid_paths.append(path)

        valid_paths.sort(key=lambda p: p['avg_y'], reverse=True)

        if self.debug_mode:
            print(f"\nFound {len(valid_paths)} valid path(s) (length >= {min_length}):")
            for i, p in enumerate(valid_paths[:3]):
                print(f"  Path {i}: length={len(p['nodes'])}, avg_y={p['avg_y']:.1f}")

        return valid_paths

    def process_frame(self, image, timestamp):
        """Detect circles, classify colors, find and highlight best paths."""
        temp_img_path = os.path.join(self.output_dir, f"temp_{timestamp}.png")
        cv2.imwrite(temp_img_path, image)

        self.detector = CircleDetector(temp_img_path, self.config)
        circles = self.detector.detect_circles()
        if circles is None or len(circles) == 0:
            print("No circles detected")
            if not self.save_images:
                os.remove(temp_img_path)
            return None

        self.detector.extract_ball_colors()
        locked = sum(1 for c in self.detector.color_data if c.get('locked', False))
        if self.debug_mode:
            unlocked = len(self.detector.color_data) - locked
            print(f"Locked: {locked}, Unlocked: {unlocked}")

        self.detector.classify_colors()
        paths = self.detector.find_optimal_paths()
        if not paths:
            print("No connectable paths found")
            if not self.save_images:
                os.remove(temp_img_path)
            return None

        bottom_paths = self.find_bottom_paths(paths)
        result_img = self.detector.draw_connections(paths)

        if bottom_paths:
            selected = bottom_paths[0]
            nodes = selected['nodes']
            for i in range(len(nodes) - 1):
                pos1 = self.detector.color_data[nodes[i]]['center']
                pos2 = self.detector.color_data[nodes[i + 1]]['center']
                cv2.line(result_img, pos1, pos2, (0, 255, 255), 8)
            if nodes:
                start_pos = self.detector.color_data[nodes[0]]['center']
                cv2.putText(result_img, "SELECTED",
                            (start_pos[0] - 50, start_pos[1] - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        if self.save_images:
            result_path = os.path.join(self.output_dir, f"result_{timestamp}.png")
            cv2.imwrite(result_path, result_img)
            print(f"+ Result saved: {result_path}")

        if self.show_preview and self.preview_label:
            self.update_preview(result_img)

        if not self.save_images:
            os.remove(temp_img_path)

        return bottom_paths

    def execute_best_path(self, paths):
        """Execute the highest-priority (bottom-most) path via swipe."""
        if not paths:
            return False
        best_path = paths[0]
        nodes = best_path['nodes']
        coords = [self.detector.color_data[n]['center'] for n in nodes]
        if self.debug_mode:
            print(f"\nExecuting path: color={best_path['color_type']}, "
                  f"length={len(nodes)}, coords={coords}")
        return self.swipe_path(coords)

    def run_once(self):
        """Run a single screenshot → detection → action cycle."""
        print(f"\n{'=' * 60}")
        print(f"Detection cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")

        image, timestamp = self.capture_screen()
        if image is None:
            return False

        paths = self.process_frame(image, timestamp)
        if paths:
            self.execute_best_path(paths)
            self.update_status(f"+ Success - connected {len(paths[0]['nodes'])} ball(s)")
        else:
            self.update_status("! No valid paths found")
        return True

    def start_bot(self):
        """Start the automation loop in a background daemon thread."""
        self.running = True

        def bot_loop():
            while self.running:
                try:
                    self.run_once()
                    for i in range(self.interval):
                        if not self.running:
                            break
                        time.sleep(1)
                        if self.debug_mode and i % 5 == 0:
                            print(f"Waiting... {self.interval - i}s remaining")
                except Exception as e:
                    print(f"- Error: {e}")
                    self.update_status(f"- Error: {e}")
                    time.sleep(5)

        self.bot_thread = threading.Thread(target=bot_loop, daemon=True)
        self.bot_thread.start()
        print("+ Bot started")

    def stop_bot(self):
        """Stop the automation loop."""
        self.running = False
        print("+ Bot stopped")

    def update_preview(self, image):
        """Resize and display an image in the GUI preview label."""
        try:
            gui_cfg = self.config.get('gui', {})
            max_w = gui_cfg.get('preview_max_width', 850)
            max_h = gui_cfg.get('preview_max_height', 450)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image_resized = cv2.resize(image_rgb, (new_w, new_h))
            pil_image = Image.fromarray(image_resized)
            photo = ImageTk.PhotoImage(pil_image)
            self.preview_label.config(image=photo)
            self.preview_label.image = photo  # prevent GC
        except Exception as e:
            print(f"Preview update failed: {e}")

    def update_status(self, message):
        """Write a timestamped message to the GUI status box."""
        if self.status_text:
            self.status_text.delete("1.0", "end")
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.status_text.insert("1.0", f"[{timestamp}] {message}")

    def create_gui(self):
        """Build and launch the Tkinter control panel."""
        gui_cfg = self.config.get('gui', {})
        root = tk.Tk()
        root.title(gui_cfg.get('title', 'ikonijoy tools'))
        root.geometry(gui_cfg.get('geometry', '900x700'))

        # --- Controls ---
        control_frame = ttk.LabelFrame(root, text="Controls", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(control_frame, text="Connect Device",
                   command=self.connect_device).grid(row=0, column=0, padx=5)
        self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_bot)
        self.start_btn.grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Stop",
                   command=self.stop_bot).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Run Once",
                   command=self.run_once).grid(row=0, column=3, padx=5)

        # --- Settings ---
        settings_frame = ttk.LabelFrame(root, text="Settings", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)

        self.debug_var = tk.BooleanVar(value=self.debug_mode)
        ttk.Checkbutton(settings_frame, text="Debug",
                        variable=self.debug_var,
                        command=lambda: setattr(self, 'debug_mode', self.debug_var.get())
                        ).grid(row=0, column=0, padx=10, sticky="w")

        self.save_var = tk.BooleanVar(value=self.save_images)
        ttk.Checkbutton(settings_frame, text="Save Images",
                        variable=self.save_var,
                        command=lambda: setattr(self, 'save_images', self.save_var.get())
                        ).grid(row=0, column=1, padx=10, sticky="w")

        self.preview_var = tk.BooleanVar(value=self.show_preview)
        ttk.Checkbutton(settings_frame, text="Show Preview",
                        variable=self.preview_var,
                        command=lambda: setattr(self, 'show_preview', self.preview_var.get())
                        ).grid(row=0, column=2, padx=10, sticky="w")

        interval_min = gui_cfg.get('interval_min', 5)
        interval_max = gui_cfg.get('interval_max', 60)
        ttk.Label(settings_frame, text="Interval (s):").grid(row=1, column=0, padx=10, sticky="w")
        self.interval_var = tk.IntVar(value=self.interval)
        interval_spin = ttk.Spinbox(settings_frame, from_=interval_min, to=interval_max,
                                    textvariable=self.interval_var, width=10)
        interval_spin.grid(row=1, column=1, sticky="w")
        interval_spin.bind('<Return>', lambda e: setattr(self, 'interval', self.interval_var.get()))

        # --- Status ---
        status_frame = ttk.LabelFrame(root, text="Status", padding=10)
        status_frame.pack(fill="x", padx=10, pady=5)
        self.status_text = tk.Text(status_frame, height=3, wrap="word")
        self.status_text.pack(fill="x")
        self.status_text.insert("1.0", "Ready")

        # --- Preview ---
        preview_frame = ttk.LabelFrame(root, text="Preview", padding=10)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(fill="both", expand=True)

        root.mainloop()
