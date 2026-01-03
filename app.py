import tkinter as tk
from tkinter import messagebox, ttk
import os
import sys
import subprocess
import threading
import time
from pathlib import Path
from config import FACE_DATA_DIR, TRAINED_DATA_DIR, logger

class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Empirical Analysis of Face Recognition - PhD Research Suite")
        self.root.geometry("1000x750")
        self.root.configure(bg="#1a1a1a")
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Color Palette
        self.bg_color = "#1a1a1a"
        self.card_color = "#2d2d2d"
        self.accent_color = "#3498db"
        self.text_color = "#ecf0f1"
        self.success_color = "#2ecc71"

        # Configure styles
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("Card.TFrame", background=self.card_color, relief="flat")
        self.style.configure("TLabel", background=self.bg_color, foreground=self.text_color, font=("Segoe UI", 11))
        self.style.configure("Card.TLabel", background=self.card_color, foreground=self.text_color, font=("Segoe UI", 11))
        self.style.configure("Header.TLabel", background=self.bg_color, foreground=self.accent_color, font=("Segoe UI", 24, "bold"))
        self.style.configure("SubHeader.TLabel", background=self.bg_color, foreground=self.text_color, font=("Segoe UI", 12, "italic"))
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=10)
        self.style.map("TButton",
            background=[('active', '#2980b9'), ('!disabled', self.accent_color)],
            foreground=[('!disabled', 'white')]
        )
        self.style.configure("Action.TButton", background=self.success_color)
        self.style.map("Action.TButton", background=[('active', '#27ae60'), ('!disabled', self.success_color)])

        self.setup_ui()

    def setup_ui(self):
        # Sidebar for navigation or stats could be added later, currently using a grid layout
        
        # Header Section
        header_frame = ttk.Frame(self.root)
        header_frame.pack(pady=30, fill="x", padx=40)
        
        ttk.Label(header_frame, text="Neural Recognition & Detection Empirical Suite", style="Header.TLabel").pack(anchor="w")
        ttk.Label(header_frame, text="PhD Thesis: Comparative Analysis of Spatial and Frequency Domain Algorithms", style="SubHeader.TLabel").pack(anchor="w")

        # Main Scrollable Area or Grid
        content_frame = ttk.Frame(self.root)
        content_frame.pack(padx=40, pady=10, fill="both", expand=True)

        # Left Column: Data & Training
        left_col = ttk.Frame(content_frame)
        left_col.place(relx=0, rely=0, relwidth=0.45, relheight=1)

        # 1. Dataset Acquisition Card
        data_card = ttk.Frame(left_col, style="Card.TFrame")
        data_card.pack(fill="x", pady=(0, 20))
        
        ttk.Label(data_card, text=" PHASE I: DATASET ACQUISITION", font=("Segoe UI", 12, "bold"), style="Card.TLabel").pack(pady=(15, 10), padx=15, anchor="w")
        
        input_frame = ttk.Frame(data_card, style="Card.TFrame")
        input_frame.pack(fill="x", padx=15, pady=10)
        
        ttk.Label(input_frame, text="Researcher/Subject Name:", style="Card.TLabel").pack(side="left")
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(input_frame, textvariable=self.name_var, font=("Segoe UI", 11))
        self.name_entry.pack(side="left", padx=10, fill="x", expand=True)
        
        self.add_btn = ttk.Button(data_card, text="START QUANTIZED COLLECTION", command=self.run_collection, style="Action.TButton")
        self.add_btn.pack(fill="x", padx=15, pady=(0, 15))

        # 2. Model Training Card
        train_card = ttk.Frame(left_col, style="Card.TFrame")
        train_card.pack(fill="x")
        
        ttk.Label(train_card, text=" PHASE II: MANIFOLD LEARNING", font=("Segoe UI", 12, "bold"), style="Card.TLabel").pack(pady=(15, 10), padx=15, anchor="w")
        
        btn_grid = ttk.Frame(train_card, style="Card.TFrame")
        btn_grid.pack(fill="x", padx=15, pady=10)
        
        ttk.Button(btn_grid, text="Train LBPH", command=lambda: self.run_training("lbph")).grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        ttk.Button(btn_grid, text="Train Eigen", command=lambda: self.run_training("eigen")).grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        ttk.Button(btn_grid, text="Train Fisher", command=lambda: self.run_training("fisher")).grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)

        # Right Column: Evaluation & Analytics
        right_col = ttk.Frame(content_frame)
        right_col.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

        # 3. Empirical Evaluation Card
        eval_card = ttk.Frame(right_col, style="Card.TFrame")
        eval_card.pack(fill="both", expand=True)
        
        ttk.Label(eval_card, text=" PHASE III: EMPIRICAL EVALUATION", font=("Segoe UI", 12, "bold"), style="Card.TLabel").pack(pady=(15, 10), padx=15, anchor="w")
        
        eval_info = "Compare computational latency and accuracy metrics:"
        ttk.Label(eval_card, text=eval_info, style="Card.TLabel", justify="left").pack(padx=15, pady=5, anchor="w")

        eval_btn_frame = ttk.Frame(eval_card, style="Card.TFrame")
        eval_btn_frame.pack(fill="x", padx=15, pady=20)

        # The 4 Algorithms for comparison
        for algo in ["Haar", "LBPH", "Eigen", "Fisher"]:
            cmd = self.run_haar_benchmark if algo == "Haar" else lambda a=algo.lower(): self.run_recognition(a)
            btn = ttk.Button(eval_btn_frame, text=f"BENCHMARK {algo.upper()}", command=cmd)
            btn.pack(fill="x", pady=5)

        ttk.Separator(eval_btn_frame, orient="horizontal").pack(fill="x", pady=15)
        
        # Abstract Viewer
        abstract_frame = ttk.Frame(eval_card, style="Card.TFrame")
        abstract_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        ttk.Label(abstract_frame, text="RESEARCH ABSTRACT:", font=("Segoe UI", 9, "bold"), style="Card.TLabel").pack(anchor="w")
        
        abstract_text = tk.Text(abstract_frame, height=8, bg=self.card_color, fg="#888", font=("Segoe UI", 9), relief="flat", wrap="word")
        abstract_text.insert("1.0", "To identify which among Fisherface, Eigenface, Haar Cascade, and LBPH will generate the most accurate results... basis of two parameters: accuracy and computational time.")
        abstract_text.config(state="disabled")
        abstract_text.pack(fill="both", expand=True)

        # Status Bar
        self.status_var = tk.StringVar(value="[SYSTEM IDLE] Ready for Empirical Phase I")
        status_frame = tk.Frame(self.root, bg="#121212", height=35)
        status_frame.pack(side="bottom", fill="x")
        
        tk.Label(status_frame, textvariable=self.status_var, bg="#121212", fg=self.accent_color, font=("Consolas", 10), padx=20).pack(side="left")
        
        self.time_var = tk.StringVar()
        tk.Label(status_frame, textvariable=self.time_var, bg="#121212", fg="#555", font=("Consolas", 10), padx=20).pack(side="right")
        self.update_clock()

    def update_clock(self):
        now = time.strftime("%H:%M:%S")
        self.time_var.set(f"SYSTEM TIME: {now}")
        self.root.after(1000, self.update_clock)

    def run_command(self, cmd_list, success_msg, error_msg):
        def task():
            try:
                self.status_var.set(f"[EXECUTING] {cmd_list[0]}")
                full_cmd = [sys.executable] + cmd_list
                process = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    self.status_var.set(f"[SUCCESS] {success_msg}")
                    messagebox.showinfo("Research Update", success_msg)
                else:
                    self.status_var.set("[ERROR] Process terminated with non-zero exit code")
                    messagebox.showerror("Process Error", f"{error_msg}\n\nDetails: {stderr}")
            except Exception as e:
                self.status_var.set("[CRITICAL] Exception in research pipeline")
                messagebox.showerror("Critical fault", str(e))

        threading.Thread(target=task).start()

    def run_collection(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Input Missing", "Please enter a subject identifier for Phase I.")
            return
        self.status_var.set(f"[ACQUIRING] Collecting samples for {name}")
        subprocess.Popen([sys.executable, "add_person.py", name])

    def run_training(self, algo):
        self.status_var.set(f"[TRAINING] Building {algo.upper()} manifold...")
        self.run_command([f"train_{algo}.py"], f"{algo.upper()} manifold synthesized successfully.", f"{algo.upper()} training failed.")

    def run_haar_benchmark(self):
        self.status_var.set("[EVALUATING] Live stream benchmarking: HAAR CASCADE DETECTION")
        subprocess.Popen([sys.executable, "benchmark_haar.py"])

    def run_recognition(self, algo):
        self.status_var.set(f"[EVALUATING] Live stream benchmarking: {algo.upper()}")
        subprocess.Popen([sys.executable, f"recog_{algo}.py"])

    def run_dlib(self):
        self.status_var.set("[EVALUATING] Live stream benchmarking: DL BASELINE")
        subprocess.Popen([sys.executable, "face_recog.py"])

if __name__ == "__main__":
    FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    root = tk.Tk()
    # Attempt to set dark title bar for Windows if possible
    try:
        from ctypes import windll, byref, sizeof, c_int
        HWND = windll.user32.GetParent(root.winfo_id())
        windll.dwmapi.DwmSetWindowAttribute(HWND, 35, byref(c_int(1)), sizeof(c_int))
    except:
        pass

    app = FaceApp(root)
    root.mainloop()
