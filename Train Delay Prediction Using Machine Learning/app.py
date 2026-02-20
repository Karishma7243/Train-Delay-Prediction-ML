import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, Canvas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# =========================
# ADMIN LOGIN WINDOW
# =========================
class AdminLogin:
    def __init__(self, root):
        self.root = root
        self.root.title("🚆 Admin Login — Train Delay Prediction")
        self.root.geometry("900x600")
        self.root.resizable(False, False)

        # Create gradient background
        self.canvas = Canvas(root, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.draw_gradient("#141E30", "#243B55")

        # Animated Title
        self.title_text = "🚆 TRAIN DELAY PREDICTION LOGIN 🚆"
        self.title_label = tk.Label(root, text="", font=("Segoe UI", 22, "bold"), bg="#141E30", fg="cyan")
        self.title_label.place(relx=0.5, rely=0.2, anchor="center")
        self.animate_title(0)

        # Login Frame
        self.login_frame = tk.Frame(root, bg="#1b1b2f", bd=3, relief="ridge")
        self.login_frame.place(relx=0.5, rely=0.55, anchor="center", width=380, height=320)

        tk.Label(self.login_frame, text="Admin Login", font=("Segoe UI", 18, "bold"),
                 bg="#1b1b2f", fg="#00e6e6").pack(pady=15)

        tk.Label(self.login_frame, text="Username", font=("Segoe UI", 12),
                 bg="#1b1b2f", fg="white").pack(pady=(10, 5))
        self.username_entry = tk.Entry(self.login_frame, font=("Segoe UI", 12), width=25, relief="flat", justify="center")
        self.username_entry.pack(pady=5)

        tk.Label(self.login_frame, text="Password", font=("Segoe UI", 12),
                 bg="#1b1b2f", fg="white").pack(pady=(15, 5))
        self.password_entry = tk.Entry(self.login_frame, font=("Segoe UI", 12),
                                       width=25, show="•", relief="flat", justify="center")
        self.password_entry.pack(pady=5)

        tk.Button(self.login_frame, text="LOGIN", font=("Segoe UI", 12, "bold"),
                  bg="#00bcd4", fg="white", width=15, relief="flat",
                  command=self.check_login, cursor="hand2").pack(pady=25)

        tk.Label(root, text="© 2025 Train Delay Prediction | Powered by AI",
                 font=("Segoe UI", 10), bg="#141E30", fg="white").place(relx=0.5, rely=0.95, anchor="center")

    # === Animated Title ===
    def animate_title(self, i):
        if i <= len(self.title_text):
            self.title_label.config(text=self.title_text[:i])
            self.root.after(100, self.animate_title, i + 1)
        else:
            self.root.after(1500, self.animate_title, 0)

    # === Gradient Background ===
    def draw_gradient(self, color1, color2):
        width, height = 900, 600
        r1, g1, b1 = self.canvas.winfo_rgb(color1)
        r2, g2, b2 = self.canvas.winfo_rgb(color2)
        r_ratio = (r2 - r1) / height
        g_ratio = (g2 - g1) / height
        b_ratio = (b2 - b1) / height
        for i in range(height):
            nr = int(r1 + (r_ratio * i))
            ng = int(g1 + (g_ratio * i))
            nb = int(b1 + (b_ratio * i))
            color = f'#{nr >> 8:02x}{ng >> 8:02x}{nb >> 8:02x}'
            self.canvas.create_line(0, i, width, i, fill=color)
        self.canvas.tag_lower("all")

    # === Check Credentials ===
    def check_login(self):
        user = self.username_entry.get().strip()
        pwd = self.password_entry.get().strip()
        if user == "admin" and pwd == "admin":
            messagebox.showinfo("Login Successful", "Welcome, Admin!")
            self.root.destroy()
            launch_main_app()
        else:
            messagebox.showerror("Access Denied", "Invalid Username or Password!")


# =========================
# MAIN APPLICATION WINDOW
# =========================
class TrainDelayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Train Delay Prediction System")
        self.root.geometry("1080x620")
        self.root.configure(bg="#1f1f2e")

        self.data = None
        self.models = {}
        self.last_split = None

        # Animated Title
        self.title_text = "🚆 TRAIN DELAY PREDICTION SYSTEM 🚆"
        self.title_label = tk.Label(root, text="", font=("Segoe UI", 22, "bold"), bg="#1f1f2e", fg="cyan")
        self.title_label.pack(pady=10)
        self.animate_title(0)

        # Layout
        frame = tk.Frame(root, bg="#1f1f2e")
        frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.text_area = scrolledtext.ScrolledText(
            frame, width=80, height=25, font=("Consolas", 11), bg="#111", fg="white", wrap="word"
        )
        self.text_area.pack(side="left", padx=15, pady=10, fill="both", expand=True)

        btn_frame = tk.Frame(frame, bg="#1f1f2e")
        btn_frame.pack(side="right", padx=10)

        buttons = [
            ("📂 Upload Dataset", self._upload_dataset),
            ("🧹 Preprocess Data", self._preprocess_data),
            ("🌳 Train Random Forest", self._train_rf),
            ("⚡ Train XGBoost", self._train_xgb),
            ("💡 Train LightGBM", self._train_lgbm),
            ("🧠 Train LSTM", self._train_lstm),
            ("📊 Show Graphs", self._show_graphs),
            ("🔮 Predict (Upload File)", self._predict_file),
            ("🚪 Logout", self._logout)
        ]
        for text, cmd in buttons:
            tk.Button(
                btn_frame, text=text, font=("Segoe UI", 11, "bold"),
                bg="#2b2b4a", fg="white", width=22, height=2,
                relief="raised", command=cmd, cursor="hand2"
            ).pack(pady=8)

    # Animated title
    def animate_title(self, i):
        if i <= len(self.title_text):
            self.title_label.config(text=self.title_text[:i])
            self.root.after(100, self.animate_title, i + 1)
        else:
            self.root.after(1000, self.animate_title, 0)

    # Utility logging
    def _clear_log(self): self.text_area.delete(1.0, tk.END)
    def _write_log(self, text): self.text_area.insert(tk.END, f"{text}\n"); self.text_area.see(tk.END)

    # ====== Upload Dataset ======
    def _upload_dataset(self):
        file = filedialog.askopenfilename(initialdir='dataset', filetypes=[("CSV Files", "*.csv")])
        if not file: return
        try:
            self.data = pd.read_csv(file)
            self._clear_log()
            self._write_log(f"✅ Loaded Dataset: {os.path.basename(file)}")
            self._write_log(f"📊 Shape: {self.data.shape}")
            self._write_log("\nFirst 5 Rows:\n")
            self._write_log(str(self.data.head()))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{e}")

    # ====== Preprocess ======
    def _preprocess_data(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please upload a dataset first.")
            return
        self._clear_log()
        df = self.data.copy()
        self._write_log("🧹 Starting Data Cleaning...")

        df.dropna(inplace=True)
        self._write_log(f"✅ Removed missing values — Remaining: {df.shape}")

        target_col = None
        for col in df.columns:
            if "delay" in col.lower():
                target_col = col
                break

        if target_col is None:
            messagebox.showerror("Error", "Dataset must contain a column like 'Delay'.")
            return

        if target_col != "Delay":
            df.rename(columns={target_col: "Delay"}, inplace=True)
            self._write_log(f"ℹ️ Using '{target_col}' as target column.")

        X = df.drop(columns=["Delay"])
        y = (df["Delay"] > 5).astype(int)
        self._write_log("📊 Converted delay minutes into binary classes (1=Delayed, 0=On Time).")

        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            self._write_log(f"🔡 Encoded categorical columns: {cat_cols}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.last_split = (X_train, X_test, y_train, y_test)
        self._write_log(f"\n✅ Train-Test Split Done: Train {X_train.shape}, Test {X_test.shape}")

    # ====== Train Models ======
    def _train_rf(self): self._train_model("Random Forest", RandomForestClassifier(n_estimators=150, random_state=42))
    def _train_xgb(self):
        self._train_model("XGBoost", XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42))
    def _train_lgbm(self): self._train_model("LightGBM", LGBMClassifier(objective='binary', random_state=42))

    def _train_model(self, name, model):
        if not self.last_split:
            messagebox.showinfo("Info", "Please preprocess data first.")
            return
        self._clear_log()

        def train():
            X_train, X_test, y_train, y_test = self.last_split
            sc = StandardScaler()
            X_train_s = sc.fit_transform(X_train)
            X_test_s = sc.transform(X_test)

            self._write_log(f"🔹 Training {name}...")
            model.fit(X_train_s, y_train)
            acc = model.score(X_test_s, y_test)
            self.models[name] = (model, sc, acc)
            self._write_log(f"✅ {name} Accuracy: {acc:.4f}")
            self._write_log("🎯 Training Complete!")

        threading.Thread(target=train).start()

    def _train_lstm(self):
        if not self.last_split:
            messagebox.showinfo("Info", "Please preprocess data first.")
            return
        self._clear_log()

        def lstm_train():
            X_train, X_test, y_train, y_test = self.last_split
            sc = StandardScaler()
            X_train_s = sc.fit_transform(X_train)
            X_test_s = sc.transform(X_test)

            self._write_log("🧠 Training LSTM...")
            model = Sequential([
                LSTM(64, input_shape=(1, X_train_s.shape[1])),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1])),
                      y_train, epochs=5, batch_size=16, verbose=0)
            _, acc = model.evaluate(X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1])), y_test, verbose=0)
            self.models["LSTM"] = (model, sc, acc)
            self._write_log(f"✅ LSTM Accuracy: {acc:.4f}")
            self._write_log("🎯 LSTM Training Complete!")

        threading.Thread(target=lstm_train).start()

    # ====== Show Graphs ======
    def _show_graphs(self):
        if not self.models:
            messagebox.showinfo("Info", "Train models first.")
            return
        names = list(self.models.keys())
        accs = [self.models[k][2] for k in names]
        plt.figure(figsize=(8, 4))
        plt.bar(names, accs)
        plt.title("Model Accuracy Comparison")
        plt.ylabel("Accuracy")
        plt.show()

    # ====== Predict File ======
    def _predict_file(self):
        self._clear_log()
        if not self.models:
            self._write_log("Train at least one model first.")
            return
        best_name = max(self.models, key=lambda k: self.models[k][2])
        model, sc, acc = self.models[best_name]
        self._write_log(f"✅ Using {best_name} (Accuracy {acc:.4f})")

        file = filedialog.askopenfilename(initialdir='dataset', filetypes=[("CSV files", "*.csv")])
        if not file:
            return
        try:
            df = pd.read_csv(file).dropna()
            if "Delay" in df.columns:
                df.drop(columns=["Delay"], inplace=True)
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols):
                df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

            X_train = self.last_split[0]
            for c in X_train.columns:
                if c not in df.columns:
                    df[c] = 0
            df = df[X_train.columns]

            Xs = sc.transform(df)
            preds = model.predict(Xs) if best_name != "LSTM" else (model.predict(Xs.reshape((Xs.shape[0], 1, Xs.shape[1]))) > 0.5).astype(int).flatten()

            df["Predicted_Status"] = ["Delayed" if p == 1 else "On Time" for p in preds]
            save = os.path.join(os.getcwd(), "predicted_results.csv")
            df.to_csv(save, index=False)
            self._write_log(f"✅ Saved predictions to {save}\n")
            self._write_log("📊 Sample Predictions:\n")
            self._write_log(str(df[["Predicted_Status"]].head(10)))
            messagebox.showinfo("Prediction Complete", f"Predictions displayed in text area and saved to:\n{save}")
        except Exception as e:
            self._write_log(f"❌ Prediction failed: {e}")

    # ====== Logout ======
    def _logout(self):
        if messagebox.askyesno("Logout", "Do you want to exit?"):
            self.root.destroy()


# =========================
# APP LAUNCHER
# =========================
def launch_main_app():
    main_root = tk.Tk()
    TrainDelayApp(main_root)
    main_root.mainloop()


if __name__ == "__main__":
    login_root = tk.Tk()
    AdminLogin(login_root)
    login_root.mainloop()
