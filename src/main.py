"""
Графический интерфейс приложения на Tkinter для интерактивного извлечения
физических эффектов и технических функций из текстов патентов. Реализует
светлую тему оформления, вкладки результатов и асинхронную обработку запросов.

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
import logging
import re
import warnings
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import csv
import json
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    from transformers.utils import logging as tl; tl.set_verbosity_error()
except Exception:
    pass

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from config import LM_STUDIO_URL, LM_STUDIO_MODEL_ID, KEYT5_MODEL_PATH
from pipeline.run_pipeline import RAGPipeline
from extraction.tech_function_extractor import TechFunctionExtractor
from preprocessing.text_preprocessor import normalize_text

BG          = "#f5f5f5"
BG_BOT      = "#e8e8e8"
BG_USER     = "#1c1c1c"
BG_INPUT    = "#ffffff"
BG_SIDE     = "#eeeeee"
BG_CARD     = "#e2e2e2"
BG_CARD_HOV = "#d6d6d6"
FG_BOT      = "#1a1a1a"
FG_USER     = "#f0f0f0"
FG_DIM      = "#999999"
FG_LABEL    = "#666666"
FG_ERR      = "#cc2222"
FG_ACCENT   = "#1a1a1a"
SEP         = "#cccccc"
BTN_IDLE    = "#1c1c1c"
BTN_HOV     = "#3a3a3a"
BTN_DIS     = "#cccccc"
BTN_TXT_IDLE = "#f5f5f5"
BTN_TXT_DIS  = "#aaaaaa"
TAB_ACTIVE   = "#1c1c1c"
TAB_INACTIVE = "#d8d8d8"
TAB_TXT_ACT  = "#f5f5f5"
TAB_TXT_INACT = "#888888"
CHECK_ON     = "#1c1c1c"
CHECK_OFF    = "#cccccc"
RADIUS       = 18

MODE_FE  = "fe"
MODE_TF  = "tf"
MODE_ALL = "all"

MODE_LABELS = {
    MODE_FE:  "Физ. эффект",
    MODE_TF:  "Тех. функции",
    MODE_ALL: "Комплексный",
}


def prettify_input_params(text):
    text = str(text or "").strip()
    if not text:
        return ""
    parts = re.split(r"Вход\d+:\s*", text)
    return "; ".join(p.strip(" .") for p in parts if p.strip())


def draw_rounded_rect(canvas, x1, y1, x2, y2, r, **kw):
    pts = [
        x1 + r, y1, x2 - r, y1,
        x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2,
        x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r,
        x1, y1 + r, x1, y1,
        x1 + r, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kw)


class RoundedBubble(tk.Canvas):
    def __init__(self, parent, text, bg_bubble, fg_text, max_wrap, font, **kw):
        super().__init__(parent, bg=BG, highlightthickness=0, bd=0, **kw)
        self._bg = bg_bubble
        self._fg = fg_text
        self._font = font
        self._wrap = max_wrap
        self._text = text
        self._pad = 14
        self._render()

    def _render(self):
        self.delete("all")
        pad = self._pad
        tmp = tk.Label(self, text=self._text, font=self._font,
                       wraplength=self._wrap, justify="left")
        tmp.update_idletasks()
        tw = tmp.winfo_reqwidth()
        th = tmp.winfo_reqheight()
        tmp.destroy()

        w = tw + pad * 2
        h = th + pad * 2
        self.configure(width=w, height=h)

        draw_rounded_rect(self, 0, 0, w, h, RADIUS,
                          fill=self._bg, outline=self._bg)
        self.create_text(pad, pad, text=self._text, anchor="nw",
                         fill=self._fg, font=self._font,
                         width=self._wrap, justify="left")


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ патентов")
        self.root.configure(bg=BG)
        self.root.geometry("1060x720")
        self.root.minsize(820, 560)

        self.pipeline = None
        self.tech_extractor = None
        self.ready = False

        self.current_mode = MODE_ALL
        self.results = []
        self.result_checks = {}

        self._build_ui()
        self._load_models()

    def _build_ui(self):
        hdr = tk.Frame(self.root, bg=BG, pady=10)
        hdr.pack(fill="x", padx=20)

        tk.Label(
            hdr, text="Анализ патентов",
            bg=BG, fg="#1a1a1a",
            font=("Helvetica Neue", 16, "bold")
        ).pack(side="left")

        self.status_lbl = tk.Label(
            hdr, text="● загрузка...",
            bg=BG, fg=FG_DIM,
            font=("Helvetica Neue", 11)
        )
        self.status_lbl.pack(side="right")

        self._build_mode_tabs(hdr)

        tk.Frame(self.root, bg=SEP, height=1).pack(fill="x")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        self._build_sidebar(body)

        tk.Frame(body, bg=SEP, width=1).pack(side="left", fill="y")

        self._build_chat(body)

        tk.Frame(self.root, bg=SEP, height=1).pack(fill="x")
        self._build_input_bar()

    def _build_mode_tabs(self, parent):
        tabs_frame = tk.Frame(parent, bg=BG)
        tabs_frame.pack(side="right", padx=(0, 16))

        self.status_lbl.pack_forget()

        self._tab_buttons = {}
        for mode in [MODE_FE, MODE_TF, MODE_ALL]:
            btn = tk.Canvas(
                tabs_frame, width=110, height=28,
                bg=BG, highlightthickness=0, bd=0
            )
            btn.pack(side="left", padx=2)
            btn.bind("<Button-1>", lambda e, m=mode: self._set_mode(m))
            self._tab_buttons[mode] = btn

        self.status_lbl.pack(side="right", padx=(16, 0))
        self._draw_tabs()

    def _draw_tabs(self):
        for mode, btn in self._tab_buttons.items():
            btn.delete("all")
            w, h = 110, 28
            is_active = (mode == self.current_mode)
            fill = TAB_ACTIVE if is_active else TAB_INACTIVE
            fg = TAB_TXT_ACT if is_active else TAB_TXT_INACT
            draw_rounded_rect(btn, 0, 0, w, h, 14, fill=fill, outline=fill)
            btn.create_text(w // 2, h // 2, text=MODE_LABELS[mode],
                            fill=fg, font=("Helvetica Neue", 11, "bold"))

    def _set_mode(self, mode):
        self.current_mode = mode
        self._draw_tabs()

    def _build_sidebar(self, parent):
        self.sidebar = tk.Frame(parent, bg=BG_SIDE, width=240)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        side_hdr = tk.Frame(self.sidebar, bg=BG_SIDE, pady=10, padx=12)
        side_hdr.pack(fill="x")
        tk.Label(
            side_hdr, text="История", bg=BG_SIDE, fg="#1a1a1a",
            font=("Helvetica Neue", 13, "bold")
        ).pack(side="left")

        self.counter_lbl = tk.Label(
            side_hdr, text="0", bg=BG_SIDE, fg=FG_DIM,
            font=("Helvetica Neue", 11)
        )
        self.counter_lbl.pack(side="right")

        tk.Frame(self.sidebar, bg=SEP, height=1).pack(fill="x")

        save_frame = tk.Frame(self.sidebar, bg=BG_SIDE, pady=8, padx=8)
        save_frame.pack(side="bottom", fill="x")

        tk.Frame(self.sidebar, bg=SEP, height=1).pack(side="bottom", fill="x")

        self._make_side_btn(save_frame, "Сохранить выбранные", self._save_selected)
        self._make_side_btn(save_frame, "Сохранить все", self._save_all)

        self.selected_lbl = tk.Label(
            save_frame, text="Выбрано: 0", bg=BG_SIDE, fg=FG_DIM,
            font=("Helvetica Neue", 10)
        )
        self.selected_lbl.pack(pady=(4, 0))

        self._cards_canvas = tk.Canvas(
            self.sidebar, bg=BG_SIDE, highlightthickness=0, bd=0
        )
        cards_sb = tk.Scrollbar(
            self.sidebar, orient="vertical",
            command=self._cards_canvas.yview,
            bg=BG_SIDE, troughcolor=BG_SIDE, width=4,
            relief="flat", bd=0
        )
        self._cards_canvas.configure(yscrollcommand=cards_sb.set)
        cards_sb.pack(side="right", fill="y")
        self._cards_canvas.pack(side="left", fill="both", expand=True)

        self._cards_frame = tk.Frame(self._cards_canvas, bg=BG_SIDE)
        self._cards_win = self._cards_canvas.create_window(
            (0, 0), window=self._cards_frame, anchor="nw"
        )
        self._cards_frame.bind(
            "<Configure>",
            lambda e: self._cards_canvas.configure(
                scrollregion=self._cards_canvas.bbox("all")
            )
        )
        self._cards_canvas.bind(
            "<Configure>",
            lambda e: self._cards_canvas.itemconfig(self._cards_win, width=e.width)
        )

    def _make_side_btn(self, parent, text, command):
        btn = tk.Label(
            parent, text=text, bg="#d0d0d0", fg="#333333",
            font=("Helvetica Neue", 11), padx=10, pady=6,
            cursor="hand2"
        )
        btn.pack(fill="x", pady=2)
        btn.bind("<Button-1>", lambda e: command())
        btn.bind("<Enter>", lambda e: btn.configure(bg="#bfbfbf"))
        btn.bind("<Leave>", lambda e: btn.configure(bg="#d0d0d0"))

    def _build_chat(self, parent):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0, bd=0)
        sb = tk.Scrollbar(
            wrap, orient="vertical", command=self.canvas.yview,
            bg=BG, troughcolor=BG, width=4, relief="flat", bd=0
        )
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.msg_frame = tk.Frame(self.canvas, bg=BG)
        self._win = self.canvas.create_window(
            (0, 0), window=self.msg_frame, anchor="nw"
        )

        self.msg_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self._win, width=e.width)
        )
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        )

        self._bot_bubble(
            "Вставьте текст патента — я извлеку физические эффекты "
            "и технические функции.\n\n"
            "Выберите режим анализа вверху: физ. эффект, "
            "тех. функции или комплексный.",
            greeting=True
        )

    def _build_input_bar(self):
        bottom = tk.Frame(self.root, bg=BG, pady=10)
        bottom.pack(fill="x", padx=16, pady=(4, 10))

        self._input_cv = tk.Canvas(
            bottom, bg=BG, highlightthickness=0, bd=0,
            relief="flat", height=56
        )
        self._input_cv.pack(fill="x", pady=(0, 8))

        self.text_input = tk.Text(
            self._input_cv, bg=BG_INPUT, fg="#1a1a1a",
            insertbackground="#333333",
            font=("Helvetica Neue", 13), relief="flat", bd=0,
            highlightthickness=0, undo=True, height=2, wrap="word",
            padx=0, pady=0, selectbackground="#bbbbbb"
        )

        self._input_cv.create_window(
            16, 10, window=self.text_input, anchor="nw", tags="entry_win"
        )

        self._input_cv.bind("<Configure>", self._resize_input)
        self.text_input.bind("<Return>", self._on_enter)
        self.text_input.bind("<<Paste>>", self._on_paste)
        self.text_input.bind("<Command-v>", self._on_paste)
        self.text_input.bind("<Command-V>", self._on_paste)
        self.text_input.bind("<Control-v>", self._on_paste)
        self.text_input.bind("<Control-V>", self._on_paste)

        self.root.bind_all("<Command-v>", self._on_paste_global)
        self.root.bind_all("<Command-V>", self._on_paste_global)
        self.root.bind_all("<Control-v>", self._on_paste_global)
        self.root.bind_all("<Control-V>", self._on_paste_global)

        self.text_input.focus_set()

        btn_row = tk.Frame(bottom, bg=BG)
        btn_row.pack(fill="x")

        tk.Label(
            btn_row,
            text="Enter — отправить  ·  Shift+Enter — новая строка  ·  ⌘V — вставить",
            bg=BG, fg=FG_DIM, font=("Helvetica Neue", 10)
        ).pack(side="left")

        paste_btn = tk.Canvas(
            btn_row, width=100, height=34, bg=BG,
            highlightthickness=0, bd=0
        )
        paste_btn.pack(side="right", padx=(10, 0))
        draw_rounded_rect(paste_btn, 0, 0, 100, 34, 17,
                          fill="#d0d0d0", outline="#d0d0d0")
        paste_btn.create_text(50, 17, text="Вставить",
                              fill="#1a1a1a", font=("Helvetica Neue", 11, "bold"))
        paste_btn.bind("<Button-1>", lambda e: self._paste_button())

        self._btn_cv = tk.Canvas(
            btn_row, width=140, height=34, bg=BG,
            highlightthickness=0, bd=0
        )
        self._btn_cv.pack(side="right")
        self._draw_btn("disabled")
        self._btn_cv.bind("<Button-1>", lambda e: self.ready and self._send())
        self._btn_cv.bind("<Enter>",
                          lambda e: self.ready and self._draw_btn("hover"))
        self._btn_cv.bind("<Leave>",
                          lambda e: self.ready and self._draw_btn("normal"))

    def _resize_input(self, event):
        w, h = event.width, 56
        self._input_cv.configure(height=h)
        self._input_cv.delete("input_bg")
        draw_rounded_rect(
            self._input_cv, 0, 0, w, h, RADIUS,
            fill=BG_INPUT, outline=BG_INPUT, width=0, tags="input_bg"
        )
        self._input_cv.tag_lower("input_bg")
        inner_x = 16
        inner_w = max(50, w - inner_x * 2)
        self._input_cv.itemconfig("entry_win", width=inner_w)
        self._input_cv.coords("entry_win", inner_x, 10)

    def _draw_btn(self, state):
        c = self._btn_cv
        c.delete("all")
        W, H = 140, 34
        if state == "disabled":
            draw_rounded_rect(c, 0, 0, W, H, H // 2,
                              fill=BTN_DIS, outline=BTN_DIS)
            c.create_text(W // 2, H // 2, text="Отправить →",
                          fill=BTN_TXT_DIS, font=("Helvetica Neue", 11, "bold"))
        elif state == "hover":
            draw_rounded_rect(c, 0, 0, W, H, H // 2,
                              fill=BTN_HOV, outline=BTN_HOV)
            c.create_text(W // 2, H // 2, text="Отправить →",
                          fill=BTN_TXT_IDLE, font=("Helvetica Neue", 11, "bold"))
        else:
            draw_rounded_rect(c, 0, 0, W, H, H // 2,
                              fill=BTN_IDLE, outline=BTN_IDLE)
            c.create_text(W // 2, H // 2, text="Отправить →",
                          fill=BTN_TXT_IDLE, font=("Helvetica Neue", 11, "bold"))

    def _on_paste_global(self, event=None):
        self.text_input.focus_force()
        return self._on_paste(event)

    def _on_paste(self, event=None):
        self.text_input.focus_force()
        text = ""
        try:
            text = self.root.clipboard_get()
        except Exception:
            try:
                import subprocess
                result = subprocess.run(
                    ["pbpaste"], capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    text = result.stdout
            except Exception:
                text = ""
        if text:
            try:
                self.text_input.insert(tk.INSERT, text)
                self.text_input.see(tk.INSERT)
            except Exception:
                pass
        return "break"

    def _paste_button(self):
        self.text_input.focus_force()
        text = ""
        try:
            text = self.root.clipboard_get()
        except Exception:
            pass
        if not text:
            try:
                import subprocess
                result = subprocess.run(
                    ["pbpaste"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    text = result.stdout
            except Exception:
                pass
        if text:
            self.text_input.insert("insert", text)
            self.text_input.see("insert")

    def _load_models(self):
        def load():
            try:
                self._status("● загрузка RAG-системы...", FG_DIM)
                self.pipeline = RAGPipeline(
                    lm_studio_url=LM_STUDIO_URL,
                    model_id=LM_STUDIO_MODEL_ID
                )
                self._status("● загрузка KeyT5-large...", FG_DIM)
                self.tech_extractor = TechFunctionExtractor(model_path=KEYT5_MODEL_PATH)
                self.ready = True
                self._status("● готово", "#1a1a1a")
                self.root.after(0, lambda: self._draw_btn("normal"))
            except Exception as e:
                self._status("● ошибка загрузки", FG_ERR)
                err_msg = f"⚠ Ошибка при загрузке моделей:\n{e}"
                self.root.after(
                    0,
                    lambda msg=err_msg: self._bot_bubble(msg, error=True)
                )

        threading.Thread(target=load, daemon=True).start()

    def _status(self, text, color):
        self.root.after(0, lambda: self.status_lbl.config(text=text, fg=color))

    def _on_enter(self, event):
        if event.state & 0x1:
            return
        self._send()
        return "break"

    def _send(self):
        if not self.ready:
            return
        text = self.text_input.get("1.0", "end").strip()

        if not text:
            self._bot_bubble("⚠ Введите текст патента для анализа.", error=True)
            return
        if len(text) < 20:
            self._bot_bubble(
                "⚠ Текст слишком короткий. Вставьте полное описание "
                "физического эффекта или фрагмент патента.",
                error=True
            )
            return

        self.text_input.delete("1.0", "end")
        self.ready = False
        self._draw_btn("disabled")

        mode = self.current_mode
        mode_label = MODE_LABELS[mode]
        self._status(f"● анализ ({mode_label})...", FG_DIM)
        self._user_bubble(text)
        self._th_row = self._thinking_bubble()

        def run():
            try:
                n = normalize_text(text)
                fi = fo = fb = None
                fe_error = None
                funcs = []

                if mode in (MODE_FE, MODE_ALL):
                    try:
                        res = self.pipeline.run(n)
                        if res and res.get("status") == "ok":
                            r = res.get("result", {})
                            fi = prettify_input_params(
                                r.get("input_params", "")) or "—"
                            fb = r.get("object", "") or "—"
                            fo = r.get("output_params", "") or "—"
                        else:
                            stage = res.get("stage", "") if res else ""
                            msg = res.get("message", "") if res else ""
                            if stage == "raw_extraction":
                                fe_error = (
                                    "Не удалось извлечь структуру эффекта "
                                    "(модель не вернула корректный ответ)."
                                )
                            elif stage == "normalization":
                                fe_error = "Ошибка нормализации результата."
                            else:
                                fe_error = (msg
                                            or "Не удалось извлечь физический эффект.")
                    except Exception as e:
                        fe_error = f"Ошибка RAG-системы: {e}"

                if mode in (MODE_TF, MODE_ALL):
                    try:
                        funcs = self.tech_extractor.extract(n)
                    except Exception:
                        funcs = []

                self.root.after(
                    0,
                    lambda: self._finish(
                        fi, fb, fo, fe_error, funcs, text, mode
                    )
                )
            except Exception as e:
                self.root.after(
                    0,
                    lambda: self._finish(
                        None, None, None, str(e), [], text, mode
                    )
                )

        threading.Thread(target=run, daemon=True).start()

    def _finish(self, fi, fb, fo, fe_error, funcs, source_text, mode):
        if hasattr(self, "_th_row") and self._th_row:
            self._th_row.destroy()

        self._result_bubble(fi, fb, fo, fe_error, funcs, mode)
        self._add_result_to_history(
            fi, fb, fo, fe_error, funcs, source_text, mode
        )

        self.ready = True
        self._draw_btn("normal")
        self._status("● готово", "#1a1a1a")

    def _update_selected_count(self):
        count = sum(
            1 for v in self.result_checks.values() if v.get()
        )
        self.selected_lbl.config(text=f"Выбрано: {count}")

    def _add_result_to_history(self, fi, fb, fo, fe_error, funcs,
                                source_text, mode):
        idx = len(self.results) + 1
        timestamp = datetime.now().strftime("%H:%M")

        entry = {
            "id": idx,
            "timestamp": timestamp,
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "source_text": source_text,
            "input_params": fi or "",
            "object": fb or "",
            "output_params": fo or "",
            "fe_error": fe_error or "",
            "tech_functions": funcs or [],
            "is_error": bool(fe_error and not fi),
        }
        self.results.append(entry)

        var = tk.BooleanVar(value=False)
        var.trace_add("write", lambda *_: self._update_selected_count())
        self.result_checks[idx] = var

        self._render_card(entry, var)
        self.counter_lbl.config(text=str(len(self.results)))

    def _render_card(self, entry, var):
        card = tk.Frame(self._cards_frame, bg=BG_CARD, pady=6, padx=8)
        card.pack(fill="x", padx=6, pady=3)

        top_row = tk.Frame(card, bg=BG_CARD)
        top_row.pack(fill="x")

        cb = tk.Checkbutton(
            top_row, variable=var,
            bg=BG_CARD, fg=CHECK_ON,
            activebackground=BG_CARD, activeforeground=CHECK_ON,
            selectcolor=BG_CARD, highlightthickness=0, bd=0
        )
        cb.pack(side="left")

        header_text = (f"#{entry['id']}  {entry['timestamp']}  · "
                       f" {entry['mode_label']}")
        tk.Label(
            top_row, text=header_text,
            bg=BG_CARD, fg=FG_LABEL,
            font=("Helvetica Neue", 10)
        ).pack(side="left", padx=(2, 0))

        if entry["is_error"]:
            tk.Label(
                top_row, text="⚠", bg=BG_CARD, fg=FG_ERR,
                font=("Helvetica Neue", 10)
            ).pack(side="right")

        preview = entry["source_text"][:60].replace("\n", " ")
        if len(entry["source_text"]) > 60:
            preview += "..."
        tk.Label(
            card, text=preview,
            bg=BG_CARD, fg=FG_DIM,
            font=("Helvetica Neue", 10),
            anchor="w", justify="left",
            wraplength=200
        ).pack(fill="x", pady=(2, 0))

        all_widgets = [card, top_row]
        for w in card.winfo_children():
            all_widgets.append(w)
        for w in top_row.winfo_children():
            all_widgets.append(w)

        def on_enter(e):
            for widget in all_widgets:
                try:
                    widget.configure(bg=BG_CARD_HOV)
                except Exception:
                    pass

        def on_leave(e):
            for widget in all_widgets:
                try:
                    widget.configure(bg=BG_CARD)
                except Exception:
                    pass

        for widget in all_widgets:
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

        self.root.after(
            50,
            lambda: self._cards_canvas.yview_moveto(1.0)
        )

    def _get_selected_results(self):
        selected = []
        for entry in self.results:
            var = self.result_checks.get(entry["id"])
            if var and var.get():
                selected.append(entry)
        return selected

    def _save_selected(self):
        selected = self._get_selected_results()
        if not selected:
            messagebox.showinfo(
                "Сохранение",
                "Не выбрано ни одного результата.\n"
                "Отметьте чекбоксы в панели истории."
            )
            return
        self._save_dialog(selected)

    def _save_all(self):
        if not self.results:
            messagebox.showinfo("Сохранение", "Нет результатов для сохранения.")
            return
        self._save_dialog(self.results)

    def _save_dialog(self, data):
        path = filedialog.asksaveasfilename(
            title="Сохранить результаты",
            defaultextension=".json",
            filetypes=[
                ("JSON", "*.json"),
                ("CSV", "*.csv"),
                ("Текстовый файл", "*.txt"),
            ]
        )
        if not path:
            return

        try:
            ext = path.rsplit(".", 1)[-1].lower() if "." in path else "json"
            if ext == "json":
                self._save_json(path, data)
            elif ext == "csv":
                self._save_csv(path, data)
            else:
                self._save_txt(path, data)
            messagebox.showinfo(
                "Сохранено",
                f"Сохранено {len(data)} результат(ов) в:\n{path}"
            )
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

    @staticmethod
    def _save_json(path, data):
        export = []
        for entry in data:
            export.append({
                "id": entry["id"],
                "timestamp": entry["timestamp"],
                "mode": entry["mode_label"],
                "source_text": entry["source_text"],
                "input_params": entry["input_params"],
                "object": entry["object"],
                "output_params": entry["output_params"],
                "tech_functions": entry["tech_functions"],
                "error": entry["fe_error"],
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _save_csv(path, data):
        fieldnames = [
            "id", "timestamp", "mode", "source_text",
            "input_params", "object", "output_params",
            "tech_functions", "error"
        ]
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in data:
                writer.writerow({
                    "id": entry["id"],
                    "timestamp": entry["timestamp"],
                    "mode": entry["mode_label"],
                    "source_text": entry["source_text"],
                    "input_params": entry["input_params"],
                    "object": entry["object"],
                    "output_params": entry["output_params"],
                    "tech_functions": "; ".join(entry["tech_functions"]),
                    "error": entry["fe_error"],
                })

    @staticmethod
    def _save_txt(path, data):
        with open(path, "w", encoding="utf-8") as f:
            f.write("РЕЗУЛЬТАТЫ АНАЛИЗА ПАТЕНТОВ\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Количество результатов: {len(data)}\n")
            f.write("=" * 60 + "\n\n")

            for entry in data:
                f.write(
                    f"--- Анализ #{entry['id']} "
                    f"({entry['timestamp']}, {entry['mode_label']}) ---\n\n"
                )
                f.write(f"Исходный текст:\n{entry['source_text']}\n\n")

                if entry["fe_error"] and not entry["input_params"]:
                    f.write(f"Ошибка: {entry['fe_error']}\n\n")
                else:
                    if entry["input_params"]:
                        f.write(f"Вход:   {entry['input_params']}\n")
                        f.write(f"Объект: {entry['object']}\n")
                        f.write(f"Выход:  {entry['output_params']}\n\n")

                if entry["tech_functions"]:
                    f.write("Технические функции:\n")
                    for func in entry["tech_functions"]:
                        f.write(f"  - {func}\n")
                    f.write("\n")

                f.write("=" * 60 + "\n\n")

    def _add_row(self, side):
        row = tk.Frame(self.msg_frame, bg=BG, pady=4)
        row.pack(fill="x", padx=14)
        return row

    def _user_bubble(self, text):
        row = self._add_row("right")
        b = RoundedBubble(
            row, text, bg_bubble=BG_USER, fg_text=FG_USER,
            max_wrap=460, font=("Helvetica Neue", 13)
        )
        b.pack(side="right", anchor="n", padx=(60, 0))
        self._scroll()

    def _bot_bubble(self, text, greeting=False, error=False):
        row = self._add_row("left")
        fg = FG_ERR if error else FG_ACCENT
        b = RoundedBubble(
            row, text, bg_bubble=BG_BOT, fg_text=fg,
            max_wrap=520, font=("Helvetica Neue", 13)
        )
        b.pack(side="left", anchor="n", padx=(0, 60))
        self._scroll()

    def _thinking_bubble(self):
        row = self._add_row("left")
        b = RoundedBubble(
            row, "анализирую...", bg_bubble=BG_BOT, fg_text=FG_DIM,
            max_wrap=300, font=("Helvetica Neue", 13, "italic")
        )
        b.pack(side="left", anchor="n", padx=(0, 60))
        self._scroll()
        return row

    def _result_bubble(self, fi, fb, fo, fe_error, funcs, mode):
        row = self._add_row("left")
        lines = []

        if mode in (MODE_FE, MODE_ALL):
            if fe_error and not fi:
                lines.append(f"⚠ {fe_error}")
            else:
                lines.append("Физический эффект:")
                lines.append(f"  вход:    {fi}")
                lines.append(f"  объект:  {fb}")
                lines.append(f"  выход:   {fo}")

        if mode in (MODE_TF, MODE_ALL):
            if lines:
                lines.append("")
            lines.append("Технические функции:")
            if funcs:
                for func in funcs:
                    lines.append(f"  ●  {func}")
            else:
                lines.append("  не найдены")

        full_text = "\n".join(lines)
        has_error = bool(fe_error and not fi)
        fg = FG_ERR if has_error else FG_BOT

        b = RoundedBubble(
            row, full_text, bg_bubble=BG_BOT, fg_text=fg,
            max_wrap=540, font=("Helvetica Neue", 12)
        )
        b.pack(side="left", anchor="n", padx=(0, 40))
        self._scroll()

    def _scroll(self):
        self.root.after(60, lambda: self.canvas.yview_moveto(1.0))



def main():
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()