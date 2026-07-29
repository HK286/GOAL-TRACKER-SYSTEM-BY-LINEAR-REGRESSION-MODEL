# GOAL-TRACKER-SYSTEM-BY-LINEAR-REGRESSION-MODEL
For your Productivity 
"""
GOAL TRACKER SYSTEM BY LINEAR REGRESSION MODEL
------------------------------------------------
A cross-platform (Android-ready) Kivy app that tracks progress toward
ANY goal (fitness, coding hours, revenue, pages read, etc.) and uses a
pure-Python linear regression (least squares) to predict:
  1. Your trend line (slope + intercept)
  2. R^2 (how well a straight line fits your progress)
  3. The day you will hit your target, at current pace

Why no numpy/sklearn: those are heavy/fragile to cross-compile for
Android via Buildozer. This regression is implemented from scratch in
~15 lines of pure Python so the whole app stays lightweight and the
APK build stays simple.

Run on desktop:
    pip install kivy
    python goal_tracker.py

Package for Android:
    pip install buildozer
    buildozer init          # (a buildozer.spec is already provided)
    buildozer -v android debug
"""

import json
import os
from datetime import datetime, timedelta

from kivy.app import App
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line, Rectangle
from kivy.utils import platform

DATA_FILE = "goal_tracker_data.json"

# ---------------------------------------------------------------------
# Pure-python linear regression (ordinary least squares, y = mx + b)
# ---------------------------------------------------------------------
def linear_regression(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    slope = num / den
    intercept = mean_y - slope * mean_x

    # R^2
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    return {"slope": slope, "intercept": intercept, "r2": r2}


def predict_day_for_target(model, target):
    """Return the x (day) at which the trend line hits `target`, or None."""
    slope = model["slope"]
    if slope == 0:
        return None
    return (target - model["intercept"]) / slope


# ---------------------------------------------------------------------
# Simple line-chart widget (no external chart lib needed)
# ---------------------------------------------------------------------
class ChartWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.points = []       # actual (day, value) pairs
        self.model = None      # regression result
        self.target = None
        self.bind(size=self.redraw, pos=self.redraw)

    def set_data(self, points, model=None, target=None):
        self.points = points
        self.model = model
        self.target = target
        self.redraw()

    def redraw(self, *args):
        self.canvas.clear()
        if not self.points:
            return
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if self.target is not None:
            max_y = max(max_y, self.target)
        if max_x == min_x:
            max_x = min_x + 1
        if max_y == min_y:
            max_y = min_y + 1

        pad = dp(20)
        w = self.width - 2 * pad
        h = self.height - 2 * pad

        def to_screen(x, y):
            sx = self.x + pad + (x - min_x) / (max_x - min_x) * w
            sy = self.y + pad + (y - min_y) / (max_y - min_y) * h
            return sx, sy

        with self.canvas:
            # axes
            Color(0.6, 0.6, 0.6, 1)
            Line(points=[self.x + pad, self.y + pad, self.x + pad, self.y + self.height - pad], width=1)
            Line(points=[self.x + pad, self.y + pad, self.x + self.width - pad, self.y + pad], width=1)

            # target line
            if self.target is not None:
                Color(0.9, 0.5, 0.1, 1)
                tx1, ty1 = to_screen(min_x, self.target)
                tx2, ty2 = to_screen(max_x, self.target)
                Line(points=[tx1, ty1, tx2, ty2], width=1, dash_offset=4, dash_length=6)

            # actual data points + connecting line
            Color(0.2, 0.6, 1, 1)
            flat = []
            for x, y in self.points:
                sx, sy = to_screen(x, y)
                flat += [sx, sy]
            if len(flat) >= 4:
                Line(points=flat, width=dp(2))
            for x, y in self.points:
                sx, sy = to_screen(x, y)
                Color(0.2, 0.6, 1, 1)
                Rectangle(pos=(sx - dp(3), sy - dp(3)), size=(dp(6), dp(6)))

            # regression trend line
            if self.model:
                Color(0.2, 0.85, 0.4, 1)
                y1 = self.model["slope"] * min_x + self.model["intercept"]
                y2 = self.model["slope"] * max_x + self.model["intercept"]
                x1s, y1s = to_screen(min_x, y1)
                x2s, y2s = to_screen(max_x, y2)
                Line(points=[x1s, y1s, x2s, y2s], width=dp(2))


# ---------------------------------------------------------------------
# Popup helper
# ---------------------------------------------------------------------
def show_popup(title, message):
    box = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(10))
    box.add_widget(Label(text=message, font_size=sp(15)))
    btn = Button(text="OK", size_hint_y=None, height=dp(44))
    box.add_widget(btn)
    popup = Popup(title=title, content=box, size_hint=(0.85, 0.4))
    btn.bind(on_release=popup.dismiss)
    popup.open()


# ---------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------
class GoalTrackerRoot(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.entries = []      # list of {"day": int, "value": float}
        self.goal_name = ""
        self.target_value = None
        self.start_date = None  # ISO date string for day 0

        self._build_ui()
        self._load_data()

    # ---------------- UI construction ----------------
    def _build_ui(self):
        self.padding = dp(14)
        self.spacing = dp(10)

        # ---- Title ----
        title = Label(
            text="GOAL TRACKER SYSTEM\nBY LINEAR REGRESSION MODEL",
            font_size=sp(20),
            bold=True,
            halign="center",
            valign="middle",
            size_hint_y=None,
            height=dp(70),
            color=(0.15, 0.75, 0.95, 1),
        )
        title.bind(size=lambda *a: setattr(title, "text_size", (title.width, None)))
        self.add_widget(title)

        # ---- Goal setup row ----
        setup_grid = GridLayout(cols=2, size_hint_y=None, height=dp(96), spacing=dp(6))
        self.goal_input = TextInput(hint_text="Goal name (e.g. Pages Read)", multiline=False,
                                     size_hint_y=None, height=dp(44), font_size=sp(15))
        self.target_input = TextInput(hint_text="Target value (e.g. 300)", multiline=False,
                                       input_filter="float", size_hint_y=None, height=dp(44),
                                       font_size=sp(15))
        set_goal_btn = Button(text="Set / Update Goal", size_hint_y=None, height=dp(44),
                               background_color=(0.15, 0.55, 0.85, 1))
        set_goal_btn.bind(on_release=self.set_goal)
        self.goal_status = Label(text="No goal set yet.", size_hint_y=None, height=dp(44),
                                  font_size=sp(13), color=(0.8, 0.8, 0.8, 1))

        setup_grid.add_widget(self.goal_input)
        setup_grid.add_widget(self.target_input)
        setup_grid.add_widget(set_goal_btn)
        setup_grid.add_widget(self.goal_status)
        self.add_widget(setup_grid)

        # ---- Entry input row ----
        entry_row = GridLayout(cols=3, size_hint_y=None, height=dp(48), spacing=dp(6))
        self.day_input = TextInput(hint_text="Day #", multiline=False, input_filter="int",
                                    font_size=sp(15))
        self.value_input = TextInput(hint_text="Progress value", multiline=False,
                                      input_filter="float", font_size=sp(15))
        add_btn = Button(text="+ Add Entry", background_color=(0.2, 0.7, 0.4, 1))
        add_btn.bind(on_release=self.add_entry)
        entry_row.add_widget(self.day_input)
        entry_row.add_widget(self.value_input)
        entry_row.add_widget(add_btn)
        self.add_widget(entry_row)

        # ---- Scrollable entry list ----
        self.list_layout = GridLayout(cols=1, size_hint_y=None, spacing=dp(4))
        self.list_layout.bind(minimum_height=self.list_layout.setter("height"))
        scroll = ScrollView(size_hint=(1, None), height=dp(140))
        scroll.add_widget(self.list_layout)
        self.add_widget(scroll)

        # ---- Predict button ----
        predict_btn = Button(text="Run Regression & Predict", size_hint_y=None, height=dp(50),
                              background_color=(0.85, 0.45, 0.15, 1), bold=True)
        predict_btn.bind(on_release=self.run_prediction)
        self.add_widget(predict_btn)

        # ---- Results ----
        self.result_label = Label(text="", font_size=sp(14), size_hint_y=None, height=dp(110),
                                   halign="left", valign="top")
        self.result_label.bind(size=lambda *a: setattr(
            self.result_label, "text_size", (self.result_label.width, None)))
        self.add_widget(self.result_label)

        # ---- Chart ----
        self.chart = ChartWidget(size_hint_y=1)
        self.add_widget(self.chart)

        # ---- Bottom row: reset ----
        bottom_row = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(6))
        reset_btn = Button(text="Reset All Data", background_color=(0.75, 0.2, 0.2, 1))
        reset_btn.bind(on_release=self.reset_data)
        bottom_row.add_widget(reset_btn)
        self.add_widget(bottom_row)

    # ---------------- Logic ----------------
    def set_goal(self, *args):
        name = self.goal_input.text.strip()
        target = self.target_input.text.strip()
        if not name or not target:
            show_popup("Missing info", "Enter both a goal name and a target value.")
            return
        self.goal_name = name
        self.target_value = float(target)
        if not self.start_date:
            self.start_date = datetime.now().date().isoformat()
        self.goal_status.text = f"Goal: {self.goal_name} -> Target {self.target_value}"
        self._save_data()

    def add_entry(self, *args):
        day_text = self.day_input.text.strip()
        val_text = self.value_input.text.strip()
        if not day_text or not val_text:
            show_popup("Missing info", "Enter both a day number and a progress value.")
            return
        entry = {"day": int(day_text), "value": float(val_text)}
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e["day"])
        self.day_input.text = ""
        self.value_input.text = ""
        self._refresh_list()
        self._save_data()

    def _refresh_list(self):
        self.list_layout.clear_widgets()
        for e in self.entries:
            row = BoxLayout(size_hint_y=None, height=dp(34))
            row.add_widget(Label(text=f"Day {e['day']}: {e['value']}", font_size=sp(13)))
            del_btn = Button(text="x", size_hint_x=None, width=dp(34),
                              background_color=(0.6, 0.15, 0.15, 1))
            del_btn.bind(on_release=lambda inst, ent=e: self._delete_entry(ent))
            row.add_widget(del_btn)
            self.list_layout.add_widget(row)

    def _delete_entry(self, entry):
        self.entries.remove(entry)
        self._refresh_list()
        self._save_data()

    def run_prediction(self, *args):
        if len(self.entries) < 2:
            show_popup("Not enough data", "Add at least 2 entries (different days) to run a regression.")
            return
        xs = [e["day"] for e in self.entries]
        ys = [e["value"] for e in self.entries]
        model = linear_regression(xs, ys)
        if model is None:
            show_popup("Can't compute", "All entries are on the same day - add a data point on a later day.")
            return

        lines = [
            f"Trend: value = {model['slope']:.3f} x day + {model['intercept']:.2f}",
            f"Fit quality (R^2): {model['r2']:.3f}",
        ]

        if self.target_value is not None:
            pred_day = predict_day_for_target(model, self.target_value)
            last_day = xs[-1]
            if pred_day is None:
                lines.append("Progress is flat -> can't project a completion day.")
            elif model["slope"] <= 0:
                lines.append("Current trend is flat/declining -> target won't be reached at this pace.")
            elif pred_day <= last_day:
                lines.append(f"Target already reached at current pace (around day {pred_day:.1f}).")
            else:
                days_left = pred_day - last_day
                lines.append(f"At this pace, target {self.target_value} is reached around day {pred_day:.1f} "
                              f"(~{days_left:.1f} days from your last entry).")
                if self.start_date:
                    try:
                        start = datetime.fromisoformat(self.start_date)
                        eta = start + timedelta(days=pred_day)
                        lines.append(f"Estimated date: {eta.date().isoformat()}")
                    except Exception:
                        pass
        else:
            lines.append("Set a target value above to get a completion-date prediction.")

        self.result_label.text = "\n".join(lines)
        self.chart.set_data([(e["day"], e["value"]) for e in self.entries],
                             model=model, target=self.target_value)

    def reset_data(self, *args):
        self.entries = []
        self.goal_name = ""
        self.target_value = None
        self.start_date = None
        self.goal_input.text = ""
        self.target_input.text = ""
        self.goal_status.text = "No goal set yet."
        self.result_label.text = ""
        self._refresh_list()
        self.chart.set_data([])
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)

    # ---------------- Persistence ----------------
    def _save_data(self):
        data = {
            "goal_name": self.goal_name,
            "target_value": self.target_value,
            "start_date": self.start_date,
            "entries": self.entries,
        }
        try:
            with open(DATA_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_data(self):
        if not os.path.exists(DATA_FILE):
            return
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            return
        self.goal_name = data.get("goal_name", "")
        self.target_value = data.get("target_value")
        self.start_date = data.get("start_date")
        self.entries = data.get("entries", [])
        if self.goal_name:
            self.goal_input.text = self.goal_name
            self.goal_status.text = f"Goal: {self.goal_name} -> Target {self.target_value}"
        if self.target_value is not None:
            self.target_input.text = str(self.target_value)
        self._refresh_list()


class GoalTrackerApp(App):
    def build(self):
        self.title = "Goal Tracker System by Linear Regression Model"
        # Simulate a phone aspect ratio on desktop for quick preview
        if platform not in ("android", "ios"):
            Window.size = (400, 750)
        return GoalTrackerRoot()


if __name__ == "__main__":
    GoalTrackerApp().run()
