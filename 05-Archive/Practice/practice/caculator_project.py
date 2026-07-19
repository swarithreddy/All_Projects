# animated_advanced_scientific_calculator_gui.py

import tkinter as tk
from tkinter import ttk
import math

class AnimatedCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Animated Scientific Calculator")
        self.root.geometry("400x600")
        self.root.resizable(False, False)

        # Memory and state variables
        self.memory = None
        self.angle_mode = 'degrees'
        self.expression = ""

        # Create calculator interface
        self.create_display()
        self.create_buttons()

        # Bind the Enter key to evaluate the expression
        self.root.bind("<Return>", lambda event: self.evaluate_expression())

    def create_display(self):
        """Create the calculator display."""
        self.display_frame = ttk.Frame(self.root)
        self.display_frame.pack(pady=10)

        self.display_var = tk.StringVar()
        self.display_entry = ttk.Entry(
            self.display_frame, textvariable=self.display_var,
            font=("Arial", 18), justify="right", width=20
        )
        self.display_entry.pack()
        self.display_entry.focus_set()  # Set focus to the display entry for keyboard input

    def create_buttons(self):
        """Create calculator buttons."""
        self.buttons_frame = ttk.Frame(self.root)
        self.buttons_frame.pack()

        # Button definitions
        buttons = [
            ("7", 1, 0), ("8", 1, 1), ("9", 1, 2), ("/", 1, 3),
            ("4", 2, 0), ("5", 2, 1), ("6", 2, 2), ("*", 2, 3),
            ("1", 3, 0), ("2", 3, 1), ("3", 3, 2), ("-", 3, 3),
            ("0", 4, 0), (".", 4, 1), ("+", 4, 2), ("=", 4, 3),
            ("sin", 5, 0), ("cos", 5, 1), ("tan", 5, 2), ("sqrt", 5, 3),
            ("log", 6, 0), ("(", 6, 1), (")", 6, 2), ("C", 6, 3),
            ("MR", 7, 0), ("MS", 7, 1), ("MC", 7, 2), ("Mode", 7, 3),
        ]

        # Create buttons with animations
        for (text, row, col) in buttons:
            button = tk.Button(
                self.buttons_frame, text=text, font=("Arial", 12),
                command=lambda t=text: self.on_button_click(t)
            )
            button.grid(row=row, column=col, sticky="nsew")
            button.bind("<Enter>", lambda e, b=button: self.animate_button_hover(b, True))
            button.bind("<Leave>", lambda e, b=button: self.animate_button_hover(b, False))

        # Configure button grid layout
        for i in range(8):
            self.buttons_frame.rowconfigure(i, weight=1)
        for j in range(4):
            self.buttons_frame.columnconfigure(j, weight=1)

    def on_button_click(self, button_text):
        """Handle button clicks."""
        if button_text == "=":
            self.evaluate_expression()
        elif button_text == "C":
            self.clear_display()
        elif button_text in ("MR", "MS", "MC"):
            self.memory_functions(button_text)
        elif button_text == "Mode":
            self.toggle_angle_mode()
        else:
            self.append_to_expression(button_text)

    def evaluate_expression(self):
        """Evaluate the current expression."""
        try:
            # Get the expression from the display entry
            self.expression = self.display_var.get()
            # Evaluate the expression using eval with math functions allowed
            result = eval(self.expression, {"__builtins__": None}, math.__dict__)
            self.display_var.set(result)
            self.expression = str(result)
            self.flash_display("green")  # Flash green for success
        except Exception:
            self.display_var.set("Error")
            self.expression = ""
            self.flash_display("red")  # Flash red for error

    def clear_display(self):
        """Clear the display and expression."""
        self.display_var.set("")
        self.expression = ""

    def memory_functions(self, action):
        """Handle memory operations."""
        if action == "MR":
            if self.memory is not None:
                self.display_var.set(self.memory)
                self.expression = str(self.memory)
        elif action == "MS":
            self.memory = self.display_var.get()
        elif action == "MC":
            self.memory = None

    def toggle_angle_mode(self):
        """Toggle between degrees and radians."""
        self.angle_mode = 'radians' if self.angle_mode == 'degrees' else 'degrees'
        self.flash_display("blue")
        self.display_var.set(f"Mode: {self.angle_mode}")

    def append_to_expression(self, value):
        """Add value to the current expression."""
        self.expression += str(value)
        self.display_var.set(self.expression)

    def animate_button_hover(self, button, hover_in):
        """Animate button on hover."""
        if hover_in:
            button.config(bg="lightblue")
        else:
            button.config(bg="SystemButtonFace")

    def flash_display(self, color):
        """Flash the display background with the given color."""
        original_color = self.display_entry.cget("background")
        self.display_entry.config(background=color)
        self.root.after(200, lambda: self.display_entry.config(background=original_color))

    def run(self):
        """Start the Tkinter event loop."""
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    calculator = AnimatedCalculator(root)
    calculator.run()
