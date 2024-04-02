# main.py

import tkinter as tk
from gui.UIManager import UIManager

def main():
    root = tk.Tk()
    ui_manager = UIManager(root)
    root.mainloop()

if __name__ == "__main__":
    main()
