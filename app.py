from scripts import gui


def main():
    import tkinter as tk

    root = tk.Tk()
    app = gui.FaceLockApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
