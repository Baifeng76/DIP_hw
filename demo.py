import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2

class MDIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MDI Image Viewer")

        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open", command=self.open_image)
        self.file_menu.add_command(label="Convert to BMP", command=self.convert_to_bmp)
        self.file_menu.add_command(label="Exit", command=root.quit)

        self.mdi_area = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.mdi_area.pack(fill=tk.BOTH, expand=True)

        self.cur_image = None

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.bmp *.jpg *.png")])
        if file_path:
            image = Image.open(file_path)
            # image = cv2.imread(file_path)
            self.cur_image = image
            image_window = tk.Toplevel(self.mdi_area)
            image_window.title(file_path)
            image_window.geometry(f"{image.width}x{image.height}")

            self.cur_image_window = image_window

            photo = ImageTk.PhotoImage(image)
            label = tk.Label(image_window, image=photo)
            label.image = photo  # Keep a reference!
            label.pack()

    def convert_to_bmp(self):
        if self.cur_image:
            bmp_path = filedialog.asksaveasfilename(defaultextension=".bmp", filetypes=[("BMP File", "*.bmp")])
            if bmp_path:
                self.cur_image.save(bmp_path)

    def update_image_window(self, new_image):
        self.current_label.pack_forget()
        self.current_photo = ImageTk.PhotoImage(new_image)
        self.current_label = tk.Label(self.current_image_window, image=self.current_photo)
        self.current_label.image = self.current_photo
        self.current_label.pack()
        


if __name__ == "__main__":
    root = tk.Tk()
    app = MDIApp(root)
    root.mainloop()
