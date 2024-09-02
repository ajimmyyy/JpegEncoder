import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import Model.tools as tools

class ImageCompressorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compressor")
        self.root.geometry("1200x500")

        # 主框架
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        # 左側框架
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.Y)

        # 右側框架
        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        self.original_frame = tk.Frame(self.right_frame)
        self.original_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.compressed_frame = tk.Frame(self.right_frame)
        self.compressed_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # 添加選擇檔案按鈕
        self.select_button = tk.Button(self.left_frame, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        # 顯示選擇的圖片路徑
        self.image_label = tk.Label(self.left_frame, text="No image selected")
        self.image_label.pack(pady=10)

        # 添加儲存路徑按鈕
        self.save_button = tk.Button(self.left_frame, text="Select Save Location", command=self.select_save_location)
        self.save_button.pack(pady=10)

        # 顯示儲存路徑
        self.save_label = tk.Label(self.left_frame, text="No save location selected")
        self.save_label.pack(pady=10)

        # 添加壓縮按鈕
        self.compress_button = tk.Button(self.left_frame, text="Compress and Save", command=self.compress_image)
        self.compress_button.pack(pady=20)

        # 原始圖片預覽
        self.preview_label_original = tk.Label(self.original_frame, text="Original Image")
        self.preview_label_original.pack(pady=10)

        self.image_display_original = tk.Label(self.original_frame)
        self.image_display_original.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # 壓縮圖片預覽
        self.preview_label_compressed = tk.Label(self.compressed_frame, text="Compressed Image")
        self.preview_label_compressed.pack(pady=10)

        self.image_display_compressed = tk.Label(self.compressed_frame)
        self.image_display_compressed.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # 顯示原始圖片大小
        self.original_size_label = tk.Label(self.left_frame, text="Original Image Size: N/A")
        self.original_size_label.pack(pady=10)

        # 顯示JPG檔案大小
        self.compressed_size_label = tk.Label(self.left_frame, text="JPEG Image Size: N/A")
        self.compressed_size_label.pack(pady=10)

        # 顯示壓縮率
        self.ratio_label = tk.Label(self.left_frame, text="Compression Ratio: N/A")
        self.ratio_label.pack(pady=10)

        # 初始化圖片路徑和儲存路徑
        self.image_path = None
        self.save_path = None

    # 選擇圖片
    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp;*.png")])
        if not self.image_path:
            messagebox.showwarning("Warning", "No image selected!")
            self.image_label.config(text="No image selected")
            self.image_display_original.config(image=None)
            self.image_display_compressed.config(image=None)
        else:
            self.image_label.config(text=f"Selected image: {self.image_path}")
            self.show_image_preview_original()

    # 顯示原圖預覽
    def show_image_preview_original(self):
        try:
            image = Image.open(self.image_path)
            image.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.image_display_original.config(image=photo)
            self.image_display_original.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    # 顯示壓縮圖片預覽
    def show_image_preview_compressed(self):
        try:
            compressed_image = Image.open(self.save_path)
            compressed_image.thumbnail((300, 300))
            photo_compressed = ImageTk.PhotoImage(compressed_image)
            self.image_display_compressed.config(image=photo_compressed)
            self.image_display_compressed.image = photo_compressed
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load compressed image: {e}")

    # 選擇儲存路徑
    def select_save_location(self):
        if self.image_path:
            default_filename = os.path.splitext(os.path.basename(self.image_path))[0] + ".jpg"
            self.save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                        filetypes=[("JPEG files", "*.jpg")],
                                                        initialfile=default_filename)
            if not self.save_path:
                messagebox.showwarning("Warning", "No save location selected!")
                self.save_label.config(text="No save location selected")
            else:
                self.save_label.config(text=f"Save location: {self.save_path}")
        else:
            messagebox.showwarning("Warning", "No image selected!")

    # 壓縮圖片
    def compress_image(self):
        if self.image_path and self.save_path:
            try:
                original_size, compressed_size = tools.CompressionImg(self.image_path, self.save_path)
                if original_size and compressed_size:
                    compression_ratio = (compressed_size / original_size) * 100
                    self.original_size_label.config(text=f"Original Image Size: {original_size:.2f} KB")
                    self.compressed_size_label.config(text=f"JPEG Image Size: {compressed_size:.2f} KB")
                    self.ratio_label.config(text=f"Compression Ratio: {compression_ratio:.2f}%")
                    self.show_image_preview_compressed()
                else:
                    self.ratio_label.config(text="Compression Ratio: Calculation error")
                messagebox.showinfo("Info", f"Image compressed and saved to {self.save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to compress image: {e}")
        else:
            messagebox.showwarning("Warning", "No image selected or save location not set!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCompressorApp(root)
    root.mainloop()