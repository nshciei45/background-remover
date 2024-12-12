import cv2
import numpy as np
from PIL import Image, ImageTk
from rembg import remove
import os

from tkinter import filedialog, messagebox
import tkinter as tk

from customtkinter import *

from ultralytics import YOLO
import torch

model = YOLO("yolov8n.pt")

set_appearance_mode('light')
set_default_color_theme('blue')

# ----------------------------------------------------------------------------------------------------------
# Image Processing (Bg_Removal)
def process_image(file_path):
        image = Image.open(file_path)
        image_no_bg = remove(image)

        if image_no_bg.mode != 'RGBA':
            image_no_bg = image_no_bg.convert('RGBA')

        image_no_bg_np = np.array(image_no_bg)
        r, g, b, alpha = cv2.split(image_no_bg_np)

        white_background = np.full(image_no_bg_np.shape, 255, dtype=np.uint8)

        # For smooth blending
        alpha = alpha.astype(float) / 255

        # Anti-aliasing
        result = white_background.copy()

        for c in range(3):
            result[:, :, c] = alpha * image_no_bg_np[:, :, c] + (1 - alpha) * white_background[:, :, c]

        result_rgb = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)
        input = cv2.imread(upload_file_path)
        result_rgb_transparent = remove(input)
        show_result_gui(image, Image.fromarray(result_rgb), file_path, result_rgb, result_rgb_transparent)

# ------------------------------------------------------------------------------------------------------------

def upload_image():
    file_path = filedialog.askopenfilename(title="Select Image",
                                           filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    global upload_file_path
    upload_file_path = file_path
    if file_path:
        process_image(file_path)

def display_image(image, label):
    image.thumbnail((200, 200))
    img_tk = ImageTk.PhotoImage(image)
    label.config(image=img_tk)
    label.image = img_tk

def save_image():
    if result_img_to_save:
        result_path = save_path
        result_image_nparray2 = cv2.cvtColor(result_image_nparray,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(result_path, 'result.png'), result_image_nparray2)
        cv2.imwrite(os.path.join(result_path, 'result_transparent.png'), result_image_nobg)
        messagebox.showinfo("Saved", f"Image saved at {result_path}")
    else:
        messagebox.showwarning("No Image", "No result image to save")

def add_more_images():
    result_window.destroy()
    upload_image()

def exit_program():
    root.quit()
# ----------------------------------------------------------------------------------------------------------

def save_detected_images():
    if result_img_to_save:
        results = model(save_path2)

        # to train model
        # model.train(data='coco8.yaml',epochs=3)
        # metrics = model.val()

        boxes = results[0].boxes.xyxy.tolist()
        boxes_names = results[0].boxes.cls.tolist()
        cropped_path = "C:/Users/Lenovo/PycharmProjects/bgr/cropped"
        bg_removed = "C:/Users/Lenovo/PycharmProjects/bgr/bg_removed"
        img = cv2.imread(save_path2)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            ultralytics_crop_object = img[int(y1):int(y2), int(x1):int(x2)]
            img_out = ultralytics_crop_object
            img_out_bg_removed = remove(ultralytics_crop_object) #sept10
            cv2.imwrite(os.path.join(bg_removed,str(model.names[boxes_names[i]])+str(i)+'_bg_removed.png'),img_out_bg_removed)
            cv2.imwrite(os.path.join(cropped_path, str(model.names[boxes_names[i]]) + str(i) + '.png'), img_out)
        results = model.predict(source=img,save=True,save_txt=True)
        messagebox.showinfo("Saved", f"Images saved at {cropped_path}")
    else:
        messagebox.showwarning("No Image", "No result image to save")
# ----------------------------------------------------------------------------------------------------------

# Result GUI
def show_result_gui(original_image, result_image, file_path, r_img_arr, r_img_nobg):
    global result_window
    result_window = tk.Toplevel(root)
    result_window.title("Processed Image Result")
    result_window.geometry("600x500")
    # result_window.config(bg="#f0f0f0")    # GUI window color

    # Frame to center the images
    image_frame = tk.Frame(result_window)
    image_frame.pack(pady=60)
    # image_frame.config(bg="#button_frame.config(bg="#f0f0f0")") # image_frame color

    # Labels for images
    original_img_label = tk.Label(image_frame)
    original_img_label.grid(row=0, column=0, padx=20)

    # Arrow
    arrow_label = tk.Label(image_frame, text="→", font=("Arial", 50))
    arrow_label.grid(row=0, column=1, padx=10)

    result_img_label = tk.Label(image_frame)
    result_img_label.grid(row=0, column=2, padx=20)

    # Text labels below images
    tk.Label(image_frame, text="Original Image", font=("Arial", 14)).grid(row=1, column=0, pady=(10, 20))
    tk.Label(image_frame, text="Result Image", font=("Arial", 14)).grid(row=1, column=2, pady=(10, 20))

    # Display images
    display_image(original_image, original_img_label)
    display_image(result_image, result_img_label)

    # Save image button
    save_btn = CTkButton(result_window,text='Save result image',command=save_image,font=('Arial',18),border_spacing=10,corner_radius=40)
    save_btn.place(relx=0.5, rely=0.7, anchor='center')

    # Save detected images button
    save_det_btn = save_btn = CTkButton(result_window,text='Save detected objects',command=save_detected_images,font=('Arial',18),border_spacing=10,corner_radius=40)
    save_det_btn.place(relx=0.5,rely=0.85,anchor='center')

    # Frame for "More" and "Exit"
    bottom_frame = tk.Frame(result_window)
    bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10, padx=20)

    # "More" label
    more_label = tk.Label(bottom_frame, text="More", font=("Arial", 12), fg="blue", cursor="hand2")
    more_label.pack(side=tk.LEFT, anchor='sw')

    more_label.bind("<Enter>", lambda e: more_label.config(font=("Arial", 12, "underline")))
    more_label.bind("<Leave>", lambda e: more_label.config(font=("Arial", 12)))
    more_label.bind("<Button-1>", lambda e: add_more_images())

    # "Exit" label
    exit_label = tk.Label(bottom_frame, text="Exit", font=("Arial", 12), fg="blue", cursor="hand2")
    exit_label.pack(side=tk.RIGHT, anchor='se')

    exit_label.bind("<Enter>", lambda e: exit_label.config(font=("Arial", 12, "underline")))
    exit_label.bind("<Leave>", lambda e: exit_label.config(font=("Arial", 12)))
    exit_label.bind("<Button-1>", lambda e: exit_program())

    global result_img_to_save
    result_img_to_save = result_image
    global result_image_nparray
    result_image_nparray = r_img_arr
    global result_image_nobg
    result_image_nobg = r_img_nobg
    global save_path
    save_path = 'C:/Users/Lenovo/PycharmProjects/bgr/result/'
    global save_path2
    save_path2 = file_path

# -----------------------------------------------------------------------------------------------------------
# Input GUI
# root = tk.Tk()
root=CTk()
root.title("Remove Image Background")
root.geometry("400x300")

# Title
label = CTkLabel(root,text='Remove Image Background',font=('',30))
label.place(relx=0.5,rely=0.3,anchor='center')

# Upload button
btn = CTkButton(root, text='Upload Image', corner_radius=40,command=upload_image,font=('Arial',20),border_spacing=10)
btn.place(relx=0.5,rely=0.7,anchor='center')

result_img_to_save = None
save_path = None

root.mainloop()

# ------------------------------------------------------------------------------------------------------
