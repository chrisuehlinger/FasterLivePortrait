#!/usr/bin/env python3
"""
Simple GUI editor for FasterLivePortrait .fsp files.
Load an FSP file, visualize and edit source landmarks of the first frame and first face,
and save the updated file.
"""
import argparse
import pickle
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

def main():
    parser = argparse.ArgumentParser(description="Edit landmarks in an FSP source file")
    parser.add_argument('--input', '-i', required=True, help='Input .fsp file')
    parser.add_argument('--output', '-o', required=True, help='Output .fsp file')
    args = parser.parse_args()

    # Load data
    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    # Get image and landmarks (first frame, first face)
    img_np = data['src_imgs'][0]
    face_info = data['src_infos'][0][0]
    lmk = face_info[1]  # numpy array shape (N,2)

    # Launch editor
    root = tk.Tk()
    root.title(f"Landmark Editor - {args.input}")

    h, w = img_np.shape[:2]
    pil_img = Image.fromarray(img_np)
    tk_img = ImageTk.PhotoImage(pil_img)

    canvas = tk.Canvas(root, width=w, height=h)
    canvas.pack()
    canvas_img = canvas.create_image(0, 0, anchor='nw', image=tk_img)

    points = []  # store circle ids
    radius = 4
    # Draw landmarks
    for (x, y) in lmk:
        cid = canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                                 fill='red', outline='white')
        points.append(cid)

    selected = {'idx': None}

    def on_click(event):
        # find nearest landmark
        px, py = event.x, event.y
        best, dist = None, 1e9
        for idx, (x,y) in enumerate(lmk):
            d = (x-px)**2 + (y-py)**2
            if d < dist and d < 100:  # threshold 10px
                best, dist = idx, d
        selected['idx'] = best

    def on_drag(event):
        idx = selected['idx']
        if idx is None:
            return
        x, y = event.x, event.y
        # clamp
        x = max(0, min(w-1, x)); y = max(0, min(h-1, y))
        lmk[idx] = [x, y]
        # move circle
        cid = points[idx]
        canvas.coords(cid, x-radius, y-radius, x+radius, y+radius)

    def on_release(event):
        selected['idx'] = None

    canvas.bind('<Button-1>', on_click)
    canvas.bind('<B1-Motion>', on_drag)
    canvas.bind('<ButtonRelease-1>', on_release)

    btn_frame = tk.Frame(root)
    btn_frame.pack(fill='x', pady=4)
    
    def save_and_exit():
        face_info[1] = lmk
        with open(args.output, 'wb') as fo:
            pickle.dump(data, fo)
        messagebox.showinfo("Saved", f"Saved updated FSP to {args.output}")
        root.destroy()

    tk.Button(btn_frame, text='Save', command=save_and_exit).pack(side='left', padx=8)
    tk.Button(btn_frame, text='Quit', command=root.destroy).pack(side='right', padx=8)

    root.mainloop()

if __name__ == '__main__':
    main()
