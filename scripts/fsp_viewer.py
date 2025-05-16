#!/usr/bin/env python3
"""
GUI Viewer for FasterLivePortrait .fsp source data files.
Load an FSP file, navigate frames and faces, overlay landmarks and display all data fields.
Also provides 3D keypoint and appearance feature visualization.
"""
import argparse
import os
import pickle
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FSPViewer(tk.Tk):
    def __init__(self, filepath=None):
        super().__init__()
        self.title("FSP Viewer")
        self.geometry("1200x700")
        # Data containers
        self.data = None
        self.photo = None
        self.current_frame = 0
        self.current_face = 0
        # UI setup
        self._build_ui()
        if filepath:
            ok = self.load_fsp(filepath)
            if not ok:
                print(f"Failed to load '{filepath}'")
                self.open_file()
        else:
            # No initial file, prompt user to open one
            self.open_file()

    def _build_ui(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open...", command=self.open_file)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

        self.canvas = tk.Canvas(self, bg='black')
        self.canvas.pack(side='left', fill='both', expand=True)

        ctrl = tk.Frame(self)
        ctrl.pack(side='right', fill='y')

        tk.Label(ctrl, text="Frame:").pack(pady=4)
        self.frame_var = tk.IntVar()
        self.frame_spin = tk.Spinbox(ctrl, from_=0, to=0, textvariable=self.frame_var,
                                     command=self.on_frame_change, width=5)
        self.frame_spin.pack()

        tk.Label(ctrl, text="Face:").pack(pady=4)
        self.face_var = tk.IntVar()
        self.face_spin = tk.Spinbox(ctrl, from_=0, to=0, textvariable=self.face_var,
                                    command=self.on_face_change, width=5)
        self.face_spin.pack()

        tk.Label(ctrl, text="Data:").pack(pady=4)
        self.text = tk.Text(ctrl, width=40, height=30)
        self.text.pack(fill='y', expand=True)

        tk.Button(ctrl, text="Refresh", command=self.refresh).pack(pady=4)
        tk.Button(ctrl, text="3D Keypoints", command=self.show_3d_keypoints).pack(pady=4)
        tk.Button(ctrl, text="Appearance", command=self.show_appearance).pack(pady=4)

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[('FSP files','*.fsp'),('Pickle','*.pkl')])
        if path:
            self.load_fsp(path)

    def load_fsp(self, path):
        if not os.path.exists(path):
            fallback = os.path.basename(path)
            if os.path.exists(fallback):
                print(f"Using basename '{fallback}' instead of '{path}'")
                path = fallback
            else:
                messagebox.showerror("Error", f"File not found: {path}")
                return False
        try:
            with open(path, 'rb') as f:
                self.data = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            return False

        n_frames = len(self.data.get('src_imgs', []))
        self.frame_spin.config(to=max(0, n_frames-1))
        self.frame_var.set(0)
        self.on_frame_change()
        return True

    def on_frame_change(self):
        if self.data is None:
            return
        self.current_frame = self.frame_var.get()
        faces = self.data['src_infos'][self.current_frame]
        n_faces = len(faces)
        self.face_spin.config(to=max(0, n_faces-1))
        self.face_var.set(0)
        self.on_face_change()

    def on_face_change(self):
        if self.data is None:
            return
        self.current_face = self.face_var.get()
        self.refresh()

    def refresh(self):
        if self.data is None:
            return
        img = self.data['src_imgs'][self.current_frame]
        h, w = img.shape[:2]
        self.photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.delete('all')
        self.canvas.config(scrollregion=(0,0,w,h))
        self.canvas.create_image(0,0, anchor='nw', image=self.photo)

        face_info = self.data['src_infos'][self.current_frame][self.current_face]
        lmk = face_info[1]
        for (x,y) in lmk:
            self.canvas.create_oval(x-3,y-3,x+3,y+3, outline='yellow')

        self.text.delete('1.0', tk.END)
        keys = ['x_s_info','source_lmk','R_s','f_s','x_s','x_c_s',
                'lip_delta_before_animation','flag_lip_zero','mask_ori_float','M']
        labels = ['MotionInfo','Source_LMK','Rotation','Feat','TransKP','CanonKP',
                  'LipDelta','LipZero','Mask','Transform']
        for key,label in zip(keys, labels):
            val = face_info[self._field_index(key)]
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            if val is None:
                self.text.insert(tk.END, f"{label}: None\n\n")
            else:
                self.text.insert(tk.END, f"{label}:\n{np.array(val)}\n\n")

    def _field_index(self, key):
        mapping = {
            'x_s_info': 0,'source_lmk': 1,'R_s': 2,'f_s': 3,'x_s': 4,
            'x_c_s': 5,'lip_delta_before_animation': 6,'flag_lip_zero': 7,
            'mask_ori_float': 8,'M': 9
        }
        return mapping.get(key, 0)

    def show_3d_keypoints(self):
        if self.data is None:
            return
        face_info = self.data['src_infos'][self.current_frame][self.current_face]
        x_s = face_info[self._field_index('x_s')]
        if isinstance(x_s, torch.Tensor):
            x_s = x_s.cpu().numpy()
        pts = np.array(x_s).reshape(-1, 3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')  # type: ignore
        plt.title('3D Keypoints')
        plt.show()

    def show_appearance(self):
        if self.data is None:
            return
        face_info = self.data['src_infos'][self.current_frame][self.current_face]
        f_s = face_info[self._field_index('f_s')]
        if isinstance(f_s, torch.Tensor):
            f_s = f_s.cpu().numpy()
        # Convert to numpy and remove singleton dims
        arr = np.array(f_s)
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            plt.plot(arr); plt.title('Appearance Feature Vector')
        elif arr.ndim == 2:
            plt.imshow(arr, aspect='auto'); plt.title('Appearance Features'); plt.colorbar()
        elif arr.ndim == 3:
            plt.imshow(arr[0], cmap='viridis'); plt.title('Appearance First Channel'); plt.colorbar()
        else:
            # For higher-dim data, visualize first 2D slice
            plt.imshow(arr.reshape(arr.shape[0], -1), aspect='auto', cmap='viridis')
            plt.title(f'Appearance first slice reshaped {arr.shape}')
            plt.colorbar()
            plt.show()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', help='Path to .fsp file')
    args = parser.parse_args()
    path = args.file if args.file and os.path.exists(args.file) else None
    app = FSPViewer(filepath=path)
    app.mainloop()