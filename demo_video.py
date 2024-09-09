import os 
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

import argparse
import random
import numpy as np
from PIL import Image, ImageOps
import torch
from tqdm import tqdm
import math
import cv2

from utils import render_meshes, demo_color as color
from demo import normalize_rgb, load_model, forward_model, get_camera_parameters

torch.cuda.empty_cache()

np.random.seed(seed=0)
random.seed(0)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run MultiHMR demo on a video.")
parser.add_argument('--video_path', type=str, default="./demo/df1.mp4", help="Path to the input video.")
args = parser.parse_args()

video_path = args.video_path
input_video_name = os.path.splitext(os.path.basename(video_path))[0]

def process_pil_image(img_pil, img_size, device=torch.device('cuda')):
    """ Open image at path, resize and pad """

    # Open and reshape
    img_pil = ImageOps.contain(img_pil, (img_size,img_size)) # keep the same aspect ratio

    # Keep a copy for visualisations.
    img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size,img_size), color=(255, 255, 255))
    img_pil = ImageOps.pad(img_pil, size=(img_size,img_size)) # pad with zero on the smallest side

    # Go to numpy 
    resize_img = np.asarray(img_pil)

    # Normalize and go to torch.
    resize_img = normalize_rgb(resize_img)
    x = torch.from_numpy(resize_img).unsqueeze(0).to(device)
    return x, img_pil_bis

def overlay_human_meshes(humans, K, model, img_pil, unique_color=False):

    # Color of humans seen in the image.
    # _color = [color[0] for _ in range(len(humans))] if unique_color else color
    _color = [(128/255, 128/255, 192/255) for _ in range(len(humans))]  # Adjust RGB as needed
    
    # Get focal and princpt for rendering.
    focal = np.asarray([K[0,0,0].cpu().numpy(),K[0,1,1].cpu().numpy()])
    princpt = np.asarray([K[0,0,-1].cpu().numpy(),K[0,1,-1].cpu().numpy()])

    # Get the vertices produced by the model.
    verts_list = humans
    faces_list = [model.smpl_layer['neutral_10'].bm_x.faces for j in range(len(humans))]

    # Render the meshes onto the image.
    pred_rend_array = render_meshes(np.asarray(img_pil), 
            verts_list,
            faces_list,
            {'focal': focal, 'princpt': princpt},
            alpha=1.0,
            color=_color)

    return pred_rend_array, _color

def interpolate_frames(start_verts, end_verts, n):
    step = (end_verts - start_verts) / (n+1)
    return [start_verts + step*(i+1) for i in range(n)]

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuro:

    def __init__(self,
                 t0,
                 x0,
                 dx0=0.0,
                 min_cutoff=1.0,
                 beta=0.0,
                 d_cutoff=1.0):
        super(OneEuro, self).__init__()
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)  # [k, c]
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)
        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class OneEuroFilter:
    """Oneeuro filter, source code: https://github.com/mkocabas/VIBE/blob/c0
    c3f77d587351c806e901221a9dc05d1ffade4b/lib/utils/smooth_pose.py.

    Args:
        min_cutoff (float, optional):
        Decreasing the minimum cutoff frequency decreases slow speed jitter
        beta (float, optional):
        Increasing the speed coefficient(beta) decreases speed lag.

    Returns:
        np.ndarray: smoothed poses
    """

    def __init__(self, min_cutoff=0.004, beta=0.7):
        super(OneEuroFilter, self).__init__()

        self.min_cutoff = min_cutoff
        self.beta = beta
        
        self.filter = None
        
    def __call__(self, idx, x):
        # x: (n_verts, 3)
        assert len(x.shape) == 2
        pose = x
        if self.filter is None and idx==0:
            self.filter = OneEuro(
                np.zeros_like(x),
                x,
                min_cutoff=self.min_cutoff,
                beta=self.beta,
            )
        else:
            t = np.ones_like(pose) * idx
            pose = self.filter(t, pose)
        
        return pose

model_name = 'multiHMR_896_L'
fov = 60
det_thresh = 0.3
nms_kernel_size = 3
model = load_model(model_name)

db = []
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Unable to open video file")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

i = 0
with tqdm(total=total_frames, desc="MultiHMR running") as pbar:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        x, img_pil_nopad = process_pil_image(pil_image, model.img_size)

        p_x, p_y = None, None
        K = get_camera_parameters(model.img_size, fov=fov, p_x=p_x, p_y=p_y)

        humans = forward_model(
            model, x, K,
            det_thresh=det_thresh,
            nms_kernel_size=nms_kernel_size)
        if len(humans) > 0:
            db.append((i, humans[0]['v3d'].cpu().numpy()))
            
        i +=1
        pbar.update(1)

interpolated = []
prev_index, prev_mesh = float("inf"), None
for i, (index, mesh) in enumerate(db):
    if i > 0 and index - prev_index > 1:
        _interpolated = interpolate_frames(
            prev_mesh, mesh, index-prev_index-1)
        interpolated += [
            (idx, verts) for idx, verts in zip(
                range(prev_index+1, index), _interpolated)]
        print(f"{index-prev_index-1} frames interpolated")
    prev_index, prev_mesh = index, mesh
db += interpolated
db = sorted(db, key=lambda x:x[0])

smooth_filter = OneEuroFilter(min_cutoff=0.004, beta=1.5)

db_smooth = []
for i, (index, mesh) in enumerate(db):
    db_smooth.append((index, smooth_filter(i, mesh)))
db_smooth_map = {k:v for k, v in db_smooth}

os.makedirs("./output/figures", exist_ok=True)
os.makedirs("./output/mesh", exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Unable to open video file")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

i = 0
with tqdm(total=total_frames, desc="Rendering") as pbar:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        x, img_pil_nopad = process_pil_image(pil_image, model.img_size)
        
        if i in db_smooth_map.keys():
            mesh = db_smooth_map[i]
            K = get_camera_parameters(model.img_size, fov=fov, p_x=None, p_y=None)
            img_overlay, _color = overlay_human_meshes([mesh], K, model, img_pil_nopad)
            img_overlay = Image.fromarray(img_overlay)
        else:
            img_overlay = img_pil_nopad
            
        img_overlay.save(
            f"./output/figures/{str(i).zfill(8)}.png"
        )
        
        i += 1
        pbar.update(1)

output_video_name = f"{input_video_name}_output"
# os.system('cd ./output/figures && ffmpeg -framerate 30 -i %08d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p ../{output_video_name}')
os.system(f'cd ./output/figures && ffmpeg -framerate 30 -i %08d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p ../{output_video_name}.mp4')


# Delete all PNG files after creating the video
folder = './output/figures'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if filename.endswith(".png"):
            os.remove(file_path)
    except Exception as e:
        print(f'Error deleting {filename}: {e}')
