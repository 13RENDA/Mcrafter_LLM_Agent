import os
import cv2
from PIL import Image
import numpy as np
import argparse
import math
from PIL import ImageDraw, ImageFont

# Parse command line arguments
parser = argparse.ArgumentParser(description='Combine agent step images into videos for multiple agents.')
parser.add_argument('--results_dir', type=str, default='results', help='Path to results folder containing agent_x subfolders')
parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output videos')
parser.add_argument('--fps', type=int, default=5, help='Frames per second for the video')
parser.add_argument('--agents', type=str, default=None, help='Comma-separated list of agent ids to process (e.g., 0,2,3). If not set, process all agent folders.')
parser.add_argument('--multi_panel', action='store_true', help='If set, create a multi-agent panel video instead of individual videos')
parser.add_argument('--output', type=str, default='multi_agent_panel.mp4', help='Output filename for multi-panel video (only used if --multi_panel)')
args = parser.parse_args()

# Find agent folders
def get_agent_folders(results_dir, agent_ids=None):
    folders = []
    for name in os.listdir(results_dir):
        if name.startswith('agent_') and os.path.isdir(os.path.join(results_dir, name)):
            agent_id = name.split('_')[-1]
            if agent_ids is None or agent_id in agent_ids:
                folders.append((agent_id, os.path.join(results_dir, name)))
    return sorted(folders, key=lambda x: int(x[0]))

# Parse agent ids if provided
agent_ids = None
if args.agents:
    agent_ids = set(args.agents.split(','))

agent_folders = get_agent_folders(args.results_dir, agent_ids)
if not agent_folders:
    raise RuntimeError(f'No agent folders found in {args.results_dir}')

os.makedirs(args.output_dir, exist_ok=True)

def make_video_for_agent(agent_id, agent_dir, output_path, fps):
    image_files = [f for f in os.listdir(agent_dir) if f.endswith('.png') and not f.endswith('_stats.png')]
    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))
    if not image_files:
        print(f'No step images found in {agent_dir}, skipping.')
        return
    first_img = Image.open(os.path.join(agent_dir, image_files[0]))
    frame_size = first_img.size  # (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for fname in image_files:
        img_path = os.path.join(agent_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        if (img_cv.shape[1], img_cv.shape[0]) != frame_size:
            img_cv = cv2.resize(img_cv, frame_size)
        video_writer.write(img_cv)
    video_writer.release()
    print(f'Agent {agent_id}: Video saved as {output_path}')

def add_label_to_image(img, label, font=None, label_height=20):
    w, h = img.size
    new_img = Image.new('RGB', (w, h + label_height), color=(0,0,0))
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    if font is None:
        font = ImageFont.load_default()
    # Use textbbox for accurate text size
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = (w - text_w) // 2
    text_y = h + (label_height - text_h) // 2
    draw.text((text_x, text_y), label, fill=(255,255,255), font=font)
    return new_img

def make_multi_panel_video(agent_folders, output_path, fps):
    # Collect sorted image lists for each agent
    all_image_lists = []
    agent_labels = []
    for agent_id, agent_dir in agent_folders:
        image_files = [f for f in os.listdir(agent_dir) if f.endswith('.png') and not f.endswith('_stats.png')]
        image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))
        if not image_files:
            print(f'No step images found in {agent_dir}, skipping agent {agent_id}.')
            continue
        all_image_lists.append([os.path.join(agent_dir, f) for f in image_files])
        agent_labels.append(f'Agent {agent_id}')
    if not all_image_lists:
        print('No agent images found for multi-panel video.')
        return
    # Use the minimum number of frames across all agents
    min_frames = min(len(lst) for lst in all_image_lists)
    n_agents = len(all_image_lists)
    # Determine grid size (try to make it as square as possible)
    grid_cols = math.ceil(math.sqrt(n_agents))
    grid_rows = math.ceil(n_agents / grid_cols)
    # Get frame size from first image
    first_img = Image.open(all_image_lists[0][0])
    frame_w, frame_h = first_img.size
    label_height = 20
    panel_w = frame_w * grid_cols
    panel_h = (frame_h + label_height) * grid_rows
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (panel_w, panel_h))
    font = ImageFont.load_default()
    for i in range(min_frames):
        panel = Image.new('RGB', (panel_w, panel_h), color=(0,0,0))
        for idx, img_list in enumerate(all_image_lists):
            row = idx // grid_cols
            col = idx % grid_cols
            img = Image.open(img_list[i]).convert('RGB')
            img_labeled = add_label_to_image(img, agent_labels[idx], font=font, label_height=label_height)
            panel.paste(img_labeled, (col * frame_w, row * (frame_h + label_height)))
        panel_np = np.array(panel)
        panel_cv = cv2.cvtColor(panel_np, cv2.COLOR_RGB2BGR)
        video_writer.write(panel_cv)
    video_writer.release()
    print(f'Multi-agent panel video saved as {output_path}')

if args.multi_panel:
    make_multi_panel_video(agent_folders, args.output, args.fps)
else:
    for agent_id, agent_dir in agent_folders:
        output_path = os.path.join(args.output_dir, f'agent_{agent_id}.mp4')
        make_video_for_agent(agent_id, agent_dir, output_path, args.fps) 