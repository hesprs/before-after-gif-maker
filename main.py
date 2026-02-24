from glob import glob
from PIL import Image
import argparse
import os
import shutil
import numpy as np
from natsort import natsorted
import cv2
from tqdm import tqdm


def make_gif(
    frame_folder: str, ext: str, fps: int, repeat: int, output_path: str
) -> None:
    """
    Generate GIF from a folder of frames.
    Duration is calculated based on frame count and FPS.
    """
    files = natsorted(glob(os.path.join(frame_folder, f"*.{ext}")))

    if not files:
        raise ValueError(f"No frames found in {frame_folder}")

    # Calculate duration per frame in milliseconds
    duration_ms = int(1000 / fps)

    frames = [
        Image.open(image).convert("RGB").quantize(method=Image.MEDIANCUT)
        for image in files
    ]

    frame_one = frames[0]
    frame_one.save(
        output_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duration_ms,
        loop=repeat,
    )


def make_essential_frames(
    restored_path: str,
    noise_path: str,
    ext: str,
    scale: float,
    total_frames: int,
    frame_dir: str,
) -> None:
    """
    Generate wipe transition frames based on exact frame count.
    """
    restored = cv2.imread(restored_path)
    noise = cv2.imread(noise_path)

    if restored is None or noise is None:
        raise ValueError(f"Failed to load images: {restored_path} or {noise_path}")

    if restored.shape != noise.shape:
        raise Exception(
            f"\nTwo images have to be the same resolution! {restored_path} vs {noise_path}"
        )

    h, w, c = restored.shape
    effective_width = int(w * scale)

    # Resize images to target scale
    resize_restored = cv2.resize(
        restored, (effective_width, int(h * scale)), interpolation=cv2.INTER_CUBIC
    )
    resize_noise = cv2.resize(
        noise, (effective_width, int(h * scale)), interpolation=cv2.INTER_CUBIC
    )
    r_h, r_w, r_c = resize_restored.shape

    # Generate exactly 'total_frames' frames
    # Frame 0 = 100% Noise, Frame N-1 = 100% Restored
    for i in tqdm(range(total_frames), desc="Frames", leave=False):
        # Calculate progress ratio (0.0 to 1.0)
        progress = i / (total_frames - 1) if total_frames > 1 else 1.0

        # Calculate split point
        split_x = int(r_w * progress)

        # Ensure boundaries
        split_x = max(0, min(split_x, r_w))

        # Compose frame
        left = resize_restored[0:r_h, 0:split_x]
        right = resize_noise[0:r_h, split_x:r_w]

        # Handle edge cases where left or right might be empty
        if left.size == 0:
            combine = right
        elif right.size == 0:
            combine = left
        else:
            combine = np.concatenate((left, right), axis=1)

        cv2.imwrite(os.path.join(frame_dir, f"frame_{str(i).zfill(4)}.{ext}"), combine)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame_dir",
        default="./frames",
        type=str,
        help="Base directory for temporary frames",
    )
    parser.add_argument(
        "--resize_scale",
        default=1,
        type=float,
        help="Scale factor for output resolution",
    )
    parser.add_argument(
        "--duration", default=5, type=float, help="Total GIF duration in seconds"
    )
    parser.add_argument(
        "--fps", default=24, type=int, help="Frames per second (smoothness)"
    )
    parser.add_argument(
        "--type",
        default="png",
        type=str,
        dest="ext",
        help="Image extension (png, jpg, etc.)",
    )
    parser.add_argument(
        "--repeat_GIF", default=0, type=int, help="0: repeat, 1: no repeat"
    )
    parser.add_argument("--result_dir", default="./result", type=str)
    args = parser.parse_args()

    # Directories
    dir_before = "./before/"
    dir_after = "./after/"
    base_frame_dir = args.frame_dir
    result_dir = args.result_dir
    ext = args.ext

    # Calculated Parameters
    total_frames = max(1, int(args.duration * args.fps))

    # Ensure directories exist
    os.makedirs(base_frame_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Scan for matching pairs
    before_files = natsorted(glob(os.path.join(dir_before, f"*.{ext}")))

    if not before_files:
        tqdm.write(f"No .{ext} files found in {dir_before}")
        return

    tqdm.write(
        f"Found {len(before_files)} images. Target: {total_frames} frames @ {args.fps}fps ({args.duration}s)"
    )

    for before_path in tqdm(before_files, desc="Processing Pairs"):
        filename = os.path.basename(before_path)
        after_path = os.path.join(dir_after, filename)

        # Check if matching file exists in 'after'
        if not os.path.exists(after_path):
            tqdm.write(f"Skipping {filename}: No matching file in {dir_after}")
            continue

        # Prepare paths
        name_without_ext = os.path.splitext(filename)[0]
        output_gif = os.path.join(result_dir, f"{name_without_ext}.gif")

        # Create unique temp folder for this pair
        temp_frame_dir = os.path.join(base_frame_dir, f"{name_without_ext}_frames")
        os.makedirs(temp_frame_dir, exist_ok=True)

        try:
            # 1. Generate Frames
            tqdm.write(f"Generating {total_frames} frames for {filename}...")
            make_essential_frames(
                after_path,
                before_path,
                ext,
                args.resize_scale,
                total_frames,
                temp_frame_dir,
            )

            # 2. Generate GIF
            tqdm.write(f"Encoding GIF for {filename}...")
            make_gif(temp_frame_dir, ext, args.fps, args.repeat_GIF, output_gif)

            # 3. Cleanup temp frames
            shutil.rmtree(temp_frame_dir)

        except Exception as e:
            tqdm.write(f"Error processing {filename}: {e}")
            if os.path.exists(temp_frame_dir):
                shutil.rmtree(temp_frame_dir)
            continue

    tqdm.write("Batch processing finished !!")


if __name__ == "__main__":
    main()
