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
    frame_folder: str, ext: str, total_time: float, repeat: int, output_path: str
) -> None:
    files = natsorted(glob(os.path.join(frame_folder, f"*.{ext}")))
    if not files:
        raise ValueError(f"No frames found in {frame_folder}")
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
        duration=(total_time / len(frames)) * 1000,
        loop=repeat,
    )


def make_essential_frames(
    restored_path: str,
    noise_path: str,
    ext: str,
    scale: float,
    step: int,
    frame_dir: str,
) -> None:
    """Generate wipe transition frames between two images."""
    restored = cv2.imread(restored_path)
    noise = cv2.imread(noise_path)

    if restored is None or noise is None:
        raise ValueError(f"Failed to load images: {restored_path} or {noise_path}")

    if restored.shape != noise.shape:
        raise Exception(
            f"\nTwo images have to be the same resolution! {restored_path} vs {noise_path}"
        )

    h, w, c = restored.shape
    resize_restored = cv2.resize(
        restored, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )
    resize_noise = cv2.resize(
        noise, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )
    r_h, r_w, r_c = resize_restored.shape

    start_frame = resize_noise
    cv2.imwrite(os.path.join(frame_dir, f"frame_0000.{ext}"), start_frame)

    data_file = 1
    # Ensure range stops correctly
    limit = (r_w // step) * step
    for i in tqdm(range(0, limit, step), desc="Frames", leave=False):
        left = resize_restored[0:r_h, 0 : (step + i)]
        right = resize_noise[0:r_h, (step + i) : r_w]
        combine = np.concatenate((left, right), axis=1)
        cv2.imwrite(
            os.path.join(frame_dir, f"frame_{str(data_file).zfill(4)}.{ext}"), combine
        )
        data_file += 1

    cv2.imwrite(os.path.join(frame_dir, f"frame_last.{ext}"), resize_restored)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame_dir",
        default="./frames",
        type=str,
        help="Base directory for temporary frames",
    )
    parser.add_argument("--resize_scale", default=1, type=float)
    parser.add_argument(
        "--step",
        default=10,
        type=int,
        help="the number of pixels progressed each frame",
    )
    parser.add_argument(
        "--type",
        default="png",
        type=str,
        dest="ext",
        help="Image extension (png, jpg, etc.)",
    )
    parser.add_argument(
        "--period",
        default=3,
        type=float,
        help="gif duration in seconds (GIF players seem to ignore this)",
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

    # Ensure directories exist
    os.makedirs(base_frame_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Scan for matching pairs
    # We scan 'before' and check if 'after' has the same filename
    before_files = natsorted(glob(os.path.join(dir_before, f"*.{ext}")))

    if not before_files:
        tqdm.write(f"No .{ext} files found in {dir_before}")
        return

    tqdm.write(
        f"Found {len(before_files)} images in {dir_before}. Starting batch processing..."
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

        # Create unique temp folder for this pair to avoid frame conflicts
        temp_frame_dir = os.path.join(base_frame_dir, f"{name_without_ext}_frames")
        os.makedirs(temp_frame_dir, exist_ok=True)

        try:
            # 1. Generate Frames
            # tqdm write avoids breaking the progress bar
            tqdm.write(f"Generating frames for {filename}...")
            make_essential_frames(
                after_path,
                before_path,
                ext,
                args.resize_scale,
                args.step,
                temp_frame_dir,
            )

            # 2. Generate GIF
            tqdm.write(f"Encoding GIF for {filename}...")
            make_gif(temp_frame_dir, ext, args.period, args.repeat_GIF, output_gif)

            # 3. Cleanup temp frames
            shutil.rmtree(temp_frame_dir)

        except Exception as e:
            tqdm.write(f"Error processing {filename}: {e}")
            # Ensure cleanup even on error
            if os.path.exists(temp_frame_dir):
                shutil.rmtree(temp_frame_dir)
            continue

    tqdm.write("Batch processing finished !!")


if __name__ == "__main__":
    main()
