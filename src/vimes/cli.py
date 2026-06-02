import argparse
from pathlib import Path

from .animate import PygameAnimator
from .preprocess import preprocess_to_frames
from .temp_to_color import add_temperatures_and_rgb

BASE_DIR = Path.cwd()
DEFAULT_HDF5_PATH = BASE_DIR / "BSE_Detailed_Output_0.h5"
DEFAULT_FRAMES_PATH = BASE_DIR / "frames_data.npz"


def parse_preprocessing_arguments():
    parser = argparse.ArgumentParser(description="Parse preprocessing settings.")

    parser.add_argument(
        "hdf5",
        default=DEFAULT_HDF5_PATH,
        type=Path,
        help="Path to the input HDF5 file.",
    )

    parser.add_argument(
        "out",
        default=DEFAULT_FRAMES_PATH,
        type=Path,
        help="Path to the output frames file.",
    )

    return parser.parse_args()


def parse_animation_arguments():
    parser = argparse.ArgumentParser(description="Parse scaling and image settings.")

    parser.add_argument(
        "frames",
        default=DEFAULT_FRAMES_PATH,
        type=Path,
        help="Path to the input frames file.",
    )

    parser.add_argument(
        "--scaling",
        choices=["log", "linear"],
        default="linear",
        help="The type of scaling to apply (log or linear).",
    )

    parser.add_argument(
        "--images",
        choices=["tulips", "default"],
        default="default",
        help="The set of images to use (tulips or default).",
    )

    parser.add_argument(
        "--save-mp4", type=str, default=None, help="Save animation to MP4 file"
    )

    parser.add_argument(
        "--no-display", action="store_true", help="Run headless (do not open a window)"
    )

    return parser.parse_args()


def preprocess():
    args = parse_preprocessing_arguments()

    if not args.hdf5.exists():
        raise FileNotFoundError(f"{args.hdf5} not found.")

    preprocess_to_frames(args.hdf5, args.out)
    add_temperatures_and_rgb(args.hdf5, args.out, args.out)


def animate():
    args = parse_animation_arguments()

    if not args.frames.exists():
        raise FileNotFoundError(
            f"{args.frames} not found. Run compas_preprocess.py first."
        )

    print(f"scaling {args.scaling}, images {args.images}")
    animator = PygameAnimator(
        args.frames,
        save_mp4=args.save_mp4,
        no_display=args.no_display,
        use_log_scaling=args.scaling == "log",
        use_tulips_color=args.images == "tulips",
    )
    animator.run()
