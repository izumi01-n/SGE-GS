# paper_tools/paper_render.py

import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

import cv2
import imageio
import numpy as np

import torch
import torchvision

from tqdm import tqdm
from os import makedirs
from argparse import ArgumentParser

from scene import Scene
from gaussian_renderer import render
from gaussian_renderer import GaussianModel

from arguments import (
    ModelParams,
    PipelineParams,
    get_combined_args
)

from utils.general_utils import safe_state



# Depth Visualization
def normalize_depth(depth):

    depth = depth.astype(np.float32)

    valid = depth[depth > 0]

    if len(valid) == 0:
        valid = depth.reshape(-1)

    d_min = np.percentile(valid, 1)
    d_max = np.percentile(valid, 99)

    depth = np.clip(depth, d_min, d_max)

    depth = (depth - d_min) / (d_max - d_min + 1e-8)

    return depth


def colorize_depth(depth):

    if depth.ndim == 3:
        depth = depth[0]

    depth_norm = normalize_depth(depth)

    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    depth_color = cv2.applyColorMap(
        depth_uint8,
        cv2.COLORMAP_TURBO
    )

    depth_color = cv2.cvtColor(
        depth_color,
        cv2.COLOR_BGR2RGB
    )

    return depth_color


# Error Map
def compute_error_map(render, gt):

    error = np.abs(
        render.astype(np.float32)
        -
        gt.astype(np.float32)
    )

    error = np.mean(error, axis=2)

    error = error / (error.max() + 1e-8)

    error = np.sqrt(error)

    error_vis = (error * 255).astype(np.uint8)

    return error_vis



# Render Set
def render_set(
    model_path,
    split_name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    kernel_size,
    scale_factor
):

    base_path = os.path.join(
        model_path,
        split_name,
        f"paper_{iteration}"
    )

    render_path = os.path.join(base_path, "renders")
    gt_path = os.path.join(base_path, "gt")
    depth_path = os.path.join(base_path, "depth")
    normal_path = os.path.join(base_path, "normal")
    error_path = os.path.join(base_path, "error")

    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)

    for idx, view in enumerate(
        tqdm(views, desc=f"Rendering {split_name}")
    ):

        # Render
        outputs = render(
            view,
            gaussians,
            pipeline,
            background,
            kernel_size=kernel_size,
            return_depth=True,
            return_normal=True
        )

        rendering = outputs["render"]
        render_depth = outputs["render_depth"]
        render_normal = outputs["render_normal"]

        gt = view.original_image[0:3, :, :]

        # RGB
        torchvision.utils.save_image(
            rendering,
            os.path.join(
                render_path,
                f"{idx:05d}.png"
            )
        )

        torchvision.utils.save_image(
            gt,
            os.path.join(
                gt_path,
                f"{idx:05d}.png"
            )
        )

        # Depth
        depth_np = (
            render_depth
            .detach()
            .cpu()
            .numpy()
        )

        depth_color = colorize_depth(depth_np)

        imageio.imwrite(
            os.path.join(
                depth_path,
                f"{idx:05d}.png"
            ),
            depth_color
        )

        # Normal

        normal_vis = (
            (render_normal + 1.0)
            / 2.0
        )

        normal_vis = (
            normal_vis
            .detach()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        normal_vis = (
            normal_vis * 255
        ).astype(np.uint8)

        imageio.imwrite(
            os.path.join(
                normal_path,
                f"{idx:05d}.png"
            ),
            normal_vis
        )

        # Error Map

        render_np = (
            rendering
            .detach()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        gt_np = (
            gt
            .detach()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        error_vis = compute_error_map(
            render_np,
            gt_np
        )

        imageio.imwrite(
            os.path.join(
                error_path,
                f"{idx:05d}.png"
            ),
            error_vis
        )


# Render All
def render_sets(
    dataset,
    iteration,
    pipeline,
    skip_train,
    skip_test
):

    with torch.no_grad():

        gaussians = GaussianModel(
            dataset.sh_degree
        )

        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False
        )

        scale_factor = dataset.resolution

        bg_color = (
            [1,1,1]
            if dataset.white_background
            else [0,0,0]
        )

        background = torch.tensor(
            bg_color,
            dtype=torch.float32,
            device="cuda"
        )

        kernel_size = dataset.kernel_size

        if not skip_train:

            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                kernel_size,
                scale_factor
            )

        if not skip_test:

            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                kernel_size,
                scale_factor
            )


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Paper Visualization Renderer"
    )

    model = ModelParams(
        parser,
        sentinel=True
    )

    pipeline = PipelineParams(parser)

    parser.add_argument(
        "--iteration",
        default=-1,
        type=int
    )

    parser.add_argument(
        "--skip_train",
        action="store_true"
    )

    parser.add_argument(
        "--skip_test",
        action="store_true"
    )

    parser.add_argument(
        "--quiet",
        action="store_true"
    )

    args = get_combined_args(parser)

    print(
        "Rendering:",
        args.model_path
    )

    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test
    )