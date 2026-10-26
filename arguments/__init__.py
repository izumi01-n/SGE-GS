#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)

        for key, value in vars(self).items():
            shorthand = False

            if key.startswith("_"):
                shorthand = True
                key = key[1:]

            t = type(value)
            value = value if not fill_none else None

            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key,
                        ("-" + key[0:1]),
                        default=value,
                        action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key,
                        ("-" + key[0:1]),
                        default=value,
                        type=t
                    )
            else:
                if t == bool:
                    group.add_argument(
                        "--" + key,
                        default=value,
                        action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key,
                        default=value,
                        type=t
                    )

    def extract(self, args):
        group = GroupParams()

        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])

        return group



# Model Parameters
class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):

        # Basic
        self.sh_degree = 3

        self._source_path = ""
        self._model_path = ""
        self._images = "images"

        self._resolution = -1

        self._white_background = False

        self.data_device = "cuda"

        self.sky_seg = False
        self.load_normal = False
        self.load_depth = False

        self.eval = False

        self.sample_more_highres = False

        # 3D Filter kernel size
        self.kernel_size = 0.1

        # Anti-aliasing
        self.ray_jitter = False
        self.resample_gt_image = False

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g



# Pipeline Parameters
class PipelineParams(ParamGroup):
    def __init__(self, parser):

        self.convert_SHs_python = False
        self.compute_cov3D_python = False

        self.debug = False

        super().__init__(parser, "Pipeline Parameters")



# Optimization Parameters
class OptimizationParams(ParamGroup):
    def __init__(self, parser):

        # Iteration
        self.iterations = 30_000

        # Learning Rates
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001

        self.percent_dense = 0.01


        # Geometry Losses
        self.normal_loss = True
        self.sparse_loss = False
        self.flatten_loss = True
        self.depth_loss = True
        self.depth2normal_loss = False

        self.lambda_l1_normal = 0.15
        self.lambda_cos_normal = 0.01

        self.lambda_flatten = 2.0

        self.lambda_dssim = 0.2
        self.lambda_sparse = 0.001

        self.lambda_depth = 0.20
        self.lambda_depth2normal = 0.05


        # Densification
        self.densification_interval = 100

        self.opacity_reset_interval = 3000

        self.densify_from_iter = 500
        self.densify_until_iter = 15_000

        self.densify_grad_threshold = 0.0004

        self.random_background = False


        # Normal Loss Start
        self.normal_iter_start = 4000


        # GaussianPro Propagation Parameters
        self.dataset = 'free'                  

        self.propagation_interval = 200        

        self.depth_error_min_threshold = 0.05  
        self.depth_error_max_threshold = 0.05 

        self.propagated_iteration_begin = 4000 
        self.propagated_iteration_after = 15000#

        self.patch_size = 16                  

        # Progressive patch decay
        self.patch_size_decay_steps = 2500

        self.pair_path = ''

        super().__init__(parser, "Optimization Parameters")



# Merge cfg_args + commandline
def get_combined_args(parser: ArgumentParser):

    cmdlne_string = sys.argv[1:]

    cfgfile_string = "Namespace()"

    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(
            args_cmdline.model_path,
            "cfg_args"
        )

        print("Looking for config file in", cfgfilepath)

        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()

    except TypeError:
        print("Config file not found")
        pass

    # safer eval
    try:
        args_cfgfile = eval(cfgfile_string)
    except:
        args_cfgfile = Namespace()

    merged_dict = vars(args_cfgfile).copy()

    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v

    return Namespace(**merged_dict)