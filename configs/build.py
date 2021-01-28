from configs import CN

__all__ = [
    'build_config',
    'CTConfig',
]


def build_config(cfg, name):
    """
    Built the config, defined by `cfg.config.name`.
    """
    cfg_dict = {
        'ctconfig': CTConfig,
    }
    if name == '':
        return cfg
    else:
        assert name.lower() in cfg_dict, f"{name} not found."
        return cfg_dict[name.lower()](cfg)

def CTConfig(cfg):
    # loss
    cfg.loss.label_smoothing = 0.1

    # dataset
    cfg.dataset.slice_num = 16
    cfg.dataset.is_color = False # gray slice image
    cfg.dataset.is_3d = True

    # model_depth
    cfg.model.in_channels = 1
    cfg.model.dropout = 0
    # MobileNet
    cfg.model.name = 'Mobile3DNet'
    cfg.model.width_stages = [32,64,128,256,512,1024]
    cfg.model.n_cell_stages = [4,4,4,4,4,1]
    cfg.model.stride_stages = [2,2,2,1,2,1]
    cfg.model.width_mult = 1
    cfg.model.classes = 3
    cfg.model.dropout_rate = 0.
    cfg.model.bn_param = (0.1, 1e-3)

    # CAM
    cfg.cam = CN()
    cfg.cam.enable = 0
    cfg.cam.scan_path = '' # the path of a scan
    cfg.cam.label = -1
    cfg.cam.featmaps_module_name = 'global_avg_pooling' # the module name of hook
    cfg.cam.weights_module_name = 'classifier' # the module name of hook
    cfg.cam.save_path = './cam_results'
    cfg.cam.model_path = '' # load the params of the model
    cfg.cam.debug = False # if True, use FakeNet3D and FakeData to debug

    ################################################################
    # ct transforms                                                #
    # https://torchio.readthedocs.io/                              #
    ################################################################
    cfg.transforms.ct = CN()
    cfg.transforms.ct.randomflip = CN()
    cfg.transforms.ct.randomflip.enable = 1
    cfg.transforms.ct.randomflip.p = 0.5 
    cfg.transforms.ct.randomflip.axes = (0, 1,2) 
    cfg.transforms.ct.randomflip.flip_probability = 0.5

    cfg.transforms.ct.randomaffine = CN()
    cfg.transforms.ct.randomaffine.enable = 0 
    cfg.transforms.ct.randomaffine.scales = (0.5,0.5)
    cfg.transforms.ct.randomaffine.degrees = (-10,10)
    cfg.transforms.ct.randomaffine.isotropic = True
    cfg.transforms.ct.randomaffine.p = 0.5

    cfg.transforms.ct.randomblur = CN()
    cfg.transforms.ct.randomblur.enable = 0
    cfg.transforms.ct.randomblur.p = 0.5
    cfg.transforms.ct.randomblur.std = (0, 4)

    cfg.transforms.ct.randomnoise = CN()
    cfg.transforms.ct.randomnoise.enable = 0
    cfg.transforms.ct.randomnoise.p = 0.5
    cfg.transforms.ct.randomnoise.mean = (0,0.25)
    cfg.transforms.ct.randomnoise.std = (0,0.25)

    cfg.transforms.ct.randomswap = CN()
    cfg.transforms.ct.randomswap.enable = 0
    cfg.transforms.ct.randomswap.p = 0.5
    cfg.transforms.ct.randomswap.patch_size = (16,16,16)
    cfg.transforms.ct.randomswap.num_iterations = 100

    cfg.transforms.ct.randomelasticdeformation = CN()
    cfg.transforms.ct.randomelasticdeformation.enable = 0
    cfg.transforms.ct.randomelasticdeformation.p = 0.5
    cfg.transforms.ct.randomelasticdeformation.num_control_points = (4,4,4)
    cfg.transforms.ct.randomelasticdeformation.max_displacement = (7,7,7)
    cfg.transforms.ct.randomelasticdeformation.locked_borders = 0
    return cfg
