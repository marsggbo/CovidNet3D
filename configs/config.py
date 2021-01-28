import os
import sys
sys.path.append('..')

from yacs.config import CfgNode as _CfgNode

from utils import MyLogger

__all__ = [
    'CfgNode',
    'get_cfg',
    'CN'
]

class CfgNode(_CfgNode):

    def setup(self, args):
        self.merge_from_file((args.config_file))
        self.merge_from_list(args.opts)

        # calibrate the path configruration
        if self.model.resume_path:
            self.logger.path = os.path.dirname(self.model.resume_path)
        else:
            if not self.logger.name:
                self.logger.name = 'checkpoint'
            self.logger.version = self._version_logger(self.output_root, self.logger.name)
            self.logger.path = os.path.join(self.output_root, self.logger.name, f'version_{self.logger.version}')
        self.logger.log_file = os.path.join(self.logger.path, 'log.txt')
        cfg_name = os.path.basename(args.config_file)
        self.logger.cfg_file = os.path.join(self.logger.path, cfg_name)
        os.makedirs(self.logger.path, exist_ok=True)
        self.freeze()

        # backup cfg and args
        logger = MyLogger('NAS', self).getlogger()
        logger.info(self)
        logger.info(args)
        with open(self.logger.cfg_file, 'w') as f:
            f.write(str(self))

    @staticmethod
    def _version_logger(save_dir, logger_name=''):
        if logger_name:
            path = os.path.join(save_dir, logger_name)
        else:
            path = save_dir
        if (not os.path.exists(path)) or (not os.listdir(path)):
            version = 0
        else:
            versions = [int(v.split('_')[-1]) for v in os.listdir(path)]
            version = max(versions)+1
        return version

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            v = f"'{v}'" if isinstance(v, str) else v
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 4)
            s.append(attr_str)
        r += "\n".join(s)
        return r

global_cfg = CfgNode()
CN = CfgNode

def get_cfg():
    '''
    Get a copy of the default config.

    Returns:
        a CfgNode instance.
    '''
    from .default import _C
    return _C.clone()
