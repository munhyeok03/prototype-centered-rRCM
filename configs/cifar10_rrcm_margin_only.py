import importlib.util
from pathlib import Path


def _load_base_config():
    path = Path(__file__).with_name("cifar10_finetune.py")
    spec = importlib.util.spec_from_file_location("cifar10_finetune_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def get_config():
    config = _load_base_config()

    config.use_margin_aware_loss = True
    config.use_proto_loss = False
    config.use_margin_loss = True
    config.lambda_proto = 0.0
    config.lambda_margin = 0.1
    config.prototype_margin = 0.2
    config.prototype_temperature = 0.1
    config.prototype_source = "classifier_weight"
    config.normalize_features = True
    config.log_geometry_metrics = True

    return config
