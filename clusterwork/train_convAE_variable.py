import os
from experiments.experiment_nuong import ConvAE_CIFAR #ConvAE_Feat
from configs.config_nuong_variable import experiment_config

if __name__ == '__main__':
    experiment = ConvAE_CIFAR(config=experiment_config,
                                 #globs=globals(),
                                 loggers={
                                     #"visdom": ("visdom", {"auto_start": False, "port": 8080}, 1),
                                     "tensorboard": ("tensorboard", dict(target_dir=experiment_config.tensorboard_path)),
                                 },
                                 explogger_freq = 1
                                 
                                 #ignore_resume_config=True
                                 )
    experiment.run()

