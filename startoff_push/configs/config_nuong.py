from trixi.util import Config
import glob
import os
data_path_="C:\\Users\\Nuong\\Desktop\\heidelberg\\MA-Arbeitsordner\\data"
base_dir_="C:\\Users\\Nuong\\Desktop\\heidelberg\\MA-Arbeitsordner"
exp_no=2
old_models=glob.glob(f"{base_dir_}\\exp{exp_no}\\*.pt")
experiment_config = Config(
    data_path=data_path_,
    base_dir=base_dir_,
    exp_no=1,
    load_old=False,
    model_load_path=os.path.join(base_dir_,os.path.join(f"exp{exp_no}",f"cifar_cnn_{len(old_models)}.pt")),
    #tensorboard_path=os.path.join(base_dir_,f"tensorboard_log"),
    tensorboard_path=base_dir_,
    visdom_path=os.path.join(base_dir_,f"exp{exp_no}\\visdom_log"),
    run=Config(
        hyperparameter=Config(
            lr=1e-3,
            batch_size=32,
        ),
        use_cuda=True,
        gpu_nr = 0,
        load_path=None,
        n_worker=12,
        logging=Config(
            log_interval=20
        )
    ),

    log_interval=20,
    name="_test_nuong",
    n_epochs=10,
    no_cuda=True
)
