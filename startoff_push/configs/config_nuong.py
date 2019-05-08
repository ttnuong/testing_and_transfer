from trixi.util import Config
import glob
import os
from common.networks.nuong_AE import Net
data_path_="C:\\Users\\Nuong\\Desktop\\heidelberg\\MA-Arbeitsordner\\data"
base_dir_="C:\\Users\\Nuong\\Desktop\\heidelberg\\MA-Arbeitsordner\\experiment_dir"
model_path_=os.path.join(base_dir_,"20190508-150956_test_nuong")
exp_no=2
old_models=glob.glob(f"{base_dir_}\\exp{exp_no}\\*.pt")
experiment_config = Config(
    data_path=data_path_,
    base_dir=base_dir_,
    load_old_model=True,    
    load_path=os.path.join(model_path_,os.path.join("checkpoint","checkpoint_last.pth.tar")),
    #tensorboard_path=os.path.join(base_dir_,f"tensorboard_log"),
    tensorboard_path=os.path.join(base_dir_,"tensorboard_log"),
    #visdom_path=os.path.join(base_dir_,f"exp{exp_no}\\visdom_log"),
    text_log=os.path.join("log","logfile.txt"),
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
    using_model="Net",
    log_interval=20,
    name="test_nuong",
    n_epochs=1,
    no_cuda=True
)
