from trixi.util import Config
import glob
import os

data_path_="C:\\Users\\Nuong\\Desktop\\heidelberg\\MA-Arbeitsordner\\data"
base_dir_="C:\\Users\\Nuong\\Desktop\\heidelberg\\MA-Arbeitsordner\\experiment_dir"
model_path_=os.path.join(base_dir_,"20190510-084116_test_nuong")
exp_no=2
old_models=glob.glob(f"{base_dir_}\\exp{exp_no}\\*.pt")
experiment_config = Config(
    data_path=data_path_,
    base_dir=base_dir_,
    load_old_checkpoint=True,    
    load_path=os.path.join(model_path_,"checkpoint"),#os.path.join("checkpoint","checkpoint_last.pth.tar")),
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
        #load_path=None,
        normalize_data=True,
        n_worker=12,
        logging=Config(
            log_interval=20
        )
    ),
    using_model="Net",
    log_interval=20,
    name="test_nuong",
    n_epochs=10000,
    no_cuda=True,
    loss1="MSE",
    loss2="SSIM",
    #resume=os.path.join(model_path_,"result"),
    #ignore_resume_config=True,
    #resume_reset_epochs=True,
    #resume_save_types=("model", "optimizer", "simple", "th_vars")
    
)
