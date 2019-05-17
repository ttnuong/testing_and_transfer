from trixi.util import Config
import glob
import os

data_path_="D:\\video_stash\\thisenv\\data"
base_dir_="D:\\video_stash\\experiment_dir"
model_path_=os.path.join(base_dir_,"20190515-191613_test_MSE_SSIM_50-50")
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
        normalize_data=False,
        n_worker=12,
        logging=Config(
            log_interval=20
        )
    ),
    using_model="Net",
    log_interval=20,
    name="test_display_unnormalized",
    n_epochs=10,
    no_cuda=False,
    loss1=0.4,
    loss2=0.6,
    #resume=os.path.join(model_path_,"result"),
    ignore_resume_config=True,
    resume_reset_epochs=True,
    #resume_save_types=("model", "optimizer", "simple", "th_vars")
    
)
