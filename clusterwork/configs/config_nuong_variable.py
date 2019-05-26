from trixi.util import Config
import glob
import os

data_path_=os.environ['DATASET_LOCATION']
base_dir_="/home/t262g/experiment_dir"
model_path_=os.path.join(base_dir_,"FIXED20190510-084116_test_nuong")

experiment_config = Config(
    #PATH DIRS
    data_path=data_path_,
    base_dir=base_dir_,
    load_path=os.path.join(model_path_,"checkpoint"),

    #LOGGING 
    tensorboard_path=os.path.join(base_dir_,"tensorboard_log"),
    #visdom_path=os.path.join(base_dir_,"visdom_log"),
    text_log=os.path.join("log","logfile.txt"),
    log_interval=20,

    ##########

    #EXPERIMENT CONFIG
    using_model="Net_wide",
    name="test_nuong_size",
    n_epochs=2,
    no_cuda=False,
    normalize_data=True,
    load_old_checkpoint=False,
    #MODEL CONFIG
    run=Config(
            hyperparameter=Config(
                lr=1e-3,
                batch_size=64,
                #momentum=0.9
                weight_decay=0
            ),
        loss1=0.4,
        loss2=0.6
    )

    #n_worker=12,
    #resume=os.path.join(model_path_,"result"),
    #ignore_resume_config=True,
    #resume_reset_epochs=True,
    #resume_save_types=("model", "optimizer", "simple", "th_vars")
    
)
