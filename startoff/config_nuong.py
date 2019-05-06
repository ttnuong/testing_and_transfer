from trixi.util import Config
import glob


experiment_config = Config(
    data_path="C:\\Users\\Nuong\\Desktop\\heidelberg\\MA-Arbeitsordner\\data",
    load_old=True
    exp_no=1
    old_models=glob.glob(os.path.join(data_path,os.path.join(f"exp{exp_no}","*")))
    model_load_path=os.path.join(data_path,os.path.join(f"exp{exp_no}",f"cifar_cnn_{len(old_models)}.pt"))
    tensorboard_path="C:\\Users\\Nuong\\Desktop\\heidelberg\\MA-Arbeitsordner\\exp2\\tensorboard_log",
    
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
    base_dir="C:\\Users\\Nuong\\Desktop\\heidelberg\\MA-Arbeitsordner\\data",
    log_interval=20,
    name="test_nuong",
    n_epochs=10,
    no_cuda=True
)
