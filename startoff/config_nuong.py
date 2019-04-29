from trixi.util import Config

experiment_config = Config(
    data_path="C:/Users/Nuong/Desktop/MA-Arbeitsordner/data",

    convert=False
    convert_path="raw"
    #test_data="train_batch.npy",
    #test_data="test_batch.npy",
    tensorboard_path="C:/Users/Nuong/Desktop/MA-Arbeitsordner/exp2/tensorboard_log",
    
    run=Config(
        hyperparameter=Config(
            lr=1e-3,
            batch_size=128,
        ),
        use_cuda=True,
        gpu_nr = 0,
        load_path=None,
        n_worker=12,
        logging=Config(
            log_interval=20
        )
    ),
    base_dir="C:/Users/Nuong/Desktop/MA-Arbeitsordner/data",
    log_interval=20,
    name="test_nuong",
    n_epochs=10,
)
