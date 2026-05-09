default_log_path = '/lustre/fsn1/projects/rech/thj/uth68ud/slurm_out/'
default_code_path = '/lustre/fswork/projects/rech/thj/uth68ud/PycharmProjects/Qinco2/'
default_data_path = '/lustre/fswork/projects/rech/thj/ufq64xt/data_hdvc/'
default_result_path = '/lustre/fsn1/projects/rech/thj/uth68ud/qinco2_results'


default_data_info = {

    'bigann': {
        'path': '/lustre/fswork/projects/rech/thj/ufq64xt/data_hdvc/bigann/SIFT1M',
        'train_fname': 'bigann_learn.bvecs',
        'test_fname': 'bigann_base.bvecs',
        'query_fname': 'bigann_query.bvecs',
        'all_M': [1, 2, 4, 8, 16, 32, 64, 128],
    },

    'deep': {
        'path': '/lustre/fswork/projects/rech/thj/ufq64xt/data_hdvc/deep',
        'train_fname': 'learn_100m.fvecs',
        'test_fname': 'test_1m.fvecs',
        'query_fname': 'query_10k.fvecs',
        'all_M': [1, 2, 4, 8, 12, 24, 32, 96],
    },

    'gist': {
        'path': '/lustre/fswork/projects/rech/thj/ufq64xt/data_hdvc/gist',
        'train_fname': 'gist_learn.fvecs',
        'test_fname': 'gist_base.fvecs',
        'query_fname': 'gist_query.fvecs',
        'all_M': [1, 2, 4, 8, 40, 60, 320, 480, 960]
    },

    'msmarco': {
        'path': '/lustre/fswork/projects/rech/thj/ufq64xt/data_hdvc/msmarco',
        'train_fname': 'train.fvecs',
        'test_fname': 'base.fvecs',
        'query_fname': 'query.fvecs',
        'all_M': [1, 2, 4, 8, 32, 64, 256, 512, 1024]
    },

    'openai': {
        'path': '/lustre/fswork/projects/rech/thj/ufq64xt/data_hdvc/openai',
        'train_fname': 'openai_train1m.fvecs',
        'test_fname': 'openai_base1m.fvecs',
        'query_fname': 'openai_query10k.fvecs',
        'all_M': [1, 2, 4, 8, 32, 128, 256, 512, 1536]
    }
}

