import os
import argparse


def args_to_string(args):
    """
    Transform experiment's arguments into a string
    :param args:
    :return: string
    """
    args_string = ""


    args_to_show = ["method", "n_rounds", "num_epochs", "num_heads", "KB_len", "beta", "distillation_weight", "pca", "lr"]

    extracted_args = {arg: getattr(args, arg) for arg in args_to_show}


    args_string = os.path.join(args_string, str(extracted_args))

    return args_string


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--experiment',
        help='name of experiment',
        type=str
    )


    parser.add_argument(
        '--input_dimension',
        help='the dimension of one input sample; only used for synthetic dataset',
        type=int,
        default=None
    )
    parser.add_argument(
        '--output_dimension',
        help='the dimension of output space; only used for synthetic dataset',
        type=int,
        default=None
    )

    parser.add_argument(
        '--n_rounds',
        help='number of communication rounds; default is 1',
        type=int,
        default=40
    )
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--maxlen', default=15, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--distillation_weight', default=1.0, type=float)
    parser.add_argument('--distil_para', default=10.0, type=float)
    parser.add_argument('--rand', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--method', default='my', type=str)
    parser.add_argument('--KB_len', default=0, type=int)
    parser.add_argument('--KB_all', default=0, type=int)
    parser.add_argument('--update_round', default=1, type=int)
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--pca', action='store_true')  
    parser.add_argument('--output', default='output.txt', type=str)
    parser.add_argument(
        '--clients_number',
        help='count of clients',
        type=int,
        default=698
    )
    parser.add_argument(
        '--local_steps',
        help='number of local steps before communication; default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--log_freq',
        help='frequency of writing logs; defaults is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--optimizer',
        help='optimizer to be used for the training; default is sgd',
        type=str,
        default="sgd"
    )
    parser.add_argument(
        "--mu",
        help='proximal / penalty term weight, used when --optimizer=`prox_sgd` also used with L2SGD; default is `0.`',
        type=float,
        default=0
    )


    parser.add_argument(
        "--locally_tune_clients",
        help='if selected, clients are tuned locally for one epoch before writing logs;',
        action='store_true'
    )
    parser.add_argument(
        '--validation',
        help='if chosen the validation part will be used instead of test part;'
             ' make sure to use `val_frac > 0` in `generate_data.py`;',
        action='store_true'
    )
    parser.add_argument(
        "--verbose",
        help='verbosity level, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`;',
        type=int,
        default=0
    )
    parser.add_argument(
        "--logs_dir",
        help='directory to write logs; if not passed, it is set using arguments',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--save_dir",
        help='directory to save checkpoints once the training is over; if not specified checkpoints are not saved',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--seed",
        help='random seed 默认42',
        type=int,
        default=42
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    return args
