import argparse

def build_parser():
    parser = argparse.ArgumentParser(description='Optimization Utilities in Torch')
    parser.add_argument('--algo', type=str, default='GradDescent')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--line_search', type=bool, default=True)
    parser.add_argument('--search_algo', type=str, default='BacktrackLineSearch')
    parser.add_argument('--max_step_len', type=float, default=0.8)
    parser.add_argument('--step_coeff', type=float, default=0.8)
    parser.add_argument('--step_iter', type=int, default=20)

    args = parser.parse_args()
    return args
