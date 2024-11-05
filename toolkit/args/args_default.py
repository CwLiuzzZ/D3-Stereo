import configargparse

def get_opts():
    parser = configargparse.ArgumentParser()

    # dataset options
    parser.add_argument('--dataset', default = 'MiddEval3H', type=str)
    parser.add_argument('--dataset_type', type=str, default='train', choices=['train','test'],help = 'use train or test dataset')
    parser.add_argument('--keep_size', default=False, type=bool, help='do not resize the input image')
    parser.add_argument('--max_disp', type=int, default=192, help="max_disparity of the datasets")
    parser.add_argument('--max_disp_sr', type=int, default=192, help="max_disparity when estimate the disparity")
    parser.add_argument('--save_name', type=str, default='Delete', help="name for saving the results")

    # model options
    parser.add_argument('--network', type=str, default='BGNet') 
    parser.add_argument('--ckpt_path', type=str, default=None, help='pretrained checkpoint path to load')

    # training options
    parser.add_argument('--inference_type', type=str, default='evaluate', help='main inference, and has effect on the aug_config', choices=['evaluate','train','DDFM','d3_train']) 
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate') 
    parser.add_argument('--freeze_bn', default=True, type=bool)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_steps', type=int, default=2000, help='number of training steps')
    parser.add_argument('--num_workers', type=int, default=16, help='number of training epochs')
    parser.add_argument('--epoch_size', type=int, default=200, help='number of training epochs')

    # D3Stereo options
    parser.add_argument('--pt', default = False, type=bool, help='if use perspective transformation')
    parser.add_argument('--ratio_th_start', type=float, default = 0, help='start ratio_th at the k-th layer')
    parser.add_argument('--pt_offset', default = 30, type=str, help='also modified for each network in DDFM.py')

    parser.add_argument('--RBF_cycle', type=int, nargs='+', default = [9,11,14], help='Recursive Bilateral maximux iteration number')
    parser.add_argument('--NCC_enhance', type=bool, default = False, help='Use NCC for resolution enhancement')
    parser.add_argument('--ratio_th', type=float, nargs='+', default = [0.7,0.7,0.8], help='ratio_th for each layer, will be edited in DDFM.py')
    parser.add_argument('--BF_i', type=float, default = 5, help='weight parameter for Recursive Bilateral Filter, will be edited in DDFM.py')
    parser.add_argument('--D3_SR', type=int, nargs='+', default = [-1,0,1], help='D3 search propogation bound range')
    parser.add_argument('--BF_rad', default = 1, type=int, help='Bilateral filter radio, in our Recursive bilateral filter is always set to 1.')
    parser.add_argument('--D3_rad', default = 1, type=int, help='D3 diffusion radius')
    return parser.parse_args()