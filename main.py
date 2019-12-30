from StarGAN_v2 import StarGAN_v2
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of StarGAN_v2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test or refer_test ?')
    parser.add_argument('--dataset', type=str, default='celebA-HQ_gender', help='dataset_name')
    parser.add_argument('--refer_img_path', type=str, default='refer_img.jpg', help='reference image path')

    parser.add_argument('--iteration', type=int, default=100000, help='The number of training iterations')

    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch size') # each gpu
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of ckpt_save_freq')
    parser.add_argument('--gpu_num', type=int, default=1, help='The number of gpu')

    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_iter', type=int, default=50000, help='decay start iteration')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema decay value')

    parser.add_argument('--adv_weight', type=float, default=1, help='The weight of Adversarial loss')
    parser.add_argument('--sty_weight', type=float, default=1, help='The weight of Style reconstruction loss') # 0.3 for animal
    parser.add_argument('--ds_weight', type=float, default=1, help='The weight of style diversification loss') # 1 for animal
    parser.add_argument('--cyc_weight', type=float, default=1, help='The weight of Cycle-consistency loss') # 0.1 for animal

    parser.add_argument('--r1_weight', type=float, default=1, help='The weight of R1 regularization')
    parser.add_argument('--gp_weight', type=float, default=10, help='The gradient penalty lambda')

    parser.add_argument('--gan_type', type=str, default='gan', help='gan / lsgan / hinge / wgan-gp / wgan-lp / dragan')
    parser.add_argument('--sn', type=str2bool, default=False, help='using spectral norm')

    parser.add_argument('--ch', type=int, default=32, help='base channel number per layer')
    parser.add_argument('--n_layer', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    parser.add_argument('--style_dim', type=int, default=16, help='length of style code')

    parser.add_argument('--num_style', type=int, default=5, help='number of styles to sample')

    parser.add_argument('--img_height', type=int, default=256, help='The height size of image')
    parser.add_argument('--img_width', type=int, default=256, help='The width size of image ')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = StarGAN_v2(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            gan.train()
            print(" [*] Training finished!")

        elif args.phase == 'refer_test' :
            gan.refer_test()
            print(" [*] Refer test finished!")

        else :
            gan.test()
            print(" [*] Test finished!")



if __name__ == '__main__':
    main()