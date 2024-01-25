import os
import pandas as pd
import torch
import random
import numpy as np
import argparse
from score import Score
import warnings

warnings.filterwarnings("ignore")


def main():
    # 小时数据集
    MD_SubDataset = ['MD_manu', 'MD_food', 'MD_phar']
    ETTm_SubDataset = ['ETTm_MIX.csv']


    # seed
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='AnaNET: Anatomical Network')

    ## Select the dataset to train
    parser.add_argument('--dataset_class', type=str, default='MD', help='Data class for evaluation')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')

    ## Define the model to be used here:
    parser.add_argument('--model', type=str, default='AnaNET', help='model name')

    # Training settings
    parser.add_argument('--is_training', type=int, default=1, help='decide whether to train or not')
    parser.add_argument('--do_predict', type=int, default=0, help='decide whether to predict or not')
    parser.add_argument('--do_d', type=int, default=0, help='decide whether to predict D or not')
    parser.add_argument('--draw_num', type=int, default=-1, help='Which image to draw')
    parser.add_argument('--checkpoints_path', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    # Forecasting settings
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task following by Informer')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--major', type=int, default=64, help='major modes to be selected')
    parser.add_argument('--size', type=int, default=24, help='windows to decomp')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=256 * 4, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--alpha', type=int, default=100, help='VMD used')

    # Optimization
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1,
                        help='The final score is the average of the number of experiments')
    parser.add_argument('--training_metric', type=str, default='mse', help='metric during training')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multi gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.is_training:
        sign = 1
        # Reading Different Dataset Class
        current_dataset = args.dataset_class
        args.current_dataset = current_dataset
        print(current_dataset)
        subdataset = eval(current_dataset + '_SubDataset')
        # Reading Different Dataset Subsets
        for index, item in enumerate(subdataset):
            args.data = item
            metrics_df = pd.DataFrame()
            for ii in range(args.itr):
                current_setting = '{}_on_{}-{}__seq_{}-{}-{}__freq_{}__{}_{}_{}'.format(
                    args.model,
                    current_dataset,
                    item,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.freq,
                    args.target,
                    ii,
                    args.alpha
                )

                score = Score(args, current_dataset, item)
                print(
                    '####  start score on {} , progress {}/{}'.format(current_dataset, index + 1, len(subdataset)))

                print('####  start training : {}#########'.format(current_setting))
                score.train(current_setting)

                print('####  start testing : {}##########'.format(current_setting))
                metrics_dict = score.test(current_setting)
                temp_df = pd.DataFrame([metrics_dict])
                metrics_df = pd.concat([metrics_df, temp_df], axis=0)

                torch.cuda.empty_cache()

            # Calculate the average score for multiple iterations
            itr_means = pd.DataFrame(metrics_df.mean()).T
            itr_means.insert(0, 'dataset', item)
            itr_means.insert(0, 'model', args.model)
            # Save result to CSV file (create if it doesn't exist; append data if it already exists)

            folder_path = './results/' + current_setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            if sign:
                itr_means.to_csv(folder_path + 'metrics.csv', mode='w', index=False)
                sign = 0
            else:
                itr_means.to_csv(folder_path + 'metrics.csv', mode='a', header=False, index=False)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(current_setting))
                #Make predictions and save the predicted results to/result/$current_ Under the settings folder
                score.predict(current_setting, True)
                score.draw(current_setting)
        print('####  task finish    ##########')

    else:
        current_dataset = args.dataset_class
        args.current_dataset = current_dataset
        subdataset = eval(current_dataset + '_SubDataset')
        # Reading Different Dataset Subsets
        for index, item in enumerate(subdataset):
            args.data = item
            for ii in range(args.itr):
                current_setting = '{}_on_{}-{}__seq_{}-{}-{}__freq_{}_{}_{}_{}'.format(
                    args.model,
                    current_dataset,
                    item,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.freq,
                    args.target,
                    ii,
                    args.alpha
                )
                score = Score(args, current_dataset, item)

                if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(current_setting))

                    score.predict(current_setting, True)
                    score.draw(current_setting)

        print('####  task finish    ##########')


if __name__ == "__main__":
    main()
