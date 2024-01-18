import os
import time
from data_provider.data_factory import data_provider
import numpy as np
import torch
import pandas as pd
from torch import optim
import torch.nn as nn
from torch.optim import lr_scheduler
# import logging
# logging.basicConfig(level=logging.INFO)
from models import AnaNET
## If adding a custom dataset, please import it in the file
from data_provider.MD import Dataset_Electricity

## If adding a custom model, please import it in the file
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_pr


class Score():

    def __init__(self, args, current_dataset, dataset):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.dataset = dataset
        self.current_dataset = current_dataset

    def _build_model(self):
        ## If adding a custom model, please import it in the file
        model = eval(self.args.model).Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        training_metric = self.args.training_metric


        metrics = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
                   }

        criterion = metrics[training_metric]
        return criterion

    def metrics_to_dict(self, metrics_str):
        metrics_dict = {}

        key_value_pairs = metrics_str.split(', ')
        for pair in key_value_pairs:
            key, value = pair.split(':')
            metrics_dict[key.strip()] = float(value)
        return metrics_dict


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            path = os.path.join(self.args.checkpoints_path, setting)
            best_model_path = path + '/' + 'checkpoint_best.pth'
            loaded_checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(loaded_checkpoint['model_state_dict'])
            self.model.topindex_encoder = loaded_checkpoint['topindex_encoder']
            self.model.topindex_decoder = loaded_checkpoint['topindex_decoder']
            self.model.topqindex = loaded_checkpoint['topqindex']
            self.model.topkindex = loaded_checkpoint['topkindex']
            print('model_path', best_model_path)


        preds = []
        trues = []
        count = 0

        # 保存每一次平移的prloss
        d0_loss = []
        d1_loss = []
        d2_loss = []
        df_loss = []
        d96_loss = []

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                test_pred_bf = outputs.detach().cpu().numpy().squeeze(axis=2)
                test_true_bf = batch_y.detach().cpu().numpy().squeeze(axis=2)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                # 如果开启验证且预测点数大于240可以计算多天的详细指标
                if self.args.do_d and self.args.pred_len > 240:
                    for j in range(len(test_pred_bf)):

                        test_pred = test_pred_bf[j]
                        test_true = test_true_bf[j]

                        # 获取详细天的数值
                        d0_pred = test_pred[0:24]
                        d0_true = test_true[0:24]

                        d1_pred = test_pred[24:48]
                        d1_true = test_true[24:48]

                        d2_pred = test_pred[48:72]
                        d2_true = test_true[48:72]

                        df_pred = test_pred[-24:-1]
                        df_true = test_true[-24:-1]

                        # 计算详细天的指标
                        pr0, _, _ = metric(d0_pred, d0_true)
                        pr1, _, _ = metric(d1_pred, d1_true)
                        pr2, _, _ = metric(d2_pred, d2_true)
                        prf, _, _ = metric(df_pred, df_true)

                        d0_loss.append(pr0)
                        d1_loss.append(pr1)
                        d2_loss.append(pr2)
                        df_loss.append(prf)

                # 针对96的D指标，只进行一次计算，即全部的数据
                if self.args.do_d and self.args.pred_len < 200:
                    for j in range(len(test_pred_bf)):

                        test_pred = test_pred_bf[j]
                        test_true = test_true_bf[j]

                        # 计算详细天的指标
                        pr96, _, _ = metric(test_pred, test_true)

                        d96_loss.append(pr96)

                preds.append(pred)
                trues.append(true)

                # 选择两端的进行绘制
                # if i % 2 == 0:
                #     if count == self.args.draw_num:
                #         input = batch_x.detach().cpu().numpy()
                #         gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #         pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #         visual(gt, pd, os.path.join(folder_path, "test" + '.pdf'))
                #     count = count + 1

                if i % 2 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(count) + '.pdf'))
                    count = count + 1



        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        if self.args.do_d and self.args.pred_len > 240:
            visual_pr(d0_loss, os.path.join(folder_path, 'd0_loss.pdf'))
            visual_pr(d1_loss, os.path.join(folder_path, 'd1_loss.pdf'))
            visual_pr(d2_loss, os.path.join(folder_path, 'd2_loss.pdf'))
            visual_pr(df_loss, os.path.join(folder_path, 'df_loss.pdf'))

        if self.args.do_d and self.args.pred_len < 200:
            visual_pr(d96_loss, os.path.join(folder_path, 'd96_loss.pdf'))

        mae, mse, rmse = metric(preds, trues)
        metric_str = 'mae:{}, mse:{}, rmse:{}'.format(mae, mse, rmse)
        print(metric_str)

        metrics_dict = self.metrics_to_dict(metric_str)

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return metrics_dict


    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')


        path = os.path.join(self.args.checkpoints_path, setting)
        if not os.path.exists(path):
            os.makedirs(path)


        time_now = time.time()


        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)


        model_optim = self._select_optimizer()
        criterion = self._select_criterion()



        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\tepoch: {0}, iters: {1},  | loss: {2:.7f}".format(epoch + 1, i + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)


            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


        return vali_loss

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:

            path = os.path.join(self.args.checkpoints_path, setting)
            best_model_path = path + '/' + 'checkpoint_best.pth'
            loaded_checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(loaded_checkpoint['model_state_dict'])
            self.model.topindex_encoder = loaded_checkpoint['topindex_encoder']
            self.model.topindex_decoder = loaded_checkpoint['topindex_decoder']
            self.model.topqindex = loaded_checkpoint['topqindex']
            self.model.topkindex = loaded_checkpoint['topkindex']
            print('model_path：', best_model_path)



        preds = []

        self.model.eval()


        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                pred = outputs.detach().cpu().numpy().squeeze(axis=2)
                pred = pred_data.inverse_transform(pred)
                preds.append(pred[0])

        preds = np.array(preds)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'real_prediction.npy', preds)
        df = pd.DataFrame(preds[0], columns=['OT'])
        df.to_csv(folder_path + '/OT.csv', index=False)

        return

    # 绘制预测效果图
    def draw(self, setting):
        # 读取预测结果
        path = './results/' + setting + '/'
        pred_path = './results/' + setting + '/' + 'real_prediction.npy'
        pred = np.load(pred_path)

        root_path = str(self.args.root_path)
        data_path = str(self.args.data)

        df_raw = pd.read_csv(os.path.join(root_path, data_path) + '.csv')['OT']

        border1 = len(df_raw) - self.args.pred_len - self.args.seq_len
        border2 = len(df_raw)

        df_raw = (df_raw[border1:border2]).reset_index(drop=True)

        input = df_raw[:self.args.seq_len]
        gt = df_raw
        pr = np.concatenate((input, pred[0]), axis=0)
        visual(gt, pr, os.path.join(path, "pred" + '.pdf'))

    def D1_predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='D1_pred')

        if load:
            path = os.path.join(self.args.checkpoints_path, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        pr_loss = []    #保存每一次平移的prloss

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy().squeeze(axis=2)
                # 这边要进行验证
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                true = batch_y.detach().cpu().numpy().squeeze(axis=2)
                # 这边如果把预测长度 提升了 应该是只取后一半即可
                pred = pred[len(pred)/2:-1]
                true = true[len(pred)/2:-1]
                pr, _, _ = metric(pred, true)
                pr_loss.append(pr)
                pred = pred_data.inverse_transform(pred)
                preds.append(pred[0])

        preds = np.array(preds)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        # pr_loss = pr_loss.detach().cpu().numpy().squeeze()
        visual_pr(pr_loss, os.path.join(folder_path,'pr_loss.pdf'))
        return



if __name__ == "__main__" :
    pass