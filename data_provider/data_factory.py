from data_provider.MD import Dataset_Electricity
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Pred, D0_Dataset_Pred, Dataset_ETT_minute, Dataset_ETT_hour

data_dict = {
    'DCP': Dataset_Electricity,
    'ECL': Dataset_Electricity,
    'ETTm': Dataset_ETT_minute,

}


def data_provider(args, flag):
    data_class = args.current_dataset[args.current_dataset.rfind('_')+1: ]
    Data = data_dict[data_class]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    elif flag == 'D0_pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = D0_Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        args.root_path,
        data_path=args.data,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
