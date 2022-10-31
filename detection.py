from reverse_engineering import *
from config import get_argument
from dataloader import get_dataloader_label_remove, get_dataloader_partial_split
import time

def main():
    start_time = time.time()
    opt = get_argument().parse_args()

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 10
        opt.total_label = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        size = 32
        channel = 3
        opt.t_mean = torch.FloatTensor(mean).view(channel,1,1).expand(channel, size, size).cuda()
        opt.t_std = torch.FloatTensor(std).view(channel,1,1).expand(channel, size, size).cuda()
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 43
        opt.total_label = 43
        mean = [0.5]
        std = [0.5]
        size = 32
        channel = 1
        opt.t_mean = torch.FloatTensor(mean).view(channel,1,1).expand(channel, size, size).cuda()
        opt.t_std = torch.FloatTensor(std).view(channel,1,1).expand(channel, size, size).cuda()
    elif opt.dataset == "mnist":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 1
        opt.num_classes = 10
        opt.total_label = 10
        mean = [0,0,0]
        std = [1,1,1]
        size = opt.input_height
        channel = 3
        opt.t_mean = torch.FloatTensor(mean).view(channel,1,1).expand(channel, size, size).cuda()
        opt.t_std = torch.FloatTensor(std).view(channel,1,1).expand(channel, size, size).cuda()
    else:
        raise Exception("Invalid Dataset")

    trainset, transform, trainloader, testset, testloader = get_dataloader_partial_split(opt, train_fraction=opt.data_fraction, train=False)
    opt.total_label = opt.num_classes
    opt.re_dataset_total_fixed = trainset
    opt.re_dataloader_total_fixed = trainloader


    dummy_model = RegressionModel(opt, None).to(opt.device)
    opt.feature_shape = []
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        features = dummy_model.classifier.from_input_to_features(inputs.cuda(), opt.internal_index)
        for i in range(1, len(features.shape)):
            opt.feature_shape.append(features.shape[i])
        break
    del dummy_model
    init_mask = torch.ones(opt.feature_shape)

    opt.pretrain_AE = None
    get_range(opt, init_mask)

    final_mixed_value_list = []
    
    if opt.set_all2one_target == "all":
        for target in range(opt.num_classes):
            print("----------------- Analyzing all2one: target{}  -----------------".format(target))
            opt.target_label = target
            re_dataloader = get_dataloader_label_remove(opt,opt.re_dataset_total_fixed,label=opt.target_label)
            data_list = []
            for batch_idx, (inputs, labels) in enumerate(re_dataloader):
                print(batch_idx)
                print(inputs.shape)
                data_list.append(inputs)
            opt.data_now = data_list
            recorder, opt = train(opt, init_mask)
            final_mixed_value_list.append(recorder.mixed_value_best.item())

    else:
        target = int(opt.set_all2one_target)

        print("----------------- Analyzing all2one: target{}  -----------------".format(target))
        opt.target_label = target
        re_dataloader = get_dataloader_label_remove(opt,opt.re_dataset_total_fixed,label=opt.target_label)

        data_list = []
        for batch_idx, (inputs, labels) in enumerate(re_dataloader):
            print(batch_idx)
            print(inputs.shape)
            data_list.append(inputs)
        opt.data_now = data_list
        recorder, opt = train(opt, init_mask)
        final_mixed_value_list.append(recorder.mixed_value_best.item())

    end_time = time.time()
    print("total time: ",end_time-start_time)
    print("final_mixed_value_list:",final_mixed_value_list)

    if min(final_mixed_value_list) < opt.mixed_value_threshold:
        print("Trojaned")
    else:
        print("Benign")


if __name__ == "__main__":
    main()
