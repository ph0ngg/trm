import argparse
import json
import time
from time import gmtime, strftime
import test
import sys
#from models import *
from shutil import copyfile
from utils.datasets import JointDataset, collate_fn
from utils.utils import *
from utils.log import logger
from torchvision.transforms import transforms as T

from models.yolox import YOLOX, Head2
from collections import defaultdict

def train(
        cfg,
        data_cfg,
        weights_from="",
        weights_to="",
        save_every=10,
        img_size=(1088, 608),
        resume=False,
        epochs=100,
        batch_size=4,
        accumulated_batches=1,
        freeze_backbone=False,
        opt=None,
):
    # The function starts

    timme = strftime("%Y-%d-%m %H:%M:%S", gmtime())
    timme = timme[5:-3].replace('-', '_')
    timme = timme.replace(' ', '_')
    timme = timme.replace(':', '_')
    weights_to = osp.join(weights_to, 'run' + timme)
    mkdir_if_missing(weights_to)
    if resume:
        latest_resume = osp.join(weights_from, 'latest.pt')

    torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    f = open(data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()

    transforms = T.Compose([T.ToTensor()])
    # Get dataloader
    dataset = JointDataset(dataset_root, trainset_paths, img_size, augment=True, transforms=transforms)
    print(dataset[0][1].shape)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size  , shuffle=True,
                                             num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    # Initialize model
    #model = Darknet(cfg, dataset.nID)
    model1 = YOLOX()
    model1.load_state_dict(torch.load('/kaggle/input/mot-17/kaggl/Towards-Realtime-MOT/yolox_l.pth')['model'], strict = True)

    model2 = Head2()
   

    
    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    if resume:
        checkpoint = torch.load(latest_resume, map_location='cuda:0')

        # Load weights to resume from
        model2.load_state_dict(checkpoint['model'])

        model2.cuda().train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model2.parameters()), lr=opt.lr, momentum=.9)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        del checkpoint  # current, saved

    else:
        # Initialize model with backbone (optional)
        # if cfg.endswith('yolov3.cfg'):
        #     load_darknet_weights(model, osp.join(weights_from, 'darknet53.conv.74'))
        #     cutoff = 75
        # elif cfg.endswith('yolov3-tiny.cfg'):
        #     load_darknet_weights(model, osp.join(weights_from, 'yolov3-tiny.conv.15'))
        #     cutoff = 15
        model1.cuda().eval()
        model2.cuda().train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model2.parameters()), lr=opt.lr, momentum=.9,
                                    weight_decay=1e-4)

    #model = torch.nn.DataParallel(model)
    # Set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[int(0.5 * opt.epochs), int(0.75 * opt.epochs)],
                                                     gamma=0.1)

    # An important trick for detection: freeze bn during fine-tuning
    # if not opt.unfreeze_bn:
    #     for i, (name, p) in enumerate(model.named_parameters()):
    #         p.requires_grad = False if 'batch_norm' in name else True
    # for name, param in model.named_parameters():
    #     name1 = name.split('.')
    #     param.requires_grad = False
    #     if name1[1] == 'head' and (name1[2] == 'cbam' or name1[2][0] == 'n'):
    #         param.requires_grad = True

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    # model_info(model)
    t0 = time.time()
    for epoch in range(epochs):
        print(epoch)
        total_loss = 0
        epoch += start_epoch
        logger.info(('%8s%12s' + '%10s' * 6) % (
            'Epoch', 'Batch', 'box', 'conf', 'id', 'total', 'nTargets', 'time'))

        # Freeze darknet53.conv.74 for first epoch
        if freeze_backbone and (epoch < 2):
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[2]) < cutoff:  # if layer < 75
                    p.requires_grad = False if (epoch == 0) else True

        ui = -1
        rloss = defaultdict(float)  # running loss
        optimizer.zero_grad()
        for i, (imgs, targets, imgs_path, _, targets_len) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            burnin = min(1000, len(dataloader))
            if (epoch == 0) & (i <= burnin):
                lr = opt.lr * (i / burnin) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr
        
            # Compute loss, compute gradient, update parameters
            imgs[:, 0] = imgs[:, 0] * 255
            imgs[:, 1] = imgs[:, 1] * 255
            imgs[:, 2] = imgs[:, 2] * 255
            model1.cuda().eval()
            xin, yolo_outputs, reid_idx = model1(imgs.cuda())
            filtered_outputs = []
            filtered_outputs_reid_idx = []
            for batch_idx in range(len(yolo_outputs)):
                batch_data = yolo_outputs[batch_idx]
                batch_reid_idx = reid_idx[batch_idx]
                class_mask = batch_data[:, 6] == 0
                filtered_output = batch_data[class_mask, :]
                filtered_reid_idx = batch_reid_idx[class_mask]
                filtered_outputs.append(filtered_output)
                filtered_outputs_reid_idx.append(filtered_reid_idx)
            
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.requires_grad)

            loss = model2(xin, filtered_outputs,  filtered_outputs_reid_idx, targets.cuda())
            total_loss += loss
            # except:
            #     print(imgs_path)
            #print(loss)
            loss = torch.mean(loss)
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1

            # for ii, key in enumerate(model.module.loss_names):
            #     rloss[key] = (rloss[key] * ui + components[ii]) / (ui + 1)

            # rloss indicates running loss values with mean updated at every epoch
            # s = ('%8s%12s' + '%10.3g' * 6) % (
            #     '%g/%g' % (epoch, epochs - 1),
            #     '%g/%g' % (i, len(dataloader) - 1),
            #     rloss['box'], rloss['conf'],
            #     rloss['id'], rloss['loss'],
            #     rloss['nT'], time.time() - t0)
            # t0 = time.time()
            # if i % opt.print_interval == 0:
            #     logger.info(s)
        
        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'model': model2.state_dict(),
                      'optimizer': optimizer.state_dict()}

        # copyfile(cfg, weights_to + '/cfg/yolo3.cfg')
        # copyfile(data_cfg, weights_to + '/cfg/ccmcpe.json')

        #weights_to = '/kaggle/working/'
        latest = osp.join(weights_to, 'latest.pt')
        torch.save(checkpoint, latest)
        if epoch % save_every == 0 and epoch != 0:
            # making the checkpoint lite
            checkpoint["optimizer"] = []
            torch.save(checkpoint, osp.join(weights_to, "weights_epoch_" + str(epoch) + ".pt"))
        print(total_loss)
        
        # Calculate mAP
        # if epoch % 10 == 5:
        #     with torch.no_grad():
        #         mAP, R, P = test.test(cfg, data_cfg, weights=latest, batch_size=batch_size, img_size=img_size,
        #                               print_interval=40, nID=dataset.nID)
        #         test.test_emb(cfg, data_cfg, weights=latest, batch_size=batch_size, img_size=img_size,
        #                       print_interval=40, nID=dataset.nID)

        # Call scheduler.step() after opimizer.step() with pytorch > 1.1.0
        scheduler.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights-from', type=str, default='weights/',
                        help='Path for getting the trained model for resuming training (Should only be used with '
                             '--resume)')
    parser.add_argument('--weights-to', type=str, default='weights/',
                        help='Store the trained weights after resuming training session. It will create a new folder '
                             'with timestamp in the given path')
    parser.add_argument('--save-model-after', type=int, default=10,
                        help='Save a checkpoint of model at given interval of epochs')
    parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe.json', help='coco.data file path')
    parser.add_argument('--img-size', type=int, default=[1088, 608], nargs='+', help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--print-interval', type=int, default=40, help='print interval')
    parser.add_argument('--test-interval', type=int, default=9, help='test interval')
    parser.add_argument('--lr', type=float, default=1e-2, help='init lr')
    parser.add_argument('--unfreeze-bn', action='store_true', help='unfreeze bn')
    opt = parser.parse_args()

    init_seeds()

    train(
        opt.cfg,
        opt.data_cfg,
        weights_from=opt.weights_from,
        weights_to=opt.weights_to,
        save_every=opt.save_model_after,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        opt=opt,
    )
