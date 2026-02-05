import os
import time
import torch
import numpy as np
from torchvision import transforms
import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss, resize
from utils.evaluation import Evaluator
from utils.HSICommonUtils import ImageStretching
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict
from model.MambaMSF import MambaMSF
from calflops import calculate_flops
from config import TrainingConfig
from utils.experiment_utils import setup_seed

def vis_a_image(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=False):
    visualize_predict(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=only_vis_label)
    visualize_predict(gt_vis, pred_vis, save_single_predict_path.replace('.png','_mask.png'), save_single_gt_path, only_vis_label=True)

def main():
    cfg = TrainingConfig()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    save_folder = cfg.get_save_folder()
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"makedirs {save_folder}")

    # log
    save_log_path = os.path.join(save_folder, f'train_tr{cfg.train_samples}_val{cfg.val_samples}.log')
    logger = setup_logger(name=f'{cfg.data_set_name}', logfile=save_log_path)
    torch.cuda.empty_cache()
    logger.info(save_folder)

    data, gt = data_load_operate.load_data(cfg.data_set_name, cfg.data_set_path)
    height, width, channels = data.shape
    gt_reshape = gt.reshape(-1)
    img = ImageStretching(data)
    class_count = int(max(np.unique(gt)))
    
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    evaluator = Evaluator(num_class=class_count)

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    CLASS_ACC = np.zeros([len(cfg.seed_list), class_count])

    for exp_idx, curr_seed in enumerate(cfg.seed_list):
        setup_seed(curr_seed)
        
        single_experiment_name = f'run{exp_idx}_seed{curr_seed}'
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        if not os.path.exists(save_single_experiment_folder):
            os.makedirs(save_single_experiment_folder)
            
        save_vis_folder = os.path.join(save_single_experiment_folder, 'vis')
        if not os.path.exists(save_vis_folder):
            os.makedirs(save_vis_folder)

        save_weight_path = os.path.join(save_single_experiment_folder, 
                                      f"best_tr{cfg.train_samples}_val{cfg.val_samples}.pth")
        results_save_path = os.path.join(save_single_experiment_folder, 
                                       f'result_tr{cfg.train_samples}_val{cfg.val_samples}.txt')
        predict_save_path = os.path.join(save_single_experiment_folder, 
                                       f'pred_vis_tr{cfg.train_samples}_val{cfg.val_samples}.png')
        gt_save_path = os.path.join(save_single_experiment_folder, 
                                   f'gt_vis_tr{cfg.train_samples}_val{cfg.val_samples}.png')

        train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(
            cfg.ratio_list, [cfg.train_samples, cfg.val_samples], gt_reshape, class_count, cfg.flag_list[0]
        )
        
        index = (train_data_index, val_data_index, test_data_index)
        train_label, val_label, test_label = data_load_operate.generate_image_iter(
            data, height, width, gt_reshape, index
        )

        net = MambaMSF(in_channels=channels, num_classes=class_count, hidden_dim=cfg.hidden_dim)
        logger.info(net)

        x = transform(np.array(img))
        x = x.unsqueeze(0).float().to(cfg.device)

        train_label = train_label.to(cfg.device)
        test_label = test_label.to(cfg.device)
        val_label = val_label.to(cfg.device)

        net.to(cfg.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        logger.info(optimizer)

        best_val_acc = 0.0
        train_start_time = time.time()

        for epoch in range(cfg.max_epoch):
            y_train = train_label.unsqueeze(0)
            net.train()

            if cfg.split_image:
                x_part1 = x[:, :, :x.shape[2] // 2+5, :]
                y_part1 = y_train[:,:x.shape[2] // 2+5,:]
                x_part2 = x[:, :, x.shape[2] // 2 - 5: , :]
                y_part2 = y_train[:,x.shape[2] // 2 - 5:,:]
                
                y_pred_part1 = net(x_part1)
                ls1 = head_loss(loss_func, y_pred_part1, y_part1.long())
                optimizer.zero_grad()
                ls1.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                y_pred_part2 = net(x_part2)
                ls2 = head_loss(loss_func, y_pred_part2, y_part2.long())
                optimizer.zero_grad()
                ls2.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                
                logger.info(f'Iter:{epoch}|loss:{(ls1 + ls2).detach().cpu().numpy()}')
            else:
                try:
                    y_pred = net(x)
                    ls = head_loss(loss_func, y_pred, y_train.long())
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()
                    logger.info(f'Iter:{epoch}|loss:{ls.detach().cpu().numpy()}')
                except:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    cfg.split_image = True
                    continue

            # eval
            net.eval()
            with torch.no_grad():
                evaluator.reset()
                
                if cfg.split_image:
                    mid = x.shape[2] // 2
                    x_part1 = x[:, :, :mid+5, :]
                    x_part2 = x[:, :, mid-5:, :]
                    
                    y_pred_part1 = net(x_part1)
                    torch.cuda.empty_cache()
                    y_pred_part2 = net(x_part2)
                    torch.cuda.empty_cache()
                    
                    output_val = torch.cat([y_pred_part1[:, :, :mid, :], y_pred_part2[:, :, 5:, :]], dim=2)
                else:
                    output_val = net(x)
                
                y_val = val_label.unsqueeze(0)
                seg_logits = resize(
                    input=output_val,
                    size=y_val.shape[1:],
                    mode='bilinear',
                    align_corners=True
                )
                predict = torch.argmax(seg_logits, dim=1).cpu().numpy()
                Y_val_np = val_label.cpu().numpy()
                Y_val_255 = np.where(Y_val_np==-1, 255, Y_val_np)
                evaluator.add_batch(np.expand_dims(Y_val_255,axis=0), predict)
                
                OA = evaluator.Pixel_Accuracy()
                mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                Kappa = evaluator.Kappa()
                
                logger.info(f'Evaluate {epoch}|OA:{OA}|MACC:{mAcc}|Kappa:{Kappa}|MIOU:{mIOU}|IOU:{IOU}|ACC:{Acc}')

                # save model
                if OA >= best_val_acc:
                    best_val_acc = OA
                    torch.save(net.state_dict(), save_weight_path)
 
                if (epoch+1) % 50 == 0:
                    save_single_predict_path = os.path.join(save_vis_folder, f'predict_{epoch+1}.png')
                    save_single_gt_path = os.path.join(save_vis_folder, 'gt.png')
                    vis_a_image(gt, predict, save_single_predict_path, save_single_gt_path)

            torch.cuda.empty_cache()

        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        Train_Time_ALL.append(train_time)

        logger.info("\n\n====================Starting evaluation for testing set.========================\n")

        test_start_time = time.time()
        
        best_net = MambaMSF(in_channels=channels, num_classes=class_count, hidden_dim=cfg.hidden_dim)
        best_net.to(cfg.device)
        if os.path.exists(save_weight_path):
            try:
                best_net.load_state_dict(torch.load(save_weight_path, weights_only=True))
            except TypeError:
                 # Fallback for older pytorch versions that don't support weights_only
                best_net.load_state_dict(torch.load(save_weight_path))
        best_net.eval()
        
        test_evaluator = Evaluator(num_class=class_count)
        with torch.no_grad():
            test_evaluator.reset()
            
            if cfg.split_image:
                mid = x.shape[2] // 2
                x_part1 = x[:, :, :mid+5, :]
                x_part2 = x[:, :, mid-5:, :]
                
                y_pred_part1 = best_net(x_part1)
                torch.cuda.empty_cache()
                y_pred_part2 = best_net(x_part2)
                torch.cuda.empty_cache()
                
                output_test = torch.cat([y_pred_part1[:, :, :mid, :], y_pred_part2[:, :, 5:, :]], dim=2)
            else:
                output_test = best_net(x)
                
            y_test = test_label.unsqueeze(0)
            seg_logits_test = resize(
                input=output_test,
                size=y_test.shape[1:],
                mode='bilinear',
                align_corners=True
            )
            predict_test = torch.argmax(seg_logits_test, dim=1).cpu().numpy()
            Y_test_np = test_label.cpu().numpy()
            Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np)
            test_evaluator.add_batch(np.expand_dims(Y_test_255, axis=0), predict_test)
            
            OA_test = test_evaluator.Pixel_Accuracy()
            mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
            mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
            Kappa_test = evaluator.Kappa()
            
            logger.info(f'Test {epoch}|OA:{OA_test}|MACC:{mAcc_test}|Kappa:{Kappa_test}|MIOU:{mIOU_test}|IOU:{IOU_test}|ACC:{Acc_test}')
            vis_a_image(gt, predict_test, predict_save_path, gt_save_path)

        test_end_time = time.time()
        test_time = test_end_time - test_start_time
        Test_Time_ALL.append(test_time)

        with open(results_save_path, 'a+') as f:
            str_results = (f'\n======================'
                         f" exp_idx={exp_idx}"
                         f" seed={curr_seed}"
                         f" learning rate={cfg.lr}"
                         f" epochs={cfg.max_epoch}"
                         f" train ratio={cfg.ratio_list[0]}"
                         f" val ratio={cfg.ratio_list[1]}"
                         f" ======================"
                         f"\nOA={OA_test}"
                         f"\nAA={mAcc_test}"
                         f"\nkpp={Kappa_test}"
                         f"\nmIOU_test:{mIOU_test}"
                         f"\nIOU_test:{IOU_test}"
                         f"\nAcc_test:{Acc_test}\n")
            f.write(str_results)

        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        EACH_ACC_ALL.append(Acc_test)
        
        torch.cuda.empty_cache()

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    logger.info(f"\n====================Mean result of {len(cfg.seed_list)} times runs =========================")
    logger.info(f'List of OA: {list(OA_ALL)}')
    logger.info(f'List of AA: {list(AA_ALL)}')
    logger.info(f'List of KPP: {list(KPP_ALL)}')
    logger.info(f'OA= {round(np.mean(OA_ALL) * 100, 2)} +- {round(np.std(OA_ALL) * 100, 2)}')
    logger.info(f'AA= {round(np.mean(AA_ALL) * 100, 2)} +- {round(np.std(AA_ALL) * 100, 2)}')
    logger.info(f'Kpp= {round(np.mean(KPP_ALL) * 100, 2)} +- {round(np.std(KPP_ALL) * 100, 2)}')
    logger.info(f'Acc per class= {np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)} +- {np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)}')
    logger.info("Average training time=", round(np.mean(Train_Time_ALL), 2), '+-', round(np.std(Train_Time_ALL), 3))
    logger.info("Average testing time=", round(np.mean(Test_Time_ALL) * 1000, 2), '+-',
          round(np.std(Test_Time_ALL) * 1000, 3))

    mean_result_path = os.path.join(save_folder, 'mean_result.txt')
    with open(mean_result_path, 'w') as f:
        str_results = (f'\n\n***************Mean result of {len(cfg.seed_list)} times runs ********************'
                      f'\nList of OA:{list(OA_ALL)}'
                      f'\nList of AA:{list(AA_ALL)}'
                      f'\nList of KPP:{list(KPP_ALL)}'
                      f'\nOA={round(np.mean(OA_ALL) * 100, 2)}+-{round(np.std(OA_ALL) * 100, 2)}'
                      f'\nAA={round(np.mean(AA_ALL) * 100, 2)}+-{round(np.std(AA_ALL) * 100, 2)}'
                      f'\nKpp={round(np.mean(KPP_ALL) * 100, 2)}+-{round(np.std(KPP_ALL) * 100, 2)}'
                      f'\nAcc per class=\n{np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)}+-{np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)}'
                      f'\nAverage training time={np.mean(Train_Time_ALL):.2f}s+-{np.std(Train_Time_ALL):.2f}s'
                      f'\nAverage testing time={np.mean(Test_Time_ALL)*1000:.2f}ms+-{np.std(Test_Time_ALL)*1000:.2f}ms')
        f.write(str_results)

if __name__ == '__main__':
    main()
