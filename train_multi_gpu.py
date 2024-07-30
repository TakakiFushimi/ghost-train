# imports
print("started imports")

import sys
import argparse
import time
import cv2
import wandb
from PIL import Image
import os

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as scheduler

# custom imports
sys.path.append('./apex/')

from apex import amp
from network.AEI_Net import *
from network.MultiscaleDiscriminator import *
from utils.training.Dataset import FaceEmbedVGG2, FaceEmbed
from utils.training.image_processing import make_image_list, get_faceswap
from utils.training.losses import hinge_loss, compute_discriminator_loss, compute_generator_losses
#from utils.training.detector import detect_landmarks, paint_eyes
#from AdaptiveWingLoss.core import models
from arcface_model.iresnet import iresnet100

print("finished imports")

def train_one_epoch(G: AEI_Net, 
                    D: MultiscaleDiscriminator, 
                    opt_G: torch.optim.Optimizer, 
                    opt_D: torch.optim.Optimizer,
                    scheduler_G: torch.optim.lr_scheduler._LRScheduler,
                    scheduler_D: torch.optim.lr_scheduler._LRScheduler,
                    netArc: iresnet100,
                    model_ft ,
                    args: argparse.Namespace,
                    dataloader: torch.utils.data.DataLoader,
                    device: torch.device,
                    epoch: int,
                    loss_adv_accumulated: int):
    """
    Train the models for one epoch
    一つのエポックの間モデルをトレーニングする
    """
    
    for iteration, data in enumerate(dataloader):
        start_time = time.time()
        
        Xs_orig, Xs, Xt, same_person = data

        Xs_orig = Xs_orig.to(device)
        Xs = Xs.to(device)
        Xt = Xt.to(device)
        same_person = same_person.to(device)

        # get the identity embeddings of Xs
        # Xsのアイデンティティ埋め込みを取得する
        with torch.no_grad():
            embed = netArc(F.interpolate(Xs_orig, [112, 112], mode='bilinear', align_corners=False))

        diff_person = torch.ones_like(same_person)

        if args.diff_eq_same:
            same_person = diff_person
    
        # generator training
        # ジェネレータのトレーニング
        opt_G.zero_grad()
        
        Y, Xt_attr = G(Xt, embed)
        Di = D(Y)
        ZY = netArc(F.interpolate(Y, [112, 112], mode='bilinear', align_corners=False))
        
        if args.eye_detector_loss:
            Xt_eyes, Xt_heatmap_left, Xt_heatmap_right = detect_landmarks(Xt, model_ft)
            Y_eyes, Y_heatmap_left, Y_heatmap_right = detect_landmarks(Y, model_ft)
            eye_heatmaps = [Xt_heatmap_left, Xt_heatmap_right, Y_heatmap_left, Y_heatmap_right]
        else:
            eye_heatmaps = None
            
        # ジェネレータの損失を計算
        lossG, loss_adv_accumulated, L_adv, L_attr, L_id, L_rec, L_l2_eyes = compute_generator_losses(G, Y, Xt, Xt_attr, Di,
                                                                             embed, ZY, eye_heatmaps,loss_adv_accumulated, 
                                                                             diff_person, same_person, args)
        
        # 損失をスケールしてバックプロパゲーション
        with amp.scale_loss(lossG, opt_G) as scaled_loss:
            scaled_loss.backward()
        opt_G.step()
        if args.scheduler:
            scheduler_G.step()
        
        # discriminator training
        # ディスクリミネータのトレーニング
        opt_D.zero_grad()
        lossD = compute_discriminator_loss(D, Y, Xs, diff_person)
        with amp.scale_loss(lossD, opt_D) as scaled_loss:
            scaled_loss.backward()

        if (not args.discr_force) or (loss_adv_accumulated < 4.):
            opt_D.step()
        if args.scheduler:
            scheduler_D.step()
        
        
        batch_time = time.time() - start_time

        # 生成された画像を表示または保存
        if iteration % args.show_step == 0:
            images = [Xs, Xt, Y]
            if args.eye_detector_loss:
                Xt_eyes_img = paint_eyes(Xt, Xt_eyes)
                Yt_eyes_img = paint_eyes(Y, Y_eyes)
                images.extend([Xt_eyes_img, Yt_eyes_img])
            image = make_image_list(images)
            if args.use_wandb:
                wandb.log({"gen_images":wandb.Image(image, caption=f"{epoch:03}" + '_' + f"{iteration:06}")})
            else:
                cv2.imwrite('./images/generated_image.jpg', image[:,:,::-1])
        
        # 損失とトレーニング時間を出力
        if iteration % 10 == 0:
            print(f'epoch: {epoch}    {iteration} / {len(dataloader)}')
            print(f'lossD: {lossD.item()}    lossG: {lossG.item()} batch_time: {batch_time}s')
            print(f'L_adv: {L_adv.item()} L_id: {L_id.item()} L_attr: {L_attr.item()} L_rec: {L_rec.item()}')
            if args.eye_detector_loss:
                print(f'L_l2_eyes: {L_l2_eyes.item()}')
            print(f'loss_adv_accumulated: {loss_adv_accumulated}')
            if args.scheduler:
                print(f'scheduler_G lr: {scheduler_G.get_last_lr()} scheduler_D lr: {scheduler_D.get_last_lr()}')
        
        # 損失をWandBにログ
        if args.use_wandb:
            if args.eye_detector_loss:
                wandb.log({"loss_eyes": L_l2_eyes.item()}, commit=False)
            wandb.log({"loss_id": L_id.item(),
                       "lossD": lossD.item(),
                       "lossG": lossG.item(),
                       "loss_adv": L_adv.item(),
                       "loss_attr": L_attr.item(),
                       "loss_rec": L_rec.item()})
        
        # モデルの状態を保存
        if iteration % 5000 == 0:
            torch.save(G.state_dict(), f'./saved_models_{args.run_name}/G_latest.pth')
            torch.save(D.state_dict(), f'./saved_models_{args.run_name}/D_latest.pth')

            torch.save(G.state_dict(), f'./current_models_{args.run_name}/G_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')
            torch.save(D.state_dict(), f'./current_models_{args.run_name}/D_' + str(epoch)+ '_' + f"{iteration:06}" + '.pth')

        # 特定のステップでフェイススワップの結果を表示
        if (iteration % 250 == 0) and (args.use_wandb):
            G.eval()

            res1 = get_faceswap('examples/images/training//source1.png', 'examples/images/training//target1.png', G, netArc, device)
            res2 = get_faceswap('examples/images/training//source2.png', 'examples/images/training//target2.png', G, netArc, device)  
            res3 = get_faceswap('examples/images/training//source3.png', 'examples/images/training//target3.png', G, netArc, device)
            
            res4 = get_faceswap('examples/images/training//source4.png', 'examples/images/training//target4.png', G, netArc, device)
            res5 = get_faceswap('examples/images/training//source5.png', 'examples/images/training//target5.png', G, netArc, device)  
            res6 = get_faceswap('examples/images/training//source6.png', 'examples/images/training//target6.png', G, netArc, device)
            
            output1 = np.concatenate((res1, res2, res3), axis=0)
            output2 = np.concatenate((res4, res5, res6), axis=0)
            
            output = np.concatenate((output1, output2), axis=1)

            wandb.log({"our_images":wandb.Image(output, caption=f"{epoch:03}" + '_' + f"{iteration:06}")})

            G.train()


def train(args, device):
    """
    Train the models
    モデルをトレーニングする
    """
    # training params
    # トレーニングパラメータ
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    
    # initializing main models
    # メインモデルの初期化
    G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512).to(device)
    D = MultiscaleDiscriminator(input_nc=3, n_layers=5, norm_layer=torch.nn.InstanceNorm2d).to(device)
    
    G.train()
    D.train()
    
    # initializing model for identity extraction
    # アイデンティティ抽出モデルの初期化
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
    netArc=netArc.cuda()
    netArc.eval()
    
    if args.eye_detector_loss:
        model_ft = models.FAN(4, "False", "False", 98)
        checkpoint = torch.load('./AdaptiveWingLoss/AWL_detector/WFLW_4HG.pth')
        if 'state_dict' not in checkpoint:
            model_ft.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = model_ft.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}
            model_weights.update(pretrained_weights)
            model_ft.load_state_dict(model_weights)
        model_ft = model_ft.to(device)
        model_ft.eval()
    else:
        model_ft = None
    
    # オプティマイザの初期化
    opt_G = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0, 0.999), weight_decay=1e-4)
    opt_D = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0, 0.999), weight_decay=1e-4)


    # AMPの初期化
    G, opt_G = amp.initialize(G, opt_G, opt_level=args.optim_level)
    D, opt_D = amp.initialize(D, opt_D, opt_level=args.optim_level)

    # 複数のGPUを使用するためにモデルをDataParallelでラップ
    if torch.cuda.device_count() >= 2:
        print(f"Using GPUs: {0, 1}")
        G = nn.DataParallel(G, device_ids=[0, 1])
        D = nn.DataParallel(D, device_ids=[0, 1])
        netArc = nn.DataParallel(netArc, device_ids=[0, 1])

    # スケジューラの設定
    if args.scheduler:
        scheduler_G = optim.lr_scheduler.StepLR(opt_G, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
        scheduler_D = optim.lr_scheduler.StepLR(opt_D, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    else:
        scheduler_G = None
        scheduler_D = None
        
    # 事前トレーニングされた重みをロード
    if args.pretrained:
        try:
            G.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')), strict=False)
            D.load_state_dict(torch.load(args.D_path, map_location=torch.device('cpu')), strict=False)
            print("Loaded pretrained weights for G and D")
        except FileNotFoundError as e:
            print("Not found pretrained weights. Continue without any pretrained weights.")
    
    # データセットの選択
    if args.vgg:
        dataset = FaceEmbedVGG2(args.dataset_path, same_prob=args.same_person, same_identity=args.same_identity)
    else:
        dataset = FaceEmbed([args.dataset_path], same_prob=args.same_person)
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    # Accumulated adversarial loss for training discriminator only when below threshold
    # 一定の閾値以下の時にのみディスクリミネータをトレーニングするための累積敵対的損失
    loss_adv_accumulated = 20.
    
    # エポックごとにトレーニングを実行
    for epoch in range(0, max_epoch):
        train_one_epoch(G,
                        D,
                        opt_G,
                        opt_D,
                        scheduler_G,
                        scheduler_D,
                        netArc,
                        model_ft,
                        args,
                        dataloader,
                        device,
                        epoch,
                        loss_adv_accumulated)

def main(args):
    """
    Main function to start training
    トレーニングを開始するメイン関数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用するデバイスを明示的に設定
    if not torch.cuda.is_available():
        print('CUDA is not available. Using CPU.')
    else:
        print(f'Using CUDA device: {device}')
    
    print("Starting training")
    train(args, device=device)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # dataset params
    # データセットパラメータ
    parser.add_argument('--dataset_path', default='./dataset/VGG-Face2-crop/', help='Path to the dataset. If not VGG2 dataset is used, param --vgg should be set False')
    # データセットへのパス。VGG2データセットを使用しない場合、--vggパラメータをFalseに設定する必要があります。
    parser.add_argument('--G_path', default='./saved_models/G.pth', help='Path to pretrained weights for G. Only used if pretrained=True')
    # Gの事前学習済み重みのパス。pretrained=Trueの場合にのみ使用されます。
    parser.add_argument('--D_path', default='./saved_models/D.pth', help='Path to pretrained weights for D. Only used if pretrained=True')
    # Dの事前学習済み重みのパス。pretrained=Trueの場合にのみ使用されます。
    parser.add_argument('--vgg', default=True, type=bool, help='When using VGG2 dataset (or any other dataset with several photos for one identity)')
    # VGG2データセット（または同一人物の複数の写真がある他のデータセット）を使用する場合にTrueに設定します。
    
    # weights for loss
    # 損失の重み
    parser.add_argument('--weight_adv', default=1, type=float, help='Adversarial Loss weight')
    # 敵対的損失の重み
    parser.add_argument('--weight_attr', default=10, type=float, help='Attributes weight')
    # 属性損失の重み
    parser.add_argument('--weight_id', default=15, type=float, help='Identity Loss weight')
    # アイデンティティ損失の重み
    parser.add_argument('--weight_rec', default=10, type=float, help='Reconstruction Loss weight')
    # 再構成損失の重み
    parser.add_argument('--weight_eyes', default=0., type=float, help='Eyes Loss weight')
    # 目の損失の重み

    # training params you may want to change
    # 変更する可能性があるトレーニングパラメータ
    parser.add_argument('--backbone', default='unet', const='unet', nargs='?', choices=['unet', 'linknet', 'resnet'], help='Backbone for attribute encoder')
    # 属性エンコーダのバックボーン
    parser.add_argument('--num_blocks', default=2, type=int, help='Numbers of AddBlocks at AddResblock')
    # AddResblockでのAddBlocksの数
    parser.add_argument('--same_person', default=0.2, type=float, help='Probability of using same person identity during training')
    # トレーニング中に同一人物のアイデンティティを使用する確率
    parser.add_argument('--same_identity', default=True, type=bool, help='Using simswap approach, when source_id = target_id. Only possible with vgg=True')
    # ソースIDとターゲットIDが同じ場合にsimswapアプローチを使用する。vgg=Trueの場合にのみ可能
    parser.add_argument('--diff_eq_same', default=False, type=bool, help='Don\'t use info about where is defferent identities')
    # 異なるアイデンティティの情報を使用しない
    parser.add_argument('--pretrained', default=False, type=bool, help='If using the pretrained weights for training or not')
    # トレーニングに事前学習済み重みを使用するかどうか
    parser.add_argument('--discr_force', default=False, type=bool, help='If True Discriminator would not train when adversarial loss is high')
    # Trueの場合、敵対的損失が高いときにディスクリミネータをトレーニングしない
    parser.add_argument('--scheduler', default=False, type=bool, help='If True decreasing LR is used for learning of generator and discriminator')
    # Trueの場合、ジェネレータとディスクリミネータの学習に減少する学習率を使用する
    parser.add_argument('--scheduler_step', default=5000, type=int)
    # スケジューラのステップサイズ
    parser.add_argument('--scheduler_gamma', default=0.2, type=float, help='It is value, which shows how many times to decrease LR')
    # 学習率をどれだけ減少させるかを示す値
    parser.add_argument('--eye_detector_loss', default=False, type=bool, help='If True eye loss with using AdaptiveWingLoss detector is applied to generator')
    # Trueの場合、AdaptiveWingLoss検出器を使用してジェネレータに目の損失を適用する

    # info about this run
    # この実行に関する情報
    parser.add_argument('--use_wandb', default=True, type=bool, help='Use wandb to track your experiments or not')
    # wandbを使用して実験を追跡するかどうか
    parser.add_argument('--run_name', required=True, type=str, help='Name of this run. Used to create folders where to save the weights.')
    # この実行の名前。重みを保存するフォルダの作成に使用されます。
    parser.add_argument('--wandb_project', default='your-project-name', type=str)
    # wandbプロジェクトの名前
    parser.add_argument('--wandb_entity', default='fushissitakaki-japan', type=str)
    # wandbのエンティティ（ログイン名）

    # training params you probably don't want to change
    # 変更しない可能性が高いトレーニングパラメータ
    parser.add_argument('--batch_size', default=16, type=int)
    # バッチサイズ
    parser.add_argument('--lr_G', default=4e-4, type=float)
    # ジェネレータの学習率
    parser.add_argument('--lr_D', default=4e-4, type=float)
    # ディスクリミネータの学習率
    parser.add_argument('--max_epoch', default=2000, type=int)
    # 最大エポック数
    parser.add_argument('--show_step', default=500, type=int)
    # 結果を表示するステップ数
    parser.add_argument('--save_epoch', default=1, type=int)
    # モデルを保存するエポック数
    parser.add_argument('--optim_level', default='O2', type=str)
    # 最適化レベル

    args = parser.parse_args()
    
    if args.vgg == False and args.same_identity == True:
        raise ValueError("Sorry, you can't use some other dataset than VGG2 Faces with param same_identity=True")
    # vggがFalseでsame_identityがTrueの場合、VGG2 Faces以外のデータセットを使用することはできません
    
    if args.use_wandb == True:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, settings=wandb.Settings(start_method='fork'))

        config = wandb.config
        config.dataset_path = args.dataset_path
        config.weight_adv = args.weight_adv
        config.weight_attr = args.weight_attr
        config.weight_id = args.weight_id
        config.weight_rec = args.weight_rec
        config.weight_eyes = args.weight_eyes
        config.same_person = args.same_person
        config.Vgg2Face = args.vgg
        config.same_identity = args.same_identity
        config.diff_eq_same = args.diff_eq_same
        config.discr_force = args.discr_force
        config.scheduler = args.scheduler
        config.scheduler_step = args.scheduler_step
        config.scheduler_gamma = args.scheduler_gamma
        config.eye_detector_loss = args.eye_detector_loss
        config.pretrained = args.pretrained
        config.run_name = args.run_name
        config.G_path = args.G_path
        config.D_path = args.D_path
        config.batch_size = args.batch_size
        config.lr_G = args.lr_G
        config.lr_D = args.lr_D
    elif not os.path.exists('./images'):
        os.mkdir('./images')
    
    # モデルの最新の重みを保存するためのフォルダを作成
    if not os.path.exists(f'./saved_models_{args.run_name}'):
        os.mkdir(f'./saved_models_{args.run_name}')
        os.mkdir(f'./current_models_{args.run_name}')
    
    main(args)
