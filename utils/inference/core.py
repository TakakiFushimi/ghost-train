from typing import List, Tuple, Callable, Any

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .faceshifter_run import faceshifter_batch
from .image_processing import crop_face, normalize_and_torch, normalize_and_torch_batch
from .video_processing import read_video, crop_frames_and_get_transforms, resize_frames


def transform_target_to_torch(resized_frs: np.ndarray, half=True) -> torch.tensor:
    """
    Transform target, so it could be used by model
    モデルで使用できるようにターゲットを変換する
    """
    target_batch_rs = torch.from_numpy(resized_frs.copy()).cuda()
    target_batch_rs = target_batch_rs[:, :, :, [2,1,0]]/255.
        
    if half:
        target_batch_rs = target_batch_rs.half()
        
    target_batch_rs = (target_batch_rs - 0.5)/0.5 # normalize
    target_batch_rs = target_batch_rs.permute(0, 3, 1, 2)
    
    return target_batch_rs


def model_inference(full_frames: List[np.ndarray],
                    source: List,
                    target: List, 
                    netArc: Callable,
                    G: Callable,
                    app: Callable,
                    set_target: bool,
                    similarity_th=0.15,
                    crop_size=224,
                    BS=60,
                    half=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Using original frames get faceswaped frames and transformations
    オリジナルフレームを使用し、フェイスワップされたフレームとトランスフォーメーションを得る

    Args:
    full_frames (List[np.ndarray]): オリジナルフレームのリスト
    source (List): ソース画像のリスト
    target (List): ターゲット画像のリスト
    netArc (Callable): Arcfaceネットワーク
    G (Callable): 生成モデル
    app (Callable): アプリケーション（顔検出など）
    set_target (bool): ターゲットを設定するかどうか
    similarity_th (float): 類似度の閾値
    crop_size (int): クロップサイズ
    BS (int): バッチサイズ
    half (bool): 半精度（float16）を使用するかどうか
    """
    # Get Arcface embeddings of target image
    # ターゲット画像のArcface埋め込みを取得
    target_norm = normalize_and_torch_batch(np.array(target))
    target_embeds = netArc(F.interpolate(target_norm, scale_factor=0.5, mode='bilinear', align_corners=True))
    
    # Get the cropped faces from original frames and transformations to get those crops
    # オリジナルフレームからクロップされた顔と、そのクロップを得るための変換を取得
    crop_frames_list, tfm_array_list = crop_frames_and_get_transforms(full_frames, target_embeds, app, netArc, crop_size, set_target, similarity_th=similarity_th)
    
    # Normalize source images and transform to torch and get Arcface embeddings
    # ソース画像を正規化し、トーチに変換し、Arcface埋め込みを取得
    source_embeds = []
    for source_curr in source:
        source_curr = normalize_and_torch(source_curr)
        source_embeds.append(netArc(F.interpolate(source_curr, scale_factor=0.5, mode='bilinear', align_corners=True)))
    
    final_frames_list = []
    # 各クロップされたフレームと対応する変換行列、ソース埋め込みを使用してループ処理
    for idx, (crop_frames, tfm_array, source_embed) in enumerate(zip(crop_frames_list, tfm_array_list, source_embeds)):
        # Resize cropped frames and get vector which shows on which frames there were faces
        # クロップされたフレームをリサイズし、顔があったフレームを示すベクトルを取得
        resized_frs, present = resize_frames(crop_frames)
        resized_frs = np.array(resized_frs)

        # Transform embeds of Xs and target frames to use by model
        # Xとターゲットフレームの埋め込みをモデルで使用するために変換
        target_batch_rs = transform_target_to_torch(resized_frs, half=half)

        if half:
            # ソース埋め込みを半精度（float16）に変換
            source_embed = source_embed.half()

        # Run model
        # モデルを実行
        size = target_batch_rs.shape[0]
        model_output = []

        for i in tqdm(range(0, size, BS)):
            # バッチごとにフェイスシフターモデルを実行
            Y_st = faceshifter_batch(source_embed, target_batch_rs[i:i+BS], G)
            model_output.append(Y_st)
        torch.cuda.empty_cache()  # GPUメモリをクリア
        model_output = np.concatenate(model_output)  # モデル出力を結合

        # Create list of final frames with transformed faces
        # 変換された顔の最終フレームのリストを作成
        final_frames = []
        idx_fs = 0

        for pres in tqdm(present):
            if pres == 1:
                final_frames.append(model_output[idx_fs])
                idx_fs += 1
            else:
                final_frames.append([])
        final_frames_list.append(final_frames)
    
    # 最終フレームリスト、クロップフレームリスト、オリジナルフレーム、変換行列リストを返す
    return final_frames_list, crop_frames_list, full_frames, tfm_array_list
