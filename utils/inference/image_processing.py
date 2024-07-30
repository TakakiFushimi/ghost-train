import base64
from io import BytesIO
from typing import Callable, List

import numpy as np
import torch
import cv2
from .masks import face_mask_static 
from matplotlib import pyplot as plt
from insightface.utils import face_align


def crop_face(image_full: np.ndarray, app: Callable, crop_size: int) -> np.ndarray:
    """
    Crop face from image and resize
    画像から顔を切り取り、リサイズする
    """
    kps = app.get(image_full, crop_size)
    M, _ = face_align.estimate_norm(kps[0], crop_size, mode ='None') 
    align_img = cv2.warpAffine(image_full, M, (crop_size, crop_size), borderValue=0.0)         
    return [align_img]


def normalize_and_torch(image: np.ndarray) -> torch.tensor:
    """
    Normalize image and transform to torch
    画像を正規化し、トーチに変換する
    """
    image = torch.tensor(image.copy(), dtype=torch.float32).cuda()
    if image.max() > 1.:
        image = image/255.
    
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = (image - 0.5) / 0.5

    return image


def normalize_and_torch_batch(frames: np.ndarray) -> torch.tensor:
    """
    Normalize batch images and transform to torch
    バッチ画像を正規化し、トーチに変換する
    """
    batch_frames = torch.from_numpy(frames.copy()).cuda()
    if batch_frames.max() > 1.:
        batch_frames = batch_frames/255.
    
    batch_frames = batch_frames.permute(0, 3, 1, 2)
    batch_frames = (batch_frames - 0.5)/0.5

    return batch_frames


def get_final_image(final_frames: List[np.ndarray],
                    crop_frames: List[np.ndarray],
                    full_frame: np.ndarray,
                    tfm_arrays: List[np.ndarray],
                    handler) -> None:
    """
    フェイススワッピングや他の画像変換の結果を用いて最終的な画像を生成する

    Args:
    final_frames (List[np.ndarray]): 最終的に生成されたフレームのリスト
    crop_frames (List[np.ndarray]): 元の画像から切り取られたフレームのリスト
    full_frame (np.ndarray): フルサイズの元のフレーム
    tfm_arrays (List[np.ndarray]): 各フレームに適用された変換行列のリスト
    handler (Callable): ランドマーク検出器
    """

    # フルサイズの元のフレームのコピーを作成
    final = full_frame.copy()
    # 各フレームに対応するパラメータのリスト（初期値はすべてNone）
    params = [None for i in range(len(final_frames))]
    
    # 各フレームについてループ
    for i in range(len(final_frames)):
        # 224x224ピクセルにリサイズされたフレームを取得
        frame = cv2.resize(final_frames[i][0], (224, 224))
        
        # リサイズされたフレームと対応するクロップフレームからランドマークを検出
        landmarks = handler.get_without_detection_without_transform(frame)     
        landmarks_tgt = handler.get_without_detection_without_transform(crop_frames[i][0])

        # face_mask_static関数を使用して、マスクと他の出力を取得
        mask, _ = face_mask_static(crop_frames[i][0], landmarks, landmarks_tgt, params[i])
        # 変換行列を反転
        mat_rev = cv2.invertAffineTransform(tfm_arrays[i][0])

        # リサイズされたフレームとマスクを元のフルサイズのフレームに再変換
        swap_t = cv2.warpAffine(frame, mat_rev, (full_frame.shape[1], full_frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        mask_t = cv2.warpAffine(mask, mat_rev, (full_frame.shape[1], full_frame.shape[0]))
        mask_t = np.expand_dims(mask_t, 2)

        # マスクを使用して変換されたフレームを最終フレームに適用
        final = mask_t*swap_t + (1-mask_t)*final
    
    # 最終フレームをuint8型の配列に変換し、関数の結果として返す
    final = np.array(final, dtype='uint8')
    return final

def get_only_swaped_image(final_frames: List[np.ndarray],
                          crop_frames: List[np.ndarray],
                          full_frame: np.ndarray,
                          tfm_arrays: List[np.ndarray]):
    """
    フェイススワッピングや他の画像変換の結果を用いて最終的な画像を生成する

    Args:
    final_frames (List[np.ndarray]): 最終的に生成されたフレームのリスト
    full_frame (np.ndarray): フルサイズの元のフレーム
    tfm_arrays (List[np.ndarray]): 各フレームに適用された変換行列のリスト
    """

    # フルサイズの元のフレームのコピーを作成
    final = full_frame.copy()
    
    # 各フレームについてループ
    for i in range(len(final_frames)):
        # 224x224ピクセルにリサイズされたフレームを取得
        frame = cv2.resize(final_frames[i][0], (224, 224))
        
        # 変換行列を反転
        mat_rev = cv2.invertAffineTransform(tfm_arrays[i][0])

        # リサイズされたフレームとマスクを元のフルサイズのフレームに再変換
        swap_t = cv2.warpAffine(frame, mat_rev, (full_frame.shape[1], full_frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)

        # マスクを使用して変換されたフレームを最終フレームに適用
        final = swap_t
    
    # 最終フレームをuint8型の配列に変換し、関数の結果として返す
    final = np.array(final, dtype='uint8')
    return final

def show_images(images: List[np.ndarray], 
                titles=None, 
                figsize=(20, 5), 
                fontsize=15):
    """
    Display images with optional titles
    オプションのタイトルと共に画像を表示する
    """
    if titles:
        assert len(titles) == len(images), "Amount of images should be the same as the amount of titles"
    
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for idx, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image[:, :, ::-1])
        if titles:
            ax.set_title(titles[idx], fontsize=fontsize)
        ax.axis("off")