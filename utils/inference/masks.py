import numpy as np
import cv2


def expand_eyebrows(lmrks, eyebrows_expand_mod=1.0):
    """
    Expand the eyebrows based on the given modification factor
    眉毛を指定された修正係数に基づいて拡張する

    Args:
    lmrks (np.ndarray): 顔のランドマークの配列
    eyebrows_expand_mod (float): 眉毛を拡張するための修正係数
    """
    
    lmrks = np.array(lmrks.copy(), dtype=np.int32)

    # Top of the eye arrays
    # 目の上部の配列
    bot_l = lmrks[[35, 41, 40, 42, 39]]
    bot_r = lmrks[[89, 95, 94, 96, 93]]

    # Eyebrow arrays
    # 眉毛の配列
    top_l = lmrks[[43, 48, 49, 51, 50]]
    top_r = lmrks[[102, 103, 104, 105, 101]]

    # Adjust eyebrow arrays
    # 眉毛の配列を調整
    lmrks[[43, 48, 49, 51, 50]] = top_l + eyebrows_expand_mod * 0.5 * (top_l - bot_l)
    lmrks[[102, 103, 104, 105, 101]] = top_r + eyebrows_expand_mod * 0.5 * (top_r - bot_r)
    return lmrks


def get_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Get face mask of image size using given landmarks of person
    与えられた人物のランドマークを使用して、画像サイズのフェイスマスクを取得する

    Args:
    image (np.ndarray): 入力画像
    landmarks (np.ndarray): 顔のランドマークの配列
    """

    # 入力画像をグレースケールに変換
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # グレースケール画像と同じサイズのゼロ初期化されたマスクを作成
    mask = np.zeros_like(img_gray)

    # ランドマークポイントを整数型のNumPy配列に変換
    points = np.array(landmarks, np.int32)
    
    # ランドマークポイントの凸包（外接する最小の凸多角形）を計算
    convexhull = cv2.convexHull(points)
    
    # マスク画像内の凸包領域を白（255）で塗りつぶし
    cv2.fillConvexPoly(mask, convexhull, 255)
    
    # 生成されたマスクを返却
    return mask


def face_mask_static(image: np.ndarray, landmarks: np.ndarray, landmarks_tgt: np.ndarray, params = None) -> np.ndarray:
    """
    Get the final mask, using landmarks and applying blur
    ランドマークを使用して最終マスクを取得し、ぼかしを適用する

    Args:
    image (np.ndarray): 入力画像
    landmarks (np.ndarray): 入力画像のランドマークの配列
    landmarks_tgt (np.ndarray): 目標画像のランドマークの配列
    params (list, optional): ぼかしと侵食のパラメータのリスト。デフォルトは None。
    """
    if params is None:
        # 左側のランドマークの位置差を計算
        left = np.sum((landmarks[1][0]-landmarks_tgt[1][0], 
                       landmarks[2][0]-landmarks_tgt[2][0], 
                       landmarks[13][0]-landmarks_tgt[13][0]))
        
        # 右側のランドマークの位置差を計算
        right = np.sum((landmarks_tgt[17][0]-landmarks[17][0], 
                        landmarks_tgt[18][0]-landmarks[18][0], 
                        landmarks_tgt[29][0]-landmarks[29][0]))
        
        # 最大オフセットの決定
        offset = max(left, right)
        
        # オフセットに基づいて侵食とぼかしのパラメータを設定
        if offset > 6:
            erode = 15
            sigmaX = 15
            sigmaY = 10
        elif offset > 3:
            erode = 10
            sigmaX = 10
            sigmaY = 8
        elif offset < -3:
            erode = -5
            sigmaX = 5
            sigmaY = 10
        else:
            erode = 5
            sigmaX = 5
            sigmaY = 5
    else:
        # paramsが指定されている場合、その値を使用
        erode = params[0]
        sigmaX = params[1]
        sigmaY = params[2]
    
    # 侵食のパラメータに基づいて眉毛の拡張モジュールを設定
    if erode == 15:
        eyebrows_expand_mod = 2.7
    elif erode == -5:
        eyebrows_expand_mod = 0.5
    else:
        eyebrows_expand_mod = 2.0

    # 眉毛を拡張してランドマークを更新
    landmarks = expand_eyebrows(landmarks, eyebrows_expand_mod=eyebrows_expand_mod)
    
    # 入力画像と更新されたランドマークに基づいてマスクを生成
    mask = get_mask(image, landmarks)
    
    # 生成されたマスクに侵食とぼかしを適用
    mask = erode_and_blur(mask, erode, sigmaX, sigmaY, True)
    
    # マスクを255で割り、値を0から1の範囲に正規化して返す
    if params is None:
        return mask / 255, [erode, sigmaX, sigmaY]
        
    return mask / 255


def erode_and_blur(mask_input, erode, sigmaX, sigmaY, fade_to_border=True):
    """
    Erode and blur the mask based on given parameters
    与えられたパラメータに基づいてマスクを侵食およびぼかしを適用する

    Args:
    mask_input (np.ndarray): 入力マスク
    erode (int): 侵食のピクセル数
    sigmaX (float): X軸方向のぼかしシグマ値
    sigmaY (float): Y軸方向のぼかしシグマ値
    fade_to_border (bool): 境界へのフェードを適用するかどうか
    """
    mask = np.copy(mask_input)
    
    if erode > 0:
        kernel = np.ones((erode, erode), 'uint8')
        mask = cv2.erode(mask, kernel, iterations=1)
    else:
        kernel = np.ones((-erode, -erode), 'uint8')
        mask = cv2.dilate(mask, kernel, iterations=1)
        
    if fade_to_border:
        clip_size = sigmaY * 2
        mask[:clip_size, :] = 0
        mask[-clip_size:, :] = 0
        mask[:, :clip_size] = 0
        mask[:, -clip_size:] = 0
    
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigmaX, sigmaY=sigmaY)
        
    return mask