# shape_deca.py

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from scipy.spatial import procrustes

# DECA モデルを初期化
def initialize_deca(device):
    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = 'pytorch3d'
    deca_cfg.model.extract_tex = True
    deca_cfg.iscrop = True
    deca_cfg.render_orig = False
    deca = DECA(config=deca_cfg, device=device)
    return deca

# .obj データのメッシュ情報を取得
def get_mesh_data(deca, image_tensor):
    with torch.no_grad():
        # エンコードされたデータを取得
        codedict = deca.encode(image_tensor)
        # デコード処理を実行してメッシュ情報を取得
        opdict = deca.decode(codedict, return_vis=False)
        # デコードされたデータから頂点情報を取得
        obj_data = opdict['verts']  # 頂点座標 (vertices) を取得
    return obj_data

# Procrustes ロスを計算
def compute_procrustes_loss(source_obj_data, generated_obj_data):
    source_vertices = source_obj_data[0]  # (num_vertices, 3) の形状にする
    generated_vertices = generated_obj_data[0]  # 同様に (num_vertices, 3)
    # Procrustes 解析でメッシュの位置合わせを行い、形状の違いを計算
    _, _, disparity = procrustes(source_vertices.cpu().numpy(), generated_vertices.cpu().numpy())
    return disparity

# サンプル画像をロードし、テンソルに変換
def load_and_preprocess_image(image_path, device):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 224x224にリサイズして余裕を持たせる
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  # バッチサイズ1のテンソルに変換
    return image_tensor

# デモ関数
def demo_procrustes_loss(device, image_path):
    # DECAモデルを初期化
    deca = initialize_deca(device)

    # サンプル画像をロード
    image_tensor = load_and_preprocess_image(image_path, device)

    # メッシュデータを取得
    source_mesh_data = get_mesh_data(deca, image_tensor)
    generated_mesh_data = get_mesh_data(deca, image_tensor)  # 今回は同じ画像でテスト
    
    # Procrustesロスを計算
    disparity = compute_procrustes_loss(source_mesh_data, generated_mesh_data)
    print(f'Procrustes Loss: {disparity}')


