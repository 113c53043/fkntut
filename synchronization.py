import cv2
import numpy as np
import torch
from PIL import Image
import math

class SyncModule:
    def __init__(self, shape=(512, 512), marker_size=32, alpha=0.5):
        """
        [升級版同步模組]
        marker_size: 增大至 32 以抵抗模糊
        alpha: 增大至 0.5 以確保可見性
        """
        self.H, self.W = shape
        self.marker_size = marker_size
        self.alpha = alpha
        self.marker_template = self._create_marker_pattern()

    def _create_marker_pattern(self):
        s = self.marker_size
        marker = np.zeros((s, s), dtype=np.float32)
        center = (s // 2, s // 2)
        # 繪製更粗、對比度更高的同心圓
        cv2.circle(marker, center, s // 2 - 2, 1.0, -1) # 外圓 (白)
        cv2.circle(marker, center, s // 4 + 1, 0.0, -1) # 內圓 (黑)
        return marker

    def add_markers(self, img_tensor):
        img_np = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
        img_out = img_np.copy()
        s = self.marker_size
        
        # [關鍵修改] 增加邊距 (Margin)，防止旋轉時標記被切掉
        # 假設最大旋轉 15 度，邊角可能會損失約 10% 的寬度
        margin = 32 
        
        positions = [
            (margin, margin),                         # Top-Left
            (self.W - s - margin, margin),            # Top-Right
            (margin, self.H - s - margin),            # Bottom-Left
            (self.W - s - margin, self.H - s - margin)# Bottom-Right
        ]
        
        for x, y in positions:
            roi = img_out[y:y+s, x:x+s, :]
            marker_3c = np.stack([self.marker_template]*3, axis=2)
            # Alpha Blending
            blended = roi * (1 - self.alpha) + marker_3c * self.alpha
            img_out[y:y+s, x:x+s, :] = blended

        res_tensor = torch.from_numpy(img_out.transpose(2, 0, 1)).float().unsqueeze(0)
        return torch.clamp(res_tensor, 0.0, 1.0)

    def _check_geometric_consistency(self, src_pts, dst_pts):
        """幾何一致性檢查"""
        try:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        except:
            return False, None

        # 檢查旋轉縮放矩陣的行列式
        # 正常的旋轉+縮放，其行列式應該接近 1 (因為我們沒有大幅縮放)
        # 如果變形太誇張，determinant 會很奇怪
        det = np.linalg.det(M[:2, :2])
        
        # 放寬條件，只要不是極端值都接受
        if det < 0.2 or det > 5.0: return False, None 

        return True, M

    def align_image(self, img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: raise ValueError(f"Cannot load: {img_path}")
            
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        template = (self.marker_template * 255).astype(np.uint8)
        
        # [關鍵修改] 分區搜尋策略 (Quadrant-based Search)
        # 不在全圖搜，而是只在四個角落的區域搜，取該區域的「最大值」
        # 這樣即使分數只有 0.4 (因為旋轉模糊)，只要它是該區域最高的，我們就抓得到
        
        h, w = img_gray.shape
        half_h, half_w = h // 2, w // 2
        
        # 定義四個搜索區域 (ROIs)
        rois = {
            'tl': (0, 0, half_w, half_h),
            'tr': (half_w, 0, w, half_h),
            'bl': (0, half_h, half_w, h),
            'br': (half_w, half_h, w, h)
        }
        
        # 預期目標座標 (加上標記中心偏移)
        s = self.marker_size
        m = 32 # 必須與 add_markers 的 margin 一致
        expected_pos = {
            'tl': (m, m), 
            'tr': (self.W - s - m, m),
            'bl': (m, self.H - s - m), 
            'br': (self.W - s - m, self.H - s - m)
        }

        src_pts_list = []
        dst_pts_list = []
        found_count = 0
        
        # 匹配閾值 (降低以容忍旋轉模糊)
        threshold = 0.35 

        for key, (x1, y1, x2, y2) in rois.items():
            roi_img = img_gray[y1:y2, x1:x2]
            
            # 在 ROI 內匹配
            res = cv2.matchTemplate(roi_img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val >= threshold:
                # max_loc 是相對於 ROI 的座標，需要轉回全圖座標
                global_x = x1 + max_loc[0]
                global_y = y1 + max_loc[1]
                
                # 這裡 matchTemplate 回傳的是左上角，轉為中心點比較準
                # 但只要前後一致即可，這裡維持左上角
                src_pts_list.append([global_x, global_y])
                
                ex, ey = expected_pos[key]
                dst_pts_list.append([ex, ey])
                found_count += 1
            # else:
            #     print(f"  Startpoint {key} not found (max_val={max_val:.2f})")

        # 至少需要 4 個點才能做透視變換 (Homography)
        # 如果只有 3 個點，可以用仿射變換 (Affine)，但魯棒性稍差
        if found_count < 4:
            # print("  ⚠️ Sync: Less than 4 markers found, skipping alignment.")
            return Image.open(img_path).convert("RGB")

        src_pts = np.float32(src_pts_list)
        dst_pts = np.float32(dst_pts_list)
        
        # 幾何檢查
        is_valid, M = self._check_geometric_consistency(src_pts, dst_pts)
        
        if not is_valid:
            return Image.open(img_path).convert("RGB")

        # 執行校正
        aligned_img = cv2.warpPerspective(img_bgr, M, (self.W, self.H))
        
        return Image.fromarray(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))