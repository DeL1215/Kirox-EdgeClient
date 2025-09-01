# -*- coding: utf-8 -*-
# 目標：同一份程式在「一般電腦」與「Jetson Nano」都能即時偵測 person
# 特點：
#  - --model 同時支援 .pt（PyTorch）與 .engine（TensorRT）
#  - 預設 headless：終端顯示 FPS；加 --show 可開視窗
#  - 只顯示/輸出 person（classes=[0]）
#  - 支援 USB/內建相機、CSI（Jetson）、RTSP（優先 GStreamer）
#  - imgsz 預設 416（Nano 可依性能改 384/320）
#
# 使用範例：
#   電腦(USB相機)：python3 unified_person_rt.py --source 0 --model /abs/path/yolov8n.pt
#   Jetson-CSI   ：python3 unified_person_rt.py --csi --model /abs/path/yolov8n.engine
#   RTSP         ：python3 unified_person_rt.py --rtsp rtsp://user:pass@ip:554/stream --model /abs/path/yolov8n.pt
#
# 備註：
#   1) .engine 檔請在 Jetson Nano 本機匯出（避免版本不合）。
#   2) .pt 在電腦可直接跑（若有 CUDA 則走 GPU），Nano 上也能跑但較慢。

import os
import time
import argparse
import cv2

# 可能沒有裝 torch（若只跑 TensorRT 亦可），因此用 try/except
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO 的 person 類別為 0

# -----------------------------
# GStreamer Pipelines（Jetson/RTSP）
# -----------------------------
def csi_gstreamer_pipeline(width=640, height=480, fps=30, flip=0):
    """
    Jetson CSI 相機 GStreamer（使用 NVMM 加速）
    """
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
    )

def rtsp_gstreamer_pipeline(rtsp_url, width=640, height=480, latency=200, use_hw_decode=False):
    """
    RTSP 硬體/軟體解碼的 GStreamer pipeline
    Jetson 可嘗試將 avdec_h264 換成 nvv4l2decoder（若環境支援）
    """
    decoder = "nvv4l2decoder" if use_hw_decode else "avdec_h264"
    return (
        f"rtspsrc location={rtsp_url} latency={latency} ! "
        f"rtph264depay ! h264parse ! {decoder} ! "
        f"videoconvert ! videoscale ! video/x-raw, width={width}, height={height}, format=BGR ! "
        f"appsink drop=true sync=false"
    )

# -----------------------------
# 視訊來源開啟
# -----------------------------
def open_capture(args):
    # CSI（Jetson）
    if args.csi:
        pipe = csi_gstreamer_pipeline(args.width, args.height, args.fps, args.flip)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        return cap

    # RTSP
    if args.rtsp:
        # 先嘗試 GStreamer（硬體解碼選項可依需要改 True）
        pipe = rtsp_gstreamer_pipeline(args.rtsp, args.width, args.height, args.latency, use_hw_decode=False)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            # 後備：不用 GStreamer（相容性較高、效能可能較差）
            cap = cv2.VideoCapture(args.rtsp)
        return cap

    # USB / 內建攝影機 / 檔案路徑
    # 若 --source 是數字，當作相機 index；否則當作影片檔路徑
    source = args.source
    index = None
    try:
        index = int(source)
    except ValueError:
        index = None

    if index is not None:
        cap = cv2.VideoCapture(index)
        # 嘗試設定期望的擷取參數（有些攝影機/平台可能不生效）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        return cap
    else:
        # 影片檔
        cap = cv2.VideoCapture(source)
        return cap

# -----------------------------
# 繪製 person 邊框
# -----------------------------
def draw_person_boxes(frame, result, color=(0, 255, 0)):
    """
    只繪製偵測為 person 的框與信心分數
    """
    annotated = frame
    names = result.names
    # Ultralytics v8 統一用 result.boxes 存 BBoxes
    for b in result.boxes:
        cls_id = int(b.cls[0].item())
        if cls_id != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        conf = float(b.conf[0].item())
        label = f"person {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated

# -----------------------------
# 主流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="模型路徑（支援 .pt 或 .engine）")
    parser.add_argument("--source", type=str, default="0",
                        help="USB 相機索引、影片路徑；搭配 --csi 或 --rtsp 時忽略")
    parser.add_argument("--csi", action="store_true", help="使用 CSI 相機（Jetson）")
    parser.add_argument("--rtsp", type=str, default="", help="RTSP URL")
    parser.add_argument("--width", type=int, default=640, help="擷取寬度")
    parser.add_argument("--height", type=int, default=480, help="擷取高度")
    parser.add_argument("--fps", type=int, default=30, help="相機 FPS 設定")
    parser.add_argument("--flip", type=int, default=0, help="CSI 相機鏡像旋轉（0~7）")
    parser.add_argument("--latency", type=int, default=200, help="RTSP 延遲（毫秒）")
    parser.add_argument("--imgsz", type=int, default=416, help="YOLO 推論輸入尺寸（邊長）")
    parser.add_argument("--conf", type=float, default=0.5, help="信心閾值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 閾值")
    parser.add_argument("--show", action="store_true", help="顯示視窗（預設不顯示）")
    parser.add_argument("--device", type=str, default="", choices=["cpu", "cuda", ""],
                        help="覆寫推論裝置：cpu/cuda（空字串=自動偵測）")
    args = parser.parse_args()

    # 檔案存在檢查
    if not os.path.exists(args.model):
        print(f"[錯誤] 找不到模型檔：{args.model}")
        print("請確認路徑是否正確，或指定 --model /abs/path/to/yolov8n.pt 或 yolov8n.engine")
        return

    # 決定推論裝置
    # .engine 由 TensorRT 執行，不依賴 torch.device；
    # 但仍可傳 device=0 讓 Ultralytics 在 Jetson/NVIDIA 環境綁定 GPU。
    # .pt 則依據 torch 是否可用決定 cpu/cuda。
    use_trt = args.model.lower().endswith(".engine")
    if args.device:
        # 使用者強制指定
        infer_device = 0 if args.device == "cuda" else "cpu"
    else:
        # 自動：有 CUDA 就用 GPU，否則 CPU
        if _HAS_TORCH and torch.cuda.is_available():
            infer_device = 0
        else:
            infer_device = "cpu"

    # 載入模型
    # Ultralytics 會依副檔名自動走 PyTorch(.pt) 或 TensorRT(.engine)
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"[錯誤] 載入模型失敗：{e}")
        if use_trt:
            print("提示：.engine 檔需在目標機（Jetson Nano）本機匯出且 TensorRT 版本相容。")
        return

    # 視訊來源
    cap = open_capture(args)
    if not cap.isOpened():
        print("[錯誤] 相機/串流開啟失敗")
        return

    # 預熱（降低首幀延遲）
    warm_frames = 5
    for _ in range(warm_frames):
        ok, frame = cap.read()
        if not ok:
            break
        _ = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=infer_device,
            classes=[PERSON_CLASS_ID],  # 只跑 person
            verbose=False
        )

    window_name = "YOLOv8 Person (Unified PT/TRT)"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # FPS 統計
    t0 = time.time()
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("讀取影像失敗，結束")
                break

            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=infer_device,
                classes=[PERSON_CLASS_ID],  # 只跑 person
                verbose=False
            )

            r0 = results[0]
            if args.show:
                annotated = draw_person_boxes(frame, r0)
                cv2.imshow(window_name, annotated)
                # 按 ESC 離開
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frames += 1
            if frames % 30 == 0:
                t1 = time.time()
                fps = 30.0 / (t1 - t0)
                t0 = t1
                mode = "TRT" if use_trt else ("CUDA" if infer_device == 0 else "CPU")
                print(f"FPS: {fps:.2f} | mode={mode} imgsz={args.imgsz} conf={args.conf} iou={args.iou}")

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
