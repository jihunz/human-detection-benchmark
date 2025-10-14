'''
env PYTORCH_ENABLE_MPS_FALLBACK=1 python fine_tuning/v4/yolo_ft_megafall.py
-> PyTorch의 mps 백엔드 불리언 인덱싱 버그(TaskAlignedAssigner에서 발생한 shape mismatch)가 발생할 경우 문제가 발생하는 연산만 cpu로 처리 가능
'''

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
# 기본 가중치(사전학습)나 재학습할 체크포인트를 이 경로에 지정한다.
# last.pt를 넘겨주면 아래 로직이 자동으로 resume 모드로 전환된다.
DEFAULT_WEIGHTS = Path('/Users/jihunjang/workspace/ust/human-detection/fine_tuning/yolo12n.pt')
RESUME_WEIGHTS = Path('/Users/jihunjang/workspace/ust/human-detection/fine_tuning/v4/result/train23/weights/last.pt')
DATA_CFG = BASE_DIR / 'data.yaml'

from ultralytics.utils.tal import TaskAlignedAssigner
import torch

def patch_task_aligned_assigner_for_mps():
    if not torch.backends.mps.is_available():
        return

    original_forward = TaskAlignedAssigner._forward

    def _forward_mps_safe(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        if pd_scores.device.type == "mps":
            tensors = [t.cpu() if torch.is_tensor(t) else t for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            outputs = original_forward(self, *tensors)
            return tuple(out.to("mps") if torch.is_tensor(out) else out for out in outputs)
        return original_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    TaskAlignedAssigner._forward = _forward_mps_safe

patch_task_aligned_assigner_for_mps()


def _resolve_weights() -> tuple[Path, bool]:
    """Return (weight_path, resume_flag) deciding automatically when to resume."""

    # 사용자가 RESUME_WEIGHTS를 실제 last.pt로 지정해 두었으면 이를 우선 사용한다.
    if RESUME_WEIGHTS.name == 'last.pt' and RESUME_WEIGHTS.exists():
        # run 디렉터리 하위에 train_args.yaml이 있는지 확인해 resume 가능 여부를 판단한다.
        train_dir = RESUME_WEIGHTS.parent.parent
        args_file = train_dir / 'args.yaml'
        if args_file.exists():
            return RESUME_WEIGHTS, True

    # 그 외에는 사전학습 가중치로부터 새 학습을 시작한다.
    if not DEFAULT_WEIGHTS.exists():
        raise FileNotFoundError(f'기본 가중치를 찾을 수 없습니다: {DEFAULT_WEIGHTS}')
    return DEFAULT_WEIGHTS, False


def run_train(resume: bool | None = None) -> None:
    weight_path, auto_resume = _resolve_weights()
    model = YOLO(str(weight_path))

    if resume or (resume is None and auto_resume):
        # 이전 학습의 설정을 그대로 이어서 사용하되, 디바이스만 강제로 지정한다.
        model.train(resume=True, device='mps')
        return

    model.train(
        data=str(DATA_CFG),
        imgsz=640,
        epochs=100,
        patience=20,
        batch=-1,  # 최대 배치 자동 탐색
        save=True,
        save_period=1,
        single_cls=False,
        freeze=10,
        lr0=1e-3,
        device='mps',
        workers=12,
        cache='ram',
        project=str(BASE_DIR / 'result'),
        verbose=True,
    )


if __name__ == '__main__':
    run_train()
