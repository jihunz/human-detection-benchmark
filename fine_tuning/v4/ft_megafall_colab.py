# ==============================
#  Colab 단일 셀 (YOLOv12n, imgsz=640, 100 global epochs)
#  - /shards/ 제거: 고정 경로 사용
#  - labels: train+test 모두 해제 여부 검사 후 필요 시 1회만 해제
# ==============================
import os, sys, tarfile, glob, shutil, subprocess, time, json
from pathlib import Path

# 설치 & 드라이브 마운트
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
from google.colab import drive
drive.mount('/content/drive')

# ----- 고정 경로 ("/shards" 없음) -----
DRIVE_BASE = Path("/content/drive/MyDrive/db/megafallv2")
TRAIN_TAR_DIR = DRIVE_BASE / "train"                 # images_train-*.tar
TEST_TAR_DIR  = DRIVE_BASE / "test"                  # images_test-*.tar
LABEL_TAR     = DRIVE_BASE / "labels/labels_all.tar"        # 단일 라벨 tar

# 라벨을 영구 보관할 디렉터리
DRIVE_DATA = DRIVE_BASE / "data"                     # → /.../data/labels/{train,test}/...
DRIVE_RUNS = Path("/content/drive/MyDrive/yolo-megafall")  # 체크포인트/로그
STATE_PATH = DRIVE_BASE / "train_state.json"         # 진행상태 저장

# 런타임 작업 루트
WORK = Path("/content/data")
(WORK / "images/train").mkdir(parents=True, exist_ok=True)
(WORK / "images/test").mkdir(parents=True, exist_ok=True)
(WORK / "labels").mkdir(parents=True, exist_ok=True)  # 드라이브 라벨로 symlink

# ----- 학습 파라미터 -----
GLOBAL_EPOCHS = 100
IMGSZ = 640
BATCH = -1          # 실패 시 32/16으로 조정
WORKERS = 2
FREEZE = 10
LR0 = 1e-3
PATIENCE = 20
SAVE_PERIOD = 1
MODEL = "yolo12n.pt"
MOSAIC = 0
MIXUP = 0
RECT = True

# ----- 유틸 -----
def run(cmd): print(">>", " ".join(cmd)); subprocess.check_call(cmd)
def untar(tar_path: Path, dest: Path):
    with tarfile.open(tar_path) as tar: tar.extractall(dest)
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def latest_ckpt_in(dirpath: Path):
    cands = []
    for pat in ["**/weights/last.pt", "**/weights/best.pt"]:
        cands += list(dirpath.glob(pat))
    if not cands: return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0])
def load_state():
    if STATE_PATH.exists():
        try: return json.loads(STATE_PATH.read_text())
        except Exception: pass
    return {"global_epoch": 0, "shard_idx": 0}
def save_state(s): STATE_PATH.write_text(json.dumps(s))

# ----- 준비 -----
for p in [DRIVE_BASE, DRIVE_DATA, DRIVE_RUNS, TRAIN_TAR_DIR, TEST_TAR_DIR]:
    ensure_dir(p)

# data.yaml (요청한 names 매핑)
DATA_YAML = Path("/content/data.yaml")
DATA_YAML.write_text(
    "train: /content/data/images/train\n"
    "val: /content/data/images/test\n"
    "nc: 12\n"
    "names:\n"
    "  0: person\n"
    "  1: fall_person\n"
    "  2: bicycle\n"
    "  3: car\n"
    "  4: motorcycle\n"
    "  5: airplane\n"
    "  6: bus\n"
    "  7: train\n"
    "  8: truck\n"
    "  9: boat\n"
    '  10: "traffic light"\n'
    '  11: "fire hydrant"\n'
)

# ----- 라벨: train/test 모두 해제되었는지 검사 → 필요 시 1회 해제 -----
LABELS_DIR = DRIVE_DATA / "labels"
ensure_dir(DRIVE_DATA)

def count_label_txt(root: Path):
    tr = sum(1 for _ in (root/"train").rglob("*.txt")) if (root/"train").exists() else 0
    te = sum(1 for _ in (root/"test").rglob("*.txt"))  if (root/"test").exists()  else 0
    return tr, te

need_extract = True
if LABELS_DIR.exists():
    tr_cnt, te_cnt = count_label_txt(LABELS_DIR)
    if tr_cnt > 0 and te_cnt > 0:
        need_extract = False
        print(f"[labels] Already present → train={tr_cnt:,}, test={te_cnt:,} (skip extraction)")
    else:
        print(f"[labels] labels/ exists but missing parts: train={tr_cnt:,}, test={te_cnt:,} → will extract")

if need_extract:
    assert LABEL_TAR.exists(), f"라벨 tar가 없습니다: {LABEL_TAR}"
    print("[labels] extracting:", LABEL_TAR)
    untar(LABEL_TAR, DRIVE_DATA)
    tr_cnt, te_cnt = count_label_txt(LABELS_DIR)
    assert tr_cnt > 0 and te_cnt > 0, f"라벨 해제 후 비정상(train={tr_cnt}, test={te_cnt})"
    print(f"[labels] Extracted → train={tr_cnt:,}, test={te_cnt:,})")

# Colab 작업 경로에 labels symlink
if (WORK / "labels").exists() and not (WORK / "labels").is_symlink():
    shutil.rmtree(WORK / "labels", ignore_errors=True)
if not (WORK / "labels").is_symlink():
    (WORK / "labels").symlink_to(LABELS_DIR, target_is_directory=True)
print("labels symlink ->", LABELS_DIR)

# ----- val(test) 이미지: 없으면 전부 해제해 고정 -----
val_root = WORK / "images/test"
has_val = any(p.name.startswith("shard-") for p in val_root.glob("*") if p.is_dir())
if not has_val:
    test_tars = sorted(TEST_TAR_DIR.glob("images_test-*.tar"))
    if not test_tars:
        print(f"[val] 경고: {TEST_TAR_DIR}에 images_test-*.tar가 없습니다. 평가는 생략됩니다.")
    else:
        for t in test_tars:
            print("[val] extracting:", t.name)
            untar(t, WORK)
        print("[val] Extracted test shards to:", val_root)
else:
    print("[val] Test shards already exist at:", val_root)

# ----- train 이미지 샤드 목록 & 체크포인트 -----
train_tars = sorted(TRAIN_TAR_DIR.glob("images_train-*.tar"))
assert train_tars, f"{TRAIN_TAR_DIR}에 images_train-*.tar가 없습니다."
RESUME = latest_ckpt_in(DRIVE_RUNS)
print("[resume]:", RESUME or "fresh start")

# 1샤드 = 1 local epoch (전 샤드 순회가 global 1 epoch)
def yolo_train_one_epoch(resume_path):
    base = [
        "yolo","train",
        f"data={DATA_YAML}",
        "epochs=1",
        f"imgsz={IMGSZ}",
        f"batch={BATCH}",
        "device=0",
        f"workers={WORKERS}",
        f"freeze={FREEZE}",
        f"lr0={LR0}",
        f"patience={PATIENCE}",
        f"mosaic={MOSAIC}",
        f"mixup={MIXUP}",
        f"rect={RECT}",
        "cache=disk",
        f"save_period={SAVE_PERIOD}",
        f"project={DRIVE_RUNS}",
        "name=exp",
        "exist_ok=True",
    ]
    if resume_path: base.append(f"resume={resume_path}")
    else:           base.append(f"model={MODEL}")
    run(base)

# 진행상태 로드
state = load_state()
g_epoch = state["global_epoch"]
start_shard = state["shard_idx"]
print(f"[state] resume @ global_epoch={g_epoch}, shard_idx={start_shard}")

# 글로벌 에폭 루프
while g_epoch < GLOBAL_EPOCHS:
    print(f"\n=== Global Epoch {g_epoch+1}/{GLOBAL_EPOCHS} ===")
    for si in range(start_shard, len(train_tars)):
        t = train_tars[si]
        print(f"[train] extracting shard ({si+1}/{len(train_tars)}):", t.name)
        untar(t, WORK)

        yolo_train_one_epoch(RESUME)
        RESUME = latest_ckpt_in(DRIVE_RUNS)

        # 디스크 관리: train shard 폴더 과다 시 오래된 것부터 정리
        train_shards = sorted([p for p in (WORK/"images/train").glob("shard-*") if p.is_dir()],
                              key=lambda p: p.stat().st_mtime)
        if len(train_shards) > 6:
            for p in train_shards[:3]:
                shutil.rmtree(p, ignore_errors=True)
                print("[cleanup] removed", p)

        # 진행상태 저장
        save_state({"global_epoch": g_epoch, "shard_idx": si+1})

    g_epoch += 1
    save_state({"global_epoch": g_epoch, "shard_idx": 0})
    start_shard = 0
    print(f"[state] completed global_epoch={g_epoch}")

print("\n[Done] Finished GLOBAL_EPOCHS.")
print("Latest checkpoint:", RESUME or "None")
print("Checkpoints are under:", DRIVE_RUNS)