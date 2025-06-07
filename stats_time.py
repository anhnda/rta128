import os

ONNX_EXPORT_PATTERN_V1 = "python tools/export_onnx.py -c configs/rtdetr/rtdetr_r%dvd_6x_coco.yml -r scheckpoint/rtdetr_r%dvd_6x_coco_from_paddle.pth --check"
RUNNING_CMD_PATTERN_V1 = "python tools/train.py -c configs/rtdetr/rtdetr_r%dvd_6x_coco.yml -r scheckpoint/rtdetr_r%dvd_6x_coco_from_paddle.pth --test-only"
SIZE_LIST = [18, 34, 50, 101]
LOG_ROOT = "logs"
MODEL_ROOT = "models"

def get_eval_log_v1_sz(suffix="a", sz=50, n = 10):
    LOG_DIR = "%s/eval_%s" % (LOG_ROOT, suffix)
    os.makedirs(LOG_DIR, exist_ok=True)
    for i in range(n):
        out_path_i = "%s/eval_%d_%d.txt" % (LOG_DIR, sz,i)
        cmd = RUNNING_CMD_PATTERN_V1 % (sz,sz)
        cmd += " > " + out_path_i
        print("Running: ", cmd)
        os.system(cmd)

def get_eval_log_v1_all(suffix="a", n = 10):
    for sz in SIZE_LIST:
        get_eval_log_v1_sz(suffix=suffix, sz=sz, n = n)
def engine_export(suffix="a", sz=50):
    # onnx:
    cmd = ONNX_EXPORT_PATTERN_V1 % (sz, sz)
    print("Running: ", cmd)
    os.system(cmd)
    cmd = "python tools/export_trt.py -i model.onnx"
    print(cmd)
    os.system(cmd)
    os.makedirs(MODEL_ROOT, exist_ok=True)
    cmd = "mv model.onnx %s/model_%d_%s.onnx" % (MODEL_ROOT, sz, suffix)
    os.system(cmd)
    cmd = "mv model.engine %s/model_%d_%s.engine" % (MODEL_ROOT, sz, suffix)
    os.system(cmd)
def run_engine(suffix="a", sz=50, n = 10):
    log_engine = "%s/trt_%s/" % (LOG_ROOT, suffix)
    os.makedirs(log_engine, exist_ok=True)
    log_engine = log_engine + "e_%d.txt" % sz
    cmd = "python deploy_trt.py -trt %s/model_%d_%s.engine" % (MODEL_ROOT,  sz, suffix) + " >> " + log_engine
    print("Run: ", cmd)
    for i in range(n):
        os.system(cmd)
def run_engine_all(suffix="a", n= 10):
    for sz in SIZE_LIST:
        run_engine(suffix, sz, n)
        
def engine_export_all(suffix="a"):
    for sz in SIZE_LIST:
        engine_export(suffix,sz)

if __name__ == "__main__":
    # get_eval_log_v1_all(suffix="o",n=5)
    for s in ["o", "a"]:
        engine_export_all(suffix=s)
    for s in ["o", "a"]:
        run_engine_all(suffix=s,n=5)