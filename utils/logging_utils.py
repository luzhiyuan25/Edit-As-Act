import json, logging, os, time
LOGGER = logging.getLogger("planner")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setLevel(logging.INFO)
    LOGGER.addHandler(_h)

def write_planner_log(record: dict, log_dir: str | None = None, fname_prefix: str = "planner"):
    log_dir = log_dir or os.getenv("PLANNER_LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(log_dir, f"{fname_prefix}-{ts}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path

