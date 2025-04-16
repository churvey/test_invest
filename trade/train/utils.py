import cloudpickle
import os

global_context = None


class Context:

    def __init__(self, work_dir="./_tasks_", enable_cache=True):
        self.parent = None
        self.work_dir = work_dir
        self.enable_cache = enable_cache
        if not os.path.exists(work_dir):
            os.makedirs(work_dir, exist_ok = True)

    def __enter__(self):
        global global_context
        self.parent = global_context
        global_context = self
        return global_context

    def __exit__(self, exc_type, exc_value, traceback):
        global_context = self.parent


def sub_dir(sub, create=True):
    global global_context
    assert global_context is not None
    s = os.path.join(global_context.work_dir, sub)
    if create:
        os.makedirs(s, exist_ok = True)
    return s


def from_cache(path, load_func=cloudpickle.loads):
    global global_context
    if global_context and global_context.enable_cache:
        s = os.path.join(global_context.work_dir, path)
        if os.path.exists(s):
            print(f"loading from cache:{s}")
            with open(s, "rb") as f:
                return load_func(f.read())
    return None


def save_cache(path, obj, dumps=cloudpickle.dumps):
    global global_context
    if global_context and global_context.enable_cache:
        s = os.path.join(global_context.work_dir, path)
        print(f"save to cache:{s}")
        with open(s, "wb") as f:
            f.write(cloudpickle.dumps(obj))
    return None


def get_models(name, epoch = -1):
    with Context() as ctx:
        saved_models = from_cache(f"models.pkl")
        if not saved_models:
            return None
        
        models = saved_models[name][epoch]["models"]
        return models
    
if __name__ == "__main__":
    models = get_models("reg")
    print(models)