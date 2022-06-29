from os.path import join


def edgelist_file(basedir: str, graphname: str) -> str:
    return join(basedir, f"{graphname}.txt")


def criticality_file(basedir: str, graphname: str, k: str) -> str:
    return join(
        basedir, f"exaustivo_{graphname}_{k}", "edge_global_deltas.csv"
    )


def train_result_file(basedir: str, graphname: str) -> str:
    return join(basedir, f"{graphname}_train.csv")


def class_result_file(basedir: str, graphname: str) -> str:
    return join(basedir, f"{graphname}_class.csv")


def auc_result_file(basedir: str, graphname: str) -> str:
    return join(basedir, f"{graphname}_auc.csv")


def roc_result_file(basedir: str, graphname: str) -> str:
    return join(basedir, f"{graphname}_roc.csv")


def roc_curve_file(basedir: str, graphname: str, parameter_sufix: str) -> str:
    return join(basedir, "roc_auc", f"{graphname}_{parameter_sufix}.png")


def classification_metrics_file(
    basedir: str, graphname: str, parameter_sufix: str
) -> str:
    return join(
        basedir, "classification_metrics", f"{graphname}_{parameter_sufix}.png"
    )


def train_curve_file(
    basedir: str, graphname: str, parameter_sufix: str
) -> str:
    return join(basedir, "train", f"{graphname}_{parameter_sufix}.png")
