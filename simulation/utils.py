import profile
import sys
import os
import logging
import datetime
import pandas as pd
from job import Job, Trace
from policy import (
    ShortestJobFirst,
    FirstInFirstOut,
    ShortestRemainingTimeFirst,
    QuasiShortestServiceFirst,
    Lucid,
    Lucid_alwaysgpu,
    Lucid_fixed, 
    Lucid_continue,
    Lucid_nogpu,
    Lucid_node_scale,
    Tiresias,
)
from profiler import LeastGPUFirstProfiler
import numpy as np

sys.path.append("..")

PROFILER_ENABLED_SCHEDULERS = ["lucid", "lucid-alwaysgpu","lucid-nogpu", "lucid-node-scale", "lucid-fixed", "lucid-continue", "search"]


def simulate_vc(trace, vc, placement, log_dir, policy, logger, start_ts, *args):

    threshold_ratio=0.25
    if len(args) == 1:
        estimator = args[0]
    elif len(args) == 3:
        estimator, updater, learning_method = args[0], args[1], args[2]
    elif len(args) == 4:
        estimator, updater, learning_method = args[0], args[1], args[2]
        threshold_ratio = args[3]
    else:
        raise NotImplementedError("args error")
        
    if policy == "sjf":
        scheduler = ShortestJobFirst(trace, vc, placement, log_dir, logger, start_ts)
    elif policy == "fifo":
        scheduler = FirstInFirstOut(trace, vc, placement, log_dir, logger, start_ts)
    elif policy == "srtf":
        scheduler = ShortestRemainingTimeFirst(trace, vc, placement, log_dir, logger, start_ts)
    elif policy == "qssf":
        estimator = args[0]
        scheduler = QuasiShortestServiceFirst(trace, vc, placement, log_dir, logger, start_ts, estimator)
    elif policy == "lucid":
        # estimator, updater, learning_method = args[0], args[1], args[2]
        scheduler = Lucid(trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method)
    elif policy == "lucid-alwaysgpu":
        # estimator, updater, learning_method = args[0], args[1], args[2]
        scheduler = Lucid_alwaysgpu(trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method, threshold_ratio)
    elif policy == "lucid-nogpu":
        # estimator, updater, learning_method = args[0], args[1], args[2]
        scheduler = Lucid_nogpu(trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method, threshold_ratio)
    elif policy == "lucid-node-scale":
        # estimator, updater, learning_method = args[0], args[1], args[2]
        scheduler = Lucid_node_scale(trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method, threshold_ratio)
    elif policy == "lucid-fixed":
        # estimator, updater, learning_method = args[0], args[1], args[2]
        scheduler = Lucid_fixed(trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method, threshold_ratio)
    elif policy == "lucid-continue":
        # estimator, updater, learning_method = args[0], args[1], args[2]
        scheduler = Lucid_continue(trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method, threshold_ratio)
    elif policy == "tiresias":
        scheduler = Tiresias(trace, vc, placement, log_dir, logger, start_ts)
    else:
        raise ValueError(f"Invalid Scheduler Name {policy}")
    scheduler.simulate()
    logger.info(f"Finish {vc.vc_name}")
    return True


def trace_profile(trace, scale, time_limit, profiler_factor, placement, log_dir, logger, start_ts, node_scaling_num=1):
    profiler = LeastGPUFirstProfiler(trace, scale, time_limit, profiler_factor, placement, log_dir, logger, start_ts)
    profiler.set_prof_nodescale(node_scaling_num=node_scaling_num)
    profiler.profile()
    trace.reset_trace()
    logger.info("Finish Profiling")
    return trace


def get_available_schedulers():
    return ["fifo", "sjf", "srtf", "qssf", "lucid", "tiresias", "lucid-alwaysgpu", "lucid-nogpu","lucid-node-scale", "lucid-fixed", "lucid-continue", "search"]


def get_sweep_schedulers():
    # return ["fifo", "sjf", "srtf", "qssf", "tiresias", "lucid"]
    return ["fifo", "sjf", "srtf", "qssf", "tiresias"]


def get_available_placers():
    return ["random", "consolidate", "consolidateFirst"]


def trace_process(dir, date_range, read_full):
    start = "2020-04-01 00:00:00"
    if read_full == False:
        df = pd.read_csv(
            dir + "/cluster_log.csv",
            parse_dates=["submit_time"],
            usecols=["job_id", "user", "vc", "jobname", "gpu_num", "cpu_num", "state", "submit_time", "duration"],
        )
    else:
        df = pd.read_csv(
            dir + "/cluster_full_log.csv",
            parse_dates=["submit_time"],
            usecols=[
                "job_id",
                "user",
                "vc",
                "jobname",
                "gpu_num",
                "cpu_num",
                "state",
                "submit_time",
                "duration",
                "dataset",
                "model",
                "batchsize",
                "amp",
                "speed",
                "gpu_util",
                "gmem_util",
                "gmem",
            ],
        )
    # Consider gpu jobs only
    df = df[df["gpu_num"] > 0]

    # VC filter
    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vc_list = vc_df.index.to_list()
    df = df[df["vc"].isin(vc_list)]

    df.sort_values(by="submit_time", inplace=True)
    df = df[df["submit_time"] >= pd.Timestamp(start)]
    df["submit_time"] = df["submit_time"].apply(lambda x: int(datetime.datetime.timestamp(pd.Timestamp(x))))

    # Normalizing
    df["submit_time"] = df["submit_time"] - df.iloc[0]["submit_time"]

    df["remain"] = df["duration"]
    df[["start_time", "end_time"]] = sys.maxsize
    df[["ckpt_times", "queue", "jct"]] = 0
    df["status"] = None

    # Slicing simulation part
    begin = (pd.Timestamp(date_range[0]) - pd.Timestamp(start)).total_seconds()
    end = (pd.Timestamp(date_range[1]) - pd.Timestamp(start)).total_seconds()
    df = df[(df["submit_time"] >= begin) & (df["submit_time"] <= end)]

    df.sort_values(by="submit_time", inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df, begin


def trace_real_process(dir):
    df = pd.read_csv(
        dir + "/cluster_full_log.csv",
        parse_dates=["submit_time"],
        usecols=[
            "job_id",
            "user",
            "vc",
            "jobname",
            "gpu_num",
            "cpu_num",
            "state",
            "submit_time",
            "duration",
            "dataset",
            "model",
            "batchsize",
            "amp",
            "speed",
            "gpu_util",
            "gmem_util",
            "gmem",
        ],
    )

    # VC filter
    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vc_list = vc_df.index.to_list()
    df = df[df["vc"].isin(vc_list)]

    df["remain"] = df["duration"]
    df[["start_time", "end_time"]] = sys.maxsize
    df[["ckpt_times", "queue", "jct"]] = 0
    df["status"] = None
    df["submit_time"] = df["submit_time"].astype(float)
    df["submit_time"] = df["submit_time"].astype(int)
    df.reset_index(inplace=True, drop=True)

    return df, 0


def trace_pollux_process(dir, idx):
    df = pd.read_csv(
        f"{dir}/cluster_full_log_{idx}.csv",
        parse_dates=["submit_time"],
        usecols=[
            "job_id",
            "user",
            "vc",
            "jobname",
            "gpu_num",
            "cpu_num",
            "state",
            "submit_time",
            "duration",
            "dataset",
            "model",
            "batchsize",
            "amp",
            "speed",
            "gpu_util",
            "gmem_util",
            "gmem",
        ],
    )

    # VC filter
    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vc_list = vc_df.index.to_list()
    df = df[df["vc"].isin(vc_list)]

    df["remain"] = df["duration"]
    df[["start_time", "end_time"]] = sys.maxsize
    df[["ckpt_times", "queue", "jct"]] = 0
    df["status"] = None
    df["submit_time"] = df["submit_time"].astype(float)
    df["submit_time"] = df["submit_time"].astype(int)
    df.reset_index(inplace=True, drop=True)

    return df, 0


def trace_philly_process(dir, date_range, read_full):
    start = "2017-10-01 00:00:00"
    if read_full == False:
        df = pd.read_csv(
            dir + "/cluster_log.csv",
            parse_dates=["submit_time"],
            usecols=["user", "vc", "jobname", "gpu_num", "state", "submit_time", "duration"],
        )
    else:
        df = pd.read_csv(
            dir + "/cluster_full_log.csv",
            parse_dates=["submit_time"],
            usecols=[
                "user",
                "vc",
                "job_id",
                "gpu_num",
                "state",
                "submit_time",
                "duration",
                "dataset",
                "model",
                "batchsize",
                "amp",
                "speed",
                "gpu_util",
                "gmem_util",
                "gmem",
            ],
        )
    # Consider gpu jobs only
    df = df[df["gpu_num"] > 0]
    df.sort_values(by="submit_time", inplace=True)
    # VC filter
    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vc_list = vc_df.index.to_list()
    df = df[df["vc"].isin(vc_list)]

    df = df[df["submit_time"] >= pd.Timestamp(start)]
    df["submit_time"] = df["submit_time"].apply(lambda x: int(datetime.datetime.timestamp(pd.Timestamp(x))))

    df.rename(columns={"jobname": "job_id"}, inplace=True)
    df["state"] = df["state"].replace("Pass", "COMPLETED")
    df["state"] = df["state"].replace("Failed", "FAILED")
    df["state"] = df["state"].replace("Killed", "CANCELLED")

    # Normalizing
    df["submit_time"] = df["submit_time"] - df.iloc[0]["submit_time"]

    df["remain"] = df["duration"]
    df[["start_time", "end_time"]] = sys.maxsize
    df[["ckpt_times", "queue", "jct"]] = 0
    df["status"] = None

    # Slicing simulation part
    begin = (pd.Timestamp(date_range[0]) - pd.Timestamp(start)).total_seconds()
    end = (pd.Timestamp(date_range[1]) - pd.Timestamp(start)).total_seconds()
    df = df[(df["submit_time"] >= begin) & (df["submit_time"] <= end)]

    df.sort_values(by="submit_time", inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df, begin


def trace_parser(df):
    trace = Trace()

    for _, series in df.iterrows():
        trace.append_job(Job(series))
    trace.sort_jobs("submit_time")
    return trace


def logger_init(file):
    logger = logging.getLogger()
    handler_file = logging.FileHandler(f"{file}.log", "w")
    handler_stream = logging.StreamHandler()  # sys.stdout

    logger.setLevel(logging.INFO)
    handler_file.setLevel(logging.INFO)
    handler_stream.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(processName)s | %(message)s", datefmt="%Y %b %d %H:%M:%S")
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)

    logger.addHandler(handler_file)
    logger.addHandler(handler_stream)

    return logger


def cluster_concatenate(policy, placer, log_dir, dir, vc_dict, scheduler_for_vc=None):
    prefix = f"{policy}_{placer}"
    if not os.path.exists(log_dir + "/all"):
        os.mkdir(log_dir + "/all")

    # vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    # vcs = vc_df.index.to_list()
    vcs = vc_dict.keys()

    """Log"""
    cluster_log = pd.DataFrame()
    for vc in vcs:
        if policy == 'search':
            prefix = f'{scheduler_for_vc[vc]}_{placer}'
        vc_log = pd.read_csv(f"{log_dir}/{vc}/{prefix}_{vc}_log.csv")
        cluster_log = pd.concat([cluster_log, vc_log])
    cluster_log.sort_values(by="submit_time", inplace=True)

    prefix = f"{policy}_{placer}"
    cluster_log.to_csv(f"{log_dir}/all/{prefix}_all_log.csv", index=False)

    """Seq"""
    cluster_seq = pd.DataFrame()
    add_list = [
        "total_gpu_num",
        "idle_gpu_num",
        "pending_gpu_num",
        "running_gpujob_num",
        "pending_gpujob_num",
        "pending_job_num_less_8",
        "total_node_num",
        "consolidate_node_num",
        "shared_node_num",
    ]
    for vc in vcs:
        if policy == 'search':
            prefix = f'{scheduler_for_vc[vc]}_{placer}'

        vc_seq = pd.read_csv(f"{log_dir}/{vc}/{prefix}_{vc}_seq.csv")
        if len(cluster_seq) == 0:
            cluster_seq = vc_seq
            continue
        cluster_seq[add_list] = cluster_seq[add_list] + vc_seq[add_list]
        cluster_seq.dropna(inplace=True)
        cluster_seq = cluster_seq.astype(int)
        cluster_seq["gpu_utilization"] = (
            (cluster_seq["total_gpu_num"] - cluster_seq["idle_gpu_num"]) / cluster_seq["total_gpu_num"]
        ).round(3)
    
    prefix = f"{policy}_{placer}"
    cluster_seq.to_csv(f"{log_dir}/all/{prefix}_all_seq.csv", index=False)


def cluster_analysis(placer, log_dir, dir, vc_dict, filter_profile_job=False, scheduler_for_vc=None):
    """Generate Algorithm Comparsion CSV"""
    # ignore_warm_up = start_ts + 7*24*3600

    # vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    # vcs = vc_df.index.to_list()
    vcs = list(vc_dict.keys())
    vcs.append("all")

    files = os.listdir(f"{log_dir}/all")
    prefix = set()
    for file in files:
        policy = file.split("_")[0]
        placer = file.split("_")[1]
        prefix.add(f"{policy}_{placer}")
    prefix_list = sorted(list(prefix))

    # prefix_list = []
    # for i in get_available_schedulers():
    #     prefix = f"{i}_{placer}"
    #     prefix_list.append(prefix)
    
    # filter_profile_job = False

    jct_avg = pd.DataFrame()
    que_avg = pd.DataFrame()
    for prefix in prefix_list:
        for vc in vcs:
            if 'search' in prefix and vc != "all":
                placer = prefix.split('_')[1]
                prefix_vc = f"{scheduler_for_vc[vc]}_{placer}"
                vc_log = pd.read_csv(f"{log_dir}/{vc}/{prefix_vc}_{vc}_log.csv")
            else:
                vc_log = pd.read_csv(f"{log_dir}/{vc}/{prefix}_{vc}_log.csv")
            # vc_log = vc_log[vc_log['submit_time'] > ignore_warm_up]
            if filter_profile_job:
                # jct_avg.at[vc, prefix] = vc_log[(vc_log["jct"]>200) & (vc_log["gpu_num"]<=8) ]["jct"].mean() # why gpu_num <=8?
                jct_avg.at[vc, prefix] = vc_log[vc_log["jct"]>200]["jct"].mean()
                
            else:
                jct_avg.at[vc, prefix] = vc_log["jct"].mean()
                
            que_avg.at[vc, prefix] = vc_log["queue"].mean()

    jct_avg = jct_avg.astype(int)
    que_avg = que_avg.astype(int)
    if filter_profile_job:
        jct_avg.to_csv(f"{log_dir}/jct_avg_{placer}_execludeProfileGPU.csv")
    else:
        jct_avg.to_csv(f"{log_dir}/jct_avg_{placer}.csv")
    que_avg.to_csv(f"{log_dir}/que_avg_{placer}.csv")


def get_filled_trace(trace_dir, base):
    df = pd.read_csv(f"{trace_dir}/filled.csv")
    df = df[base:]
    start_ts = df.iloc[0]['submit_time']
    return df, start_ts

def get_trace(experiment_name, trace_dir, read_full, idx=None):
    if "Philly" in experiment_name:
        # trace_range = ("2017-10-01 00:00:00", "2017-10-07 23:59:00")
        # trace_df, start_ts = trace_philly_process(trace_dir, trace_range, read_full)
        trace_df, start_ts = get_filled_trace(trace_dir, 1926)
    elif "MLaas" in experiment_name:
        trace_df, start_ts = get_filled_trace(trace_dir, 15635)
        
    elif "Pollux" in experiment_name:
        trace_df, start_ts = trace_pollux_process(trace_dir, idx)
    elif "Venus" in experiment_name:
        if "Sept" in experiment_name:
            trace_range = ("2020-09-01 00:00:00", "2020-09-26 23:59:00")
            trace_df, start_ts = trace_process(trace_dir, trace_range, read_full)
        elif "July" in experiment_name:
            trace_range = ("2020-07-01 00:00:00", "2020-07-31 23:59:00")
            trace_df, start_ts = trace_process(trace_dir, trace_range, read_full)
        else:
            raise ValueError
    else:
        raise ValueError(f"Experiment {experiment_name} is not support or correct.")

    return trace_df, start_ts


def profiler_config(experiment_name, vc_dict):
    cluster = experiment_name.split("_")[0]
    profile_scale = {"Venus": 2, "Philly": 2, "MLaas":25}
    profile_time = {"Venus": 200, "Philly": 80, "MLaas":200}
    profile_factor = {"Venus": 4, "Philly": 2, "MLaas": 8}

    # Basic Config
    scale, time, factor = profile_scale[cluster], profile_time[cluster], profile_factor[cluster]
    if cluster == "Philly":
        # vc_dict["philly"] -= scale
        vc_dict["vc8Gr"] -= 1
        vc_dict["vcefl"] -= 1
    elif cluster == "Venus": # why subtract 1 for only these two vc?
        if "vc8Gr" in vc_dict:
            vc_dict["vc8Gr"] -= 1
        if "vcefl" in vc_dict:
            vc_dict["vcefl"] -= 1
        # vc_dict["vcYVn"] -= 1  # For elastic scaling
    elif cluster == "MLaas": # why subtract 1 for only these two vc?
        vc_dict["vc8Gr"] -= 1
        vc_dict["vcefl"] -= 1
    return vc_dict, scale, time, factor


def check_profiler_scale_available(experiment_name, scale, vc_dict, prof_locate_vc=None):
    # Use only for debug
    default_vc = {
        "Venus": "vc8Gr",
        "Saturn": "vcqdr",
        "Philly": "philly",
    }
    cluster = experiment_name.split("_")[0]

    if not prof_locate_vc:
        vc = default_vc[cluster]

    if scale <= vc_dict[vc]:
        return vc
    else:
        raise ValueError("Profile Node Scale Exceed VC Capacity")

def get_minimal_nodes(experiment_name):
   if experiment_name == "Venus_Sept":
        return {'vcEwI': 4, 'vcWoR': 2, 'vcHvQ': 4, 'vcvGl': 18, 'vc8Gr': 3+1, 'vcKeu': 5, 'vcKrE': 1, 'vcYVn': 5, 'vchbv': 4, 'vcLTP': 2, 'vchA3': 1, 'vcJsw': 20, 'vcefl': 1, 'vcvlY': 2, 'vcgkz': 1}

   elif experiment_name == "Philly":
        return {'vcEwI': 4, 'vcWoR': 4, 'vcHvQ': 4, 'vcvGl': 4, 'vc8Gr': 2+1, 'vcKeu': 4, 'vcKrE': 2, 'vcYVn': 4, 'vchbv': 1, 'vcLTP': 4, 'vchA3': 4, 'vcJsw': 4, 'vcefl': 2, 'vcvlY': 4, 'vcgkz': 1}
   elif experiment_name == "MLaas":
       return {'vcEwI': 25, 'vcWoR': 25, 'vcHvQ': 25, 'vcvGl': 25, 'vc8Gr': 25+1, 'vcKeu': 25, 'vcKrE': 25, 'vcYVn': 25, 'vchbv': 25, 'vcLTP': 25, 'vchA3': 25, 'vcJsw': 25, 'vcefl': 25, 'vcvlY': 25, 'vcgkz': 13}
   else:
       raise NotImplementedError("The minimal nodes for other experiment are not provided currently.")


def trace_scale_sample(trace, scale, vc_dict, sharescore_predict=None):
    numeric_fields = ['gpu_num', 'submit_time', 'duration', 'speed', 'gpu_util',  'gmem_util', 'gmem', 'remain']
    task_fields = ['model', 'batchsize', 'dataset', 'amp']
    new_total_job_list = []
    
    if sharescore_predict is not None:
        df = pd.read_csv(sharescore_predict)
        
        for job in trace.job_list:
            if job["toskip"] == 0:
                m, b, d, a = job["model"], job["batchsize"], job["dataset"], job["amp"]
                info = df.query(" model == @m and batchsize == @b and dataset == @d and amp == @a")
                job["sharescore"] = info["label"].values[0]
        
        # numeric_fields.append("sharescore")
         

    for vc in vc_dict.keys():
        vc_trace = trace.vc_trace(vc)
        job_list = vc_trace.job_list

        job_skip = [job for job in job_list if job["toskip"] == 1]
        job_no_skip = [job for job in job_list if job["toskip"] == 0]
        new_job_no_skip = []
        
        print(f"vc: {vc} job nums before scale {len(job_skip)} + {len(job_no_skip)}")
        for i in range(0, len(job_no_skip), scale[vc]):

            base_job = job_no_skip[i]
            values = []
            if sharescore_predict is not None:
                sharescore_values = []

            total_service = 0 
            total_job = 0 
            for j in range(i, min(i+scale[vc], len(job_no_skip))):
                value = []
                for field in numeric_fields:
                    value.append(job_no_skip[j][field])
                
                if sharescore_predict is not None:
                    weight = job_no_skip[j]["duration"] * job_no_skip[j]["gpu_num"]
                    sharescore_values.append(job_no_skip[j]["sharescore"] * weight)
                value.append(job_no_skip[j]["sharescore"])
                # keep service same 
                total_service += job_no_skip[j]["duration"] * job_no_skip[j]["gpu_num"]
                total_job += 1
                import pdb; pdb.set_trace() 
            mean_service = total_service / total_job
            gpu_num = np.median(values, axis=0)[0]
            duration = max(1, int(mean_service / gpu_num))
            # duration = np.mean(values, axis=0)[2]
            mean_field = np.mean(values, axis=0)
            if sharescore_predict is not None:
                weight = gpu_num * duration
                sharescore_median = int(np.median(sharescore_values) / weight)
                if sharescore_median in sharescore_values:
                    index = sharescore_values.index(sharescore_median)
                else:
                    index = np.where(np.array(sharescore_values) > sharescore_median)[0][0]

                for field in task_fields:
                    base_job[field] = job_no_skip[i+index][field]
                
            

            base_job['gpu_num'] = int(gpu_num)
            duration = max(1, int(mean_service / gpu_num))
            for i in range(1, len(numeric_fields)):
                base_job[numeric_fields[i]] = type(base_job[numeric_fields[i]])(mean_field[i])
                if numeric_fields[i] == "duration":
                    base_job[numeric_fields[i]] = duration
            
            new_job_no_skip.append(base_job)

        new_total_job_list.extend(job_skip)
        new_total_job_list.extend(new_job_no_skip)
        print(f"vc: {vc} job nums after scale {len(job_skip)} + {len(new_job_no_skip)}")

        
        # vc_trace.job_list = job_list
    new_trace = Trace() 
    new_trace.job_list = new_total_job_list
    new_trace.sort_jobs('submit_time')
    return new_trace

if __name__ == "__main__":
    files = os.listdir(f"log/Venus_Sept/all")
    prefix = set()
    for file in files:
        policy = file.split("_")[0]
        placer = file.split("_")[1]
        prefix.add(f"{policy}_{placer}")
    prefix_list = sorted(list(prefix))
    print(prefix_list)
