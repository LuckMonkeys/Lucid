import random
import operator
import os
import pandas as pd
from .policy import Policy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np


GPU_MEMORY_LIMITATION = 24576  # RTX 3090 24GB Memory for our benchmarking


class Lucid(Policy):
    def __init__(self, trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method, threshold_ratio=0.25):
        super(Lucid, self).__init__(trace, vc, placement, log_dir, logger, start_ts)
        # import pdb; pdb.set_trace() 
        self.estimator = estimator
        self.updater = updater
        self._name = "lucid"
        
        self.perfect_colocate_analysis = False 
        self.learning_fixed_colocate_analysis = False 
        self.continue_learning_colocate_analysis = False 
        self.learning_method = learning_method 
        
        if learning_method == 'perfect': 
            self.perfect_colocate_analysis = True 
        elif learning_method == 'fixed': 
            self.learning_fixed_colocate_analysis = True 
        elif learning_method == 'continue': 
            self.continue_learning_colocate_analysis = True 
        
        self.threshold_ratio=threshold_ratio
            
        self.enable_colocate()
        self.adaptive_colocate = 0
        self.obtain_workload_estimates()
        self.obtain_colocate_analysis()
        self.obtain_cluster_prediction()

    def enable_colocate(self):
        self._vc.colocate_enable = 1

    def obtain_workload_estimates(self):
        estimate = self.estimator.data
        for job in self.trace.job_list:
            if job["toskip"] == 0:
                job["priority"] = estimate[estimate["job_id"] == job["job_id"]]["priority"].iloc[0] * job["gpu_num"]

    def is_pair_sharing(self, jobA, jobB):
        if jobA['submit_time'] + jobA['duration'] > jobB['submit_time']:
            return True
        return False 



    def query_speed(self, trail, colo):
        m, d, b, a = trail["model"], trail["dataset"], trail["batchsize"], trail["amp"]
        info1 = colo.query("model1 == @m and batchsize1 == @b and dataset1 == @d and amp1 == @a")
        info2 = colo.query("model2 == @m and batchsize2 == @b and dataset2 == @d and amp2 == @a")

        speed1, len1, speed2, len2 = 0, len(info1), 0, len(info2)
        if len1 > 0:
            speed1 = info1["speed1"].sum()
        if len2 > 0:
            speed2 = info2["speed2"].sum()

        avg = (speed1 + speed2) / max(len1 + len2, 1)
        return round(avg, 3)
    
    
    def obtain_colocate_train_data(self, colocate_df_gt,joblist):
        df = self.colo_df
        # colocate_df_gt = pd.read_csv("data/colocate_info.csv")
        pred_df = pd.DataFrame(columns=["amp", "gpu_util", "gmem_util", "gmem", "label"])
        collect_colocate_df = pd.DataFrame(columns=["model1", "model2", "batchsize1", "batchsize2", "dataset1", "dataset2", "amp1", "amp2", "speed1", "speed2"])
        
        previous_jobs = list() 
        for job in joblist:
            if job["toskip"] == 0:
                selected_pair = None 
                for pair_job in previous_jobs[::-1]: 
                    if self.is_pair_sharing(pair_job, job): 
                        selected_pair = pair_job
                        break 
                if selected_pair is None:
                    previous_jobs.append(job)
                else: 
                    previous_jobs.remove(selected_pair)
                    modela, dataseta, batchsizea, ampa = job["model"], job["dataset"], job["batchsize"], job['amp']
                    modelb, datasetb, batchsizeb, ampb = selected_pair["model"], selected_pair["dataset"], selected_pair['batchsize'], selected_pair['amp']
                    
                    info1 = colocate_df_gt.query("model1 == @modela and batchsize1 == @batchsizea and dataset1 == @dataseta and amp1 == @ampa and model2 == @modelb and batchsize2 == @batchsizeb and dataset2 == @datasetb and amp1 == @ampb")
                    info1 = info1.drop(columns=['gpu_util', 'gmem_util', 'gmem'])
                    if len(info1) > 0: 
                        if len(collect_colocate_df) == 0:
                            collect_colocate_df = info1
                        else:
                            collect_colocate_df = pd.concat([collect_colocate_df, info1], ignore_index=True)
                
        for job in joblist: 
            if job["toskip"] == 0:
                avg_speed = self.query_speed(job, collect_colocate_df)
                # if avg_speed == 0:
                #     raise ValueError("avg_speed is zero")
                m, b, d, a = job["model"], job["batchsize"], job["dataset"], job["amp"]
                info = df.query("model == @m and batchsize == @b and dataset == @d and amp == @a")
                if avg_speed < 0.85:
                    auto_label = 2
                elif avg_speed < 0.95:
                    auto_label = 1
                else:
                    auto_label = 0
                new_row = {
                    "amp": info["amp"].values[0], 
                    "gpu_util": info["gpu_util"].values[0],
                    "gmem_util": info["gmem_util"].values[0],
                    "gmem": info["gmem"].values[0],
                    "label": auto_label
                }
                new_row_df = pd.DataFrame(new_row, index=[0])
                if len(pred_df) == 0:
                    pred_df = new_row_df
                else:
                    pred_df = pd.concat([pred_df, new_row_df], ignore_index=True)
            
        
        pred_df = pred_df.drop_duplicates()
        column_types = {
            "amp": int,
            "gpu_util": float,
            "gmem_util": float,
            "gmem": float,
            "label": int
        }
        pred_df = pred_df.astype(column_types)

        return pred_df

    def update_df_with_clf(self, df, clf):
        single = df.drop(columns=["dataset", "batchsize", "speed", "model"]) 
        test_data = single.drop(columns="label")
        replacement = clf.predict(test_data)
        df[['label']] = pd.DataFrame(replacement, columns=['label'], index=single.index)
        
    
    def obtain_colocate_analysis(self):
        self.get_colocate_data()
        df = self.colo_df
        colocate_df_gt = pd.read_csv("data/colocate_info.csv")
        
        # whether update df 
        if self.learning_fixed_colocate_analysis or self.continue_learning_colocate_analysis:
            if self.learning_fixed_colocate_analysis:
                train_len = int(len(self.trace.job_list) * 0.1)
                top_10_jobs = [job for job in self.trace.job_list[:train_len] if job["toskip"] == 0]
                train_df = self.obtain_colocate_train_data(colocate_df_gt=colocate_df_gt, joblist=top_10_jobs)
            else:
                finish_jobs = [job for job in self.trace.job_list if job["status"] == "end" and job["toskip"] == 0]
                train_df = self.obtain_colocate_train_data(colocate_df_gt=colocate_df_gt, joblist=finish_jobs)

            if len(train_df) > 0 :
                train_data = train_df.drop(columns="label")
                train_label = train_df[['label']].astype(int)

                clf = DecisionTreeClassifier()
                clf.fit(train_data, train_label)

                self.update_df_with_clf(df, clf) 
            else:
                print("-------------------------------train_df is None---------------------------------------")

        elif self.perfect_colocate_analysis:
            pass 
        else:
            raise ValueError("No learning method is specified")
         
        for job in self.trace.job_list:
            if job["toskip"] == 0:
                m, b, d, a = job["model"], job["batchsize"], job["dataset"], job["amp"]
                info = df.query(" model == @m and batchsize == @b and dataset == @d and amp == @a")
                job["sharescore"] = info["label"].values[0]
    
    
    def obtain_colocate_analysis_old(self):
        self.get_colocate_data()
        df = self.colo_df
        colocate_df_gt = pd.read_csv("data/colocate_info.csv")
        

        if self.perfect_colocate_analysis: 
             for job in self.trace.job_list:
                if job["toskip"] == 0:
                    m, b, d, a = job["model"], job["batchsize"], job["dataset"], job["amp"]
                    info = df.query(" model == @m and batchsize == @b and dataset == @d and amp == @a")
                    job["sharescore"] = info["label"].values[0]
                    # job["sharescore"] = random.randint(0, 2)# the worst one 
                    
        elif self.learning_fixed_colocate_analysis: 
            pred_df = pd.DataFrame(columns=["amp", "gpu_util", "gmem_util", "gmem", "label"])
            collect_colocate_df = pd.DataFrame(columns=["model1", "model2", "batchsize1", "batchsize2", "dataset1", "dataset2", "amp1", "amp2", "speed1", "speed2"])
            
            train_len = int(len(self.trace.job_list) * 0.1)
            previous_jobs = list() 
            for job in self.trace.job_list[:train_len]:
                if job["toskip"] == 0:
                    selected_pair = None 
                    for pair_job in previous_jobs[::-1]: 
                        if self.is_pair_sharing(pair_job, job): 
                            selected_pair = pair_job
                            break 
                    if selected_pair is None:
                        previous_jobs.append(job)
                    else: 
                        previous_jobs.remove(selected_pair)
                        modela, dataseta, batchsizea, ampa = job["model"], job["dataset"], job["batchsize"], job['amp']
                        modelb, datasetb, batchsizeb, ampb = selected_pair["model"], selected_pair["dataset"], selected_pair['batchsize'], selected_pair['amp']
                        
                        info1 = colocate_df_gt.query("model1 == @modela and batchsize1 == @batchsizea and dataset1 == @dataseta and amp1 == @ampa and model2 == @modelb and batchsize2 == @batchsizeb and dataset2 == @datasetb and amp1 == @ampb")
                        info1 = info1.drop(columns=['gpu_util', 'gmem_util', 'gmem'])
                        # import pdb; pdb.set_trace() 
                        if len(info1) > 0: 
                            collect_colocate_df = pd.concat([collect_colocate_df, info1], ignore_index=True)
                    
            for job in self.trace.job_list[:train_len]: 
                if job["toskip"] == 0:
                    avg_speed = self.query_speed(job, collect_colocate_df)
                    # if avg_speed == 0:
                    #     raise ValueError("avg_speed is zero")
                    m, b, d, a = job["model"], job["batchsize"], job["dataset"], job["amp"]
                    info = df.query("model == @m and batchsize == @b and dataset == @d and amp == @a")
                    if avg_speed < 0.85:
                        auto_label = 2
                    elif avg_speed < 0.95:
                        auto_label = 1
                    else:
                        auto_label = 0
                    new_row = {
                        "amp": info["amp"].values[0], 
                        "gpu_util": info["gpu_util"].values[0],
                        "gmem_util": info["gmem_util"].values[0],
                        "gmem": info["gmem"].values[0],
                        "label": auto_label
                    }
                    new_row_df = pd.DataFrame(new_row, index=[0])
                    pred_df = pd.concat([pred_df, new_row_df], ignore_index=True)
                
                # m, b, d, a = job["model"], job["batchsize"], job["dataset"], job["amp"]
                # info = df.query("model == @m and batchsize == @b and dataset == @d and amp == @a")
                # new_row = {
                #     "amp": info["amp"].values[0], 
                #     "gpu_util": info["gpu_util"].values[0],
                #     "gmem_util": info["gmem_util"].values[0],
                #     "gmem": info["gmem"].values[0],
                #     "label": info["label"].values[0]
                # }
                # new_row_df = pd.DataFrame(new_row, index=[0])
                # pred_df = pd.concat([pred_df, new_row_df], ignore_index=True)
                # print(pred_df)
                # pred_df = pred_df.append((info["amp"], info["gpu_util"], info["gmem_util"], info["gmem"], info["label"]))
            
            # import pdb; pdb.set_trace() 
            pred_df = pred_df.drop_duplicates()
            column_types = {
                "amp": int,
                "gpu_util": float,
                "gmem_util": float,
                "gmem": float,
                "label": int
            }
            pred_df = pred_df.astype(column_types)

            train_data = pred_df.drop(columns="label")
            train_label = pred_df[['label']].astype(int)
            clf = DecisionTreeClassifier()
            clf.fit(train_data, train_label)
            
            single = df.drop(columns=["dataset", "batchsize", "speed", "model"]) 
            test_data = single.drop(columns="label")
            replacement = clf.predict(test_data)
            df[['label']] = pd.DataFrame(replacement, columns=['label'], index=single.index)
            for job in self.trace.job_list:
                if job["toskip"] == 0:
                    m, b, d, a = job["model"], job["batchsize"], job["dataset"], job["amp"]
                    info = df.query(" model == @m and batchsize == @b and dataset == @d and amp == @a")
                    job["sharescore"] = info["label"].values[0]
                    
        elif self.continue_learning_colocate_analysis:
            for job in self.trace.job_list:
                if job["toskip"] == 0:
                    m, b, d, a = job["model"], job["batchsize"], job["dataset"], job["amp"]
                    info = df.query(" model == @m and batchsize == @b and dataset == @d and amp == @a")
                    job["sharescore"] = info["label"].values[0]
                # import pdb; pdb.set_trace()

    def obtain_cluster_prediction(self):
        cluster = self.estimator.cluster_name
        self.get_time_series_data(cluster)
        if cluster == "Venus":
            self.get_profile_scaling_data(cluster)

    def obtain_job_from_id(self, id):
        for job in self.run_list:
            if job["job_id"] == id:
                return job

    def obtain_history_throughput(self, metric, ratio):
        time_df = self.time_df[self.time_df["time"] < self.time]
        if len(time_df) == 0:
            return 10
        metric_history = time_df[metric].tolist()
        threshold = np.percentile(metric_history, ratio * 100)
        return threshold
    
    # Prescient Adaptive Sharing
    def check_pas(self):
        threshold = self.obtain_history_throughput(metric="submit_gpu_job", ratio=self.threshold_ratio) 
        if self.check_future_cluster_throughput(metric="pred_gpu_job") <= threshold:
        # if self.check_future_cluster_throughput(metric="pred_gpu_job") <= 2:
        # if self.check_future_cluster_throughput(metric='pred_gpu_num') <= self._vc.vc_free_gpus():
            return 0
        else:
            return 1
    
    def colocate_update(self, job, target_job):
        speed1, speed2, gutil, gmem = self.updater.query_info(job, target_job)
        job["exclusive"], target_job["exclusive"] = 0, 0
        job["rate"], target_job["rate"] = speed1, speed2
        job["Tcolocate"] = self.time
        return gutil, gmem

    def speed_recover(self, job_list):
        if isinstance(job_list, list):
            for job in job_list:
                job["exclusive"] = 1
                job["rate"] = 1
                job["Tdelocate"] = self.time
        else:
            job_list["exclusive"] = 1
            job_list["rate"] = 1
            job_list["Tdelocate"] = self.time

    def ablation_picker(self, job):
        mem_limit = GPU_MEMORY_LIMITATION - job["gmem"]
        affinity_jobs = []
        for j in self.run_list:
            if j["exclusive"] == 0:
                continue
            if j["gpu_num"] == job["gpu_num"] and j["gmem"] < mem_limit:
                affinity_jobs.append(j)

        if affinity_jobs:
            return affinity_jobs[0]
        else:
            return False

    def job_pair_picker_time_aware(self, job):
        mem_limit = GPU_MEMORY_LIMITATION - job["gmem"]
        affinity_jobs = []
        for j in self.run_list:
            if j["exclusive"] == 0:
                continue
            if j["gpu_num"] == job["gpu_num"] and j["gmem"] < mem_limit and (job["sharescore"] + j["sharescore"]) <= 2:
                affinity_jobs.append(j)
                # print('--'* 20)
                # print(job['priority'] / job['gpu_num'], job['duration'])
                # print(j['priority'] / j['gpu_num'], j['duration'])
                # import pdb; pdb.set_trace() 
                # if job["priority"] <= j["priority"] * 2:
                    

        if affinity_jobs:
            # if job["sharescore"] == 0 or job["sharescore"] == 1:
            #     affinity_jobs.sort(key=lambda x: x.__getitem__("sharescore"))
            #     return affinity_jobs[0]
            # else:
            #     return affinity_jobs[0]
            return affinity_jobs[0]
            # return random.choice(affinity_jobs)
        else:
            return False

    def job_allocate_info_update(self, job):
        job["start_time"] = self.time
        job["queue"] = job["queue"] + self.time - job["submit_time"]
        job["status"] = "run"
        self.que_list.remove(job)
        self.run_list.append(job)

    def simulate(self):
        prev_index = 0
        stale_que = []
        delta = 10
        
        update_start = 0.1
        num_skip_jobs = len([job for job in self.trace.job_list if job["toskip"] == 1])
        num_jobs = len(self.trace.job_list)

        
        
        while self.end_job_num != self.total_job_num:
            new_job_num = 0

            """1. Check & Release End Jobs"""
            run_ls = self.run_list.copy()  # Avoid list.remove() issue
            remove_list = list() 
            
            for idx, job in enumerate(run_ls):
                if job["remain"] <= 0:
                    job["status"] = "end"
                    job["end_time"] = self.time
                    self.end_job_num += 1
                    if self._vc.colocate_enable and job["exclusive"] == 0:
                        colo_job_id = self._vc.check_vc_colocate_jobs(job)
                        if colo_job_id:
                            colo_job = self.obtain_job_from_id(colo_job_id)
                            self.speed_recover(colo_job)
                    
                    self._vc.release_resource(job)
                    remove_list.append(job)
                    # if self.estimator.name != "LGBEstimator" and self.estimator.name != "PhillyEstimator":
                    #     self.estimator.update_train_data(job)
                else:
                    job["remain"] -= job["rate"] * delta 
            
            for job in remove_list: 
                self.run_list.remove(job)
            
            """2. Check New Jobs"""
            # New Job
            for idx in range(prev_index, self.total_job_num):
                job = self.trace.job_list[idx]
                if job["toskip"]:
                    prev_index = idx + 1
                    self.end_job_num += 1
                    continue

                if self.time - job["submit_time"] >= 0 and self.time - job['submit_time'] < delta: # very important 
                    job["status"] = "pend"
                    self.que_list.append(job)
                    prev_index = idx
                    new_job_num += 1
                elif job["submit_time"] > self.time:
                    break

            """3. Sort Job According to Priority"""
            self.que_list.sort(key=lambda x: x.__getitem__("priority"))

            """4. Allocate Job"""
            que_ls = self.que_list.copy()
            
            if  (self.time - self.start_ts) % 100 == 0:
                # import pdb; pdb.set_trace() 
                self.adaptive_colocate = self.check_pas()
            

            # if self.continue_learning_colocate_analysis and self.time>0 and  (self.time - self.start_ts) % 10000 == 0:
            if self.continue_learning_colocate_analysis and self.time>0:
    
                num_end_jobs = len([job for job in self.trace.job_list if job["status"] == "end" and job["toskip"] == 0])
                if num_end_jobs >= (num_jobs - num_skip_jobs) * update_start:
                    self.obtain_colocate_analysis()
                    update_start += 0.1
                    # print(update_start)
             
            # if  (self.time - self.start_ts) % 100 == 0:
            if self.adaptive_colocate == 0:  # Disable colocation
                for job in que_ls:
                    if self.job_placer(job):
                        self.job_allocate_info_update(job)
                    else:
                        break
            else:
                for job in que_ls:
                    
                    if job["gpu_num"] <= 8:
                        # target_job = self.ablation_picker(job)
                        target_job = self.job_pair_picker_time_aware(job)
                        if target_job:
                            gutil, gmem = self.colocate_update(job, target_job)
                            assert self.colocate_job_placer(job, target_job, gutil, gmem)
                            self.job_allocate_info_update(job)
                        else:
                            if self.job_placer(job):
                                self.job_allocate_info_update(job)
                    else:
                        if self.job_placer(job):
                            self.job_allocate_info_update(job)

            """Echo Profiler Scaling"""
            if self.vc_echo_scaling and  (self.time - self.start_ts) % 10 == 0 and self.time in self.scaling_time_list:
                scaling_num = (
                    -1 * self.profile_scaling_df[self.profile_scaling_df["time"] == self.time]["scaling_num"].values[0]
                )
                self.logger.info(f"Time: {self.time}, Scaling Num: {scaling_num}")
                self._vc.update_vc_node(change_node_num=scaling_num)

            """5. Log & Result Recorder"""
            if  (self.time - self.start_ts) % 10000 == 0:
                self.runtime_log()

            # Sample Cluster State Every Minute
            if  (self.time - self.start_ts) % 60 == 0:
                self.seq_recorder()

            self.time += delta
            

        self.log_recorder(self._name)
        
class Lucid_fixed(Lucid):
    def __init__(self, trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method, threshold_ratio=0.25):
        super(Lucid_fixed, self).__init__(trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method="fixed", threshold_ratio=threshold_ratio)
        self._name = "lucid-fixed"
    
class Lucid_continue(Lucid):
    def __init__(self, trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method, threshold_ratio=0.25):
        super(Lucid_continue, self).__init__(trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method="continue", threshold_ratio=threshold_ratio)
        self._name = "lucid-continue"

class Lucid_alwaysgpu(Lucid):
    def __init__(self, trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method, threshold_ratio=0.25):
        super(Lucid_alwaysgpu, self).__init__(trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method="fixed", threshold_ratio=threshold_ratio)
        self._name = "lucid-alwaysgpu"
    
    def check_pas(self):
        return 1

class Lucid_nogpu(Lucid):
    def __init__(self, trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method, threshold_ratio=0.25):
        super(Lucid_nogpu, self).__init__(trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method="fixed" , threshold_ratio=threshold_ratio)
        self._name = "lucid-nogpu"
    
    def check_pas(self):
        return 0

class Lucid_node_scale(Lucid):
    def __init__(self, trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method, threshold_ratio=0.25):
        super().__init__(trace, vc, placement, log_dir, logger, start_ts, estimator, updater, learning_method="fixed" , threshold_ratio=threshold_ratio)

        self.profnode_scaling_num = None
        self.get_nodescale_num()
        self._name = f"lucid-node-scale-{self.profnode_scaling_num}"

    def get_nodescale_num(self):
        self.profnode_scaling_num  = abs(self.profile_scaling_df["scaling_num"].values[0])
        if self.profnode_scaling_num == 0:
            self.vc_echo_scaling = False

   