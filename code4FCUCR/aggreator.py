from abc import ABC, abstractmethod
import copy
import os
import time
import random
from collections import OrderedDict

from copy import deepcopy
import torch
import numpy as np
import numpy.linalg as LA
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from utils.args import *
from utils.myfunction import *

class Aggregator(ABC):
    def __init__(
            self,
            clients,
            log_freq,
            global_model,
            args,
            sampling_rate=1.,
            sample_with_replacement=False,

            verbose=0,
            seed=None,

            **kwargs
    ):

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)
        self.args = args




        self.clients = clients

        self.device = args.device

        self.log_freq = log_freq
        self.verbose = verbose
        self.global_model = global_model

        self.n_clients = len(clients)

    

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement 
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients = list()

        self.c_round = 0
        self.hr = 0.0
        self.ndcg = 0.0
        self.whether_end = 0

        self.KB_feats = OrderedDict()

    @abstractmethod
    def mix(self):
        pass

    @abstractmethod
    def update_clients(self):
        pass

    def update_test_clients(self):
        for client in self.test_clients:
            copy_model(target=client.feature_extractor, source=self.global_model)

        for client in self.test_clients:
            client.update_sample_weights()
            client.update_learners_weights()


    def write_logs(self):
        for global_logger, clients in [
            (self.global_train_logger, self.clients),
            (self.global_test_logger, self.test_clients)
        ]:
            if len(clients) == 0 :
                continue

            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.

            total_n_samples = 0
            total_n_test_samples = 0
            for client_id, client in enumerate(clients):
                train_loss, train_acc, test_loss, test_acc = client.write_logs()
                if self.verbose > 1:
                    print(f"Client {client_id}..")

                    with np.printoptions(precision=3, suppress=True):
                        print("Pi: ", client.learners_weights.numpy())

                    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%|", end="")
                    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}% |")

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples

                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_test_acc /= total_n_test_samples

            if self.verbose > 0:
                print("Global..")
                print(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |", end="")
                print(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.3f}% |")
                print("+" * 50)

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)

            if self.verbose > 0:
                print("#" * 80)

    def sample_clients(self):
        """
        sample a list of clients without repetition

        """
        if self.sample_with_replacement:
            self.sampled_clients = \
                self.rng.choices(
                    population=self.clients,
                    k=self.n_clients_per_round,
                )
        else:
            self.sampled_clients = self.rng.sample(self.clients, k=self.n_clients_per_round)



class CentralizedAggregator(Aggregator):



    def mix(self):

        self.sample_clients()

        for client in self.sampled_clients:
            print(f"this client id is {client.user_id}")
            if self.args.method == 'my':
                client.step()
            else:
                client.step_nom()

            if self.args.method == 'my':

                self.KB_feats[client.session_id] = deepcopy(client.com_feats.detach())
 
                if self.args.KB_all != 0:
                    if len(self.KB_feats) > self.args.KB_all:
                        self.KB_feats.popitem(last=False)


        # self.global_model.to(self.device)
        average_model(self.sampled_clients, target_model=self.global_model)



        self.update_clients()

        self.c_round += 1



    def update_clients(self):

        feats_dict = {}

        feats_vectors = []
        client_vectors = []
        session_dict = {}
        if self.args.method == 'my' and self.c_round % self.args.update_round == 0:
            for i, (session_id, state_dict) in enumerate(self.KB_feats.items()):
                feats_vector = torch.cat([state_dict.view(-1)])  
                feats_dict[session_id] = feats_vector
                session_dict[i] = session_id  
                feats_vectors.append(feats_vector.cpu().numpy())
            
            for client in self.clients:
                client_feat = client.get_feats().view(-1)
                client_vectors.append(client_feat.cpu().numpy())



            scores = cosine_similarity(client_vectors, feats_vectors)
            print("计算了相似度")
            



        
        #  分发给所有的client
        for idx, client in enumerate(self.clients):
            client.store_params()
            if self.args.method == "my" and self.c_round % self.args.update_round == 0:
                # 自己的方法进行部分拷贝， 只拷贝transformer部分
                partial_copy_model(target=client.feature_extractor, source=self.global_model)
                # copy_model(target=client.feature_extractor, source=self.global_model)
                

                score= scores[idx]
                sorted_score = np.argsort(score)[::-1] # 先有一个降序的序列

                top5_indices = []
                
                for i in sorted_score:
                    if len(top5_indices) == self.args.KB_len:
                        break
                    if session_dict[i] in client.user_session:
                        pass
                    else:
                        top5_indices.append(i)

                session_top5 = []
                for i in top5_indices:
                    session_top5.append((session_dict[i], score[i]))


                if not session_top5:
                    print("其他模型中没有相似，没有进行模型转移")
                else:
                    beta = self.args.beta      
                    avg_state_dict = copy.deepcopy(client.feature_extractor.mlp4logfeats.state_dict())
                    avg_feats = torch.zeros_like(self.KB_feats[session_top5[0][0]])


                    
                    score_soft = torch.tensor([score for _, score in session_top5])  # shape: (top5,)
                    weights = torch.nn.functional.softmax(score_soft, dim=0)  # shape: (top5,)
                    for i, (session_id, score) in enumerate(session_top5):
                        feats = self.KB_feats[session_id]
                        avg_feats += weights[i] * feats  

                    if self.args.pca:
                        feats_list = [self.KB_feats[session_id] for session_id, _ in session_top5]
                        enhanced_feats = enhance_sessions_with_pca(feats_list)
                        client.other_feats = enhanced_feats
                    else:
                        client.other_feats = avg_feats
                        
                    print(f"成功将相似度前{len(session_top5)}模型的参数平均并赋值到clients中的com_feats")


            elif self.args.method == "base":
 
                copy_model(target=client.feature_extractor, source=self.global_model)
            
 
            
            print("已经分发")

    def cosine_similarity(self, param1, param2):
        
        return torch.nn.functional.cosine_similarity(param1.unsqueeze(0), param2.unsqueeze(0)).item()
        












































