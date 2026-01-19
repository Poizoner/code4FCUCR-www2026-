import sys

import torch
import warnings
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
torch.set_printoptions(precision=8)  
class mysampler(object):
    def __init__(self, data, all_item_ts, args):
    
        self.args = args
        self.train = data["train"] 
        self.valid = data["valid"]
        self.test = data["test"]
        self.all_item_ts = all_item_ts
        
        self.user_item = set(self.train['item_id'].tolist()) | \
                 set(self.valid['item_id'].tolist()) | \
                 set(self.test['item_id'].tolist())
        
        self.user_sessions = {
            user_id: list(sessions)
            for user_id, sessions in self.train.groupby('user_id')['session_id'].unique().items()
        }

        self.current_index = {
            user_id: 0
            for user_id, session in self.user_sessions.items()
        }  


    def get_nextsession(self, user_id):
        if user_id not in self.user_sessions:
            print(f"User {user_id} does not exist!")
            return None
        session_list = self.user_sessions[user_id]
        index = self.current_index[user_id]

        if index >= len(session_list):  
            print(f"All sessions of user {user_id} have been traversed!")
            self.current_index[user_id] = 0
            index = self.current_index[user_id]


        session_id = session_list[index]  
        self.current_index[user_id] += 1  

        return self.train[(self.train['user_id'] == user_id) & (self.train['session_id'] == session_id)], session_id

    def get_next_back(self, user_id):
        if user_id not in self.user_sessions:
            
            return None
        session_list = self.user_sessions[user_id]
        index = self.current_index[user_id]

        if index >= len(session_list): 
           
            index = 0


        session_id = session_list[index] 
        

        return self.train[(self.train['user_id'] == user_id) & (self.train['session_id'] == session_id)], session_id


    #   Each time we get a dataset (a session), we process it and return the corresponding seq, pos, and neg.
    # These are the interaction sequence, positive sample, and negative sample, respectively.
    def nextbatch(self, session_data, session_id):
        seq = np.zeros([self.args.maxlen], dtype=np.int32)
        pos = np.zeros([self.args.maxlen], dtype=np.int32)
        neg = np.zeros([self.args.maxlen], dtype=np.int32)


        nxt = session_data.iloc[-1]
        idx = self.args.maxlen - 1

        for index, row in session_data.iloc[-2::-1].iterrows():
            seq[idx] = row.item_id
            pos[idx] = nxt.item_id
            if index != 0:
                neg[idx] = random_neq(all=self.all_item_ts, s=set(self.train['item_id']))

            nxt = row

            idx -= 1
            if(idx == -1):
                break
   
        seq = seq.reshape(1, -1)
        pos = pos.reshape(1, -1)
        neg = neg.reshape(1, -1)
        return session_id, seq, pos, neg



 
    def get_test(self):
        seq = np.zeros([self.args.maxlen], dtype=np.int32)

        nxt = self.test.iloc[-1]
        idx = self.args.maxlen - 1
   
        item_index = [nxt.item_id]
        for index, row in self.test.iloc[-2::-1].iterrows():
            seq[idx] = row.item_id
            idx -= 1
            if idx == -1: break

        for _ in range(100):
            t = np.random.choice(list(self.all_item_ts), replace=False)
            while t in set(self.train['item_id']):
                t = np.random.choice(list(self.all_item_ts), replace=False)
            item_index.append(t)

        seq = seq.reshape(1, -1)


        return seq, item_index

    def get_valid(self):
        seq = np.zeros([self.args.maxlen], dtype=np.int32)

        nxt = self.valid.iloc[-1]
        idx = self.args.maxlen - 1
   
        item_index = [nxt.item_id]
        for index, row in self.valid.iloc[-2::-1].iterrows():
            seq[idx] = row.item_id
            idx -= 1
            if idx == -1: break

        for _ in range(100):
            t = np.random.choice(list(self.all_item_ts), replace=False)
            while t in set(self.train['item_id']):
                t = np.random.choice(list(self.all_item_ts), replace=False)
            item_index.append(t)

        seq = seq.reshape(1, -1)

        return seq, item_index












def random_neq(all, s):

    t = np.random.choice(list(all), replace=False)
    while t in s:
        t = np.random.choice(list(all), replace=False)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

    
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)


def getsessiondata(data):
    data

def copy_model(target, source):
  

    target.load_state_dict(source.state_dict())


def partial_copy_model(target, source, prefix_filter='mlp4logfeats'):

    source_state = source.state_dict()
    target_state = target.state_dict()

    
    for name, param in source_state.items():
        if name.startswith(prefix_filter):
            pass
        else:
            target_state[name].copy_(param.detach())

    


def average_model(clients, target_model, average_gradients=False):

    feature_models = []
    weights = (1 / len(clients)) * torch.ones(len(clients), device="cpu")
    for client_id, client in enumerate(clients):
        feature_models.append(client.feature_extractor)

    target_state_dict = target_model.state_dict(keep_vars=True)


    for key in target_state_dict:

        if target_state_dict[key].data.dtype == torch.float32:
          
            target_state_dict[key].data.fill_(0.)

            target_state_dict[key].grad = target_state_dict[key].data.clone()  
            target_state_dict[key].grad.data.fill_(0.)

            for model_id, model in enumerate(feature_models):
                state_dict = model.state_dict(keep_vars=True)
                target_state_dict[key].data += state_dict[key].data.clone() * weights[model_id]
                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad += weights[model_id] * state_dict[key].grad.clone()
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )
        else:

            target_state_dict[key].data.fill_(0)
            for model_id, model in enumerate(feature_models):
                state_dict = model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()


def evaluate(model, sampler, u, test_other_feats, args):
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0


    seq, item_index = sampler.get_test()


    predictions = -model.predict(user_ids=u, log_seqs=seq, KB_seqs=test_other_feats, item_indices=item_index)


    predictions = predictions[0]

    rank = predictions.argsort().argsort()[0].item()

    valid_user += 1

    if rank < 10:
        NDCG += 1 / np.log2(rank + 2)
        HT += 1



    return NDCG, HT

def evaluate_valid(model, sampler, u, valid_other_feats, args):
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    seq, item_index = sampler.get_valid()

    predictions = -model.predict(user_ids=u, log_seqs=seq, KB_seqs= valid_other_feats, item_indices=item_index)
    # print(predictions)
    predictions = predictions[0]

    rank = predictions.argsort().argsort()[0].item()

    valid_user += 1

    if rank < 10:
        NDCG += 1 / np.log2(rank + 2)
        HT += 1

    return NDCG, HT



def enhance_sessions_with_pca(feats_list, output_dim=1, target_shape=(15, 50)):
    # flatten each (1, 15, 50) to (750,)
    flattened = [s.view(-1).cpu().numpy() for s in feats_list]  # list of (750,)

    # stack into matrix: (20, 750)
    X = np.stack(flattened, axis=0)

    # PCA to reduce 750 → output_dim (e.g., 1)
    pca = PCA(n_components=output_dim)
    reduced = pca.fit_transform(X)  # shape: (20, output_dim)

    #target_shape（15×50）
    avg = np.mean(reduced, axis=0)  # shape: (output_dim,)
    repeated = np.tile(avg, np.prod(target_shape))[:np.prod(target_shape)]  
    final = repeated.reshape((1, *target_shape))  # shape: (1, 15, 50)

    return torch.tensor(final, dtype=torch.float32).to(feats_list[0].device)
