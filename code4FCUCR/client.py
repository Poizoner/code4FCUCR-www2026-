import torch
import time
import numpy as np
from copy import deepcopy
from utils.myfunction import *
from sklearn.metrics.pairwise import cosine_similarity

class client():
    def __init__(
            self,
            user_id,
            feature_extractor,
            data,
            all_item_ts,
            local_steps,
            args,
            KB_global=False,  # Whether to share globally,
            Distillation=False  # Whether to distill from the past


    ):


        self.user_id = user_id
        self.feature_extractor = feature_extractor
        self.action_selector = feature_extractor.mlp4logfeats
        self.local_steps = local_steps
        self.args = args
        self.train_session_count = data['train']['session_id'].nunique()
        self.all_item_ts = all_item_ts
        self.sampler = mysampler(data, all_item_ts, args)
        self.model_params = None  # Used to store model parameters
        self.KB = {}
        self.user_session = set(data['train']['session_id'].unique())
        self.session_id = None
        self.pos = None
        self.neg = None
        self.seq = None
        self.com_feats = None
        self.test_hr = 0.0
        self.test_ndcg = 0.0
        self.other_feats = None


        if data['test'].empty:
            self.have_test = False
        else:
            self.have_test = True

        if data['valid'].empty:
            self.have_valid = False
        else:
            self.have_valid = True


    def store_params(self): 
        self.model_params = self.feature_extractor.state_dict()
        print(f"client{self.user_id} feature parameters have been stored")

        return

    def flatten_params(self,model):
        
        params = []
        for param in model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def calculate_similarity(self, model1, model2):
        """Calculate the cosine similarity between two model parameters"""
        params1_flat = self.flatten_params(model1)
        params2_flat = self.flatten_params(model2)
        # To handle possible dimensionality mismatches, consider aligning or comparing only shared parameters.
        # This assumes that the two models have the same structure, so the number of parameters and dimensions should be consistent.
        if params1_flat.shape != params2_flat.shape:
            return -1 
        similarity = cosine_similarity(params1_flat.reshape(1, -1), params2_flat.reshape(1, -1))[0][0]
        return similarity

    def select_similar(self):
        similarities = []
        

        for session_id, mlp_model in self.KB.items():
            if session_id in self.user_session:
                pass
            else:
                similarity_score = self.calculate_similarity(self.feature_extractor.mlp4logfeats, mlp_model)
                similarities.append((similarity_score, mlp_model, session_id))

        similarities.sort(key=lambda item: item[0], reverse=True)
        print(len(similarities))

        if similarities == []:
            print("There are no similar models among other users, and no model transfer has been performed.")
            pass
        else:
            
            top_5_similar = similarities[:5]
            first_model_state_dict = top_5_similar[0][1].state_dict()
            averaged_state_dict = {}

            for key in first_model_state_dict:
                
                all_params_for_key = [model.state_dict()[key] for _,model,_ in top_5_similar]

                
                if isinstance(all_params_for_key[0], torch.Tensor):
                    stacked_params = torch.stack(all_params_for_key)
                    averaged_param = torch.mean(stacked_params, dim=0)
                    averaged_state_dict[key] = averaged_param
                else:
                    averaged_state_dict[key] = all_params_for_key[0]
            self.feature_extractor.mlp4logfeats.load_state_dict(averaged_state_dict)

        return

    
    def get_feats(self):
        self.feature_extractor.to(self.args.device)
        session_id, seq, pos, neg = self.sampler.nextbatch(*self.sampler.get_next_back(self.user_id))
        with torch.no_grad():
            new_feats = self.feature_extractor.log2feats(seq)
        self.feature_extractor.to("cpu")
        return new_feats

    def get_valid_feats(self):
        self.feature_extractor.to(self.args.device)
        seq, item_index = self.sampler.get_valid()
        with torch.no_grad():
            new_feats = self.feature_extractor.log2feats(seq)
        self.feature_extractor.to("cpu")
        return new_feats

    def get_test_feats(self):
        self.feature_extractor.to(self.args.device)
        seq, item_index = self.sampler.get_test()
        with torch.no_grad():
            new_feats = self.feature_extractor.log2feats(seq)
        self.feature_extractor.to("cpu")
        return new_feats




    def step(self):  

        self.feature_extractor.to(self.args.device)



        epoch_start_idx = 1
        bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
        mse_criterion = torch.nn.MSELoss()
        adam_optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=self.args.lr, betas=(0.9, 0.98))
        transformer_params = []
        mlp_params = []

        for name, param in self.feature_extractor.named_parameters():
            if 'mlp4logfeats' in name:
                mlp_params.append(param)
            else:
                transformer_params.append(param)

        num_batch = (len(self.sampler.train['session_id'].unique()) - 1) // self.args.batch_size + 1
        self.session_id, seq, pos, neg = self.sampler.nextbatch(*self.sampler.get_nextsession(self.user_id))
        u = self.user_id
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)



        model_params_temple = self.feature_extractor.state_dict()  
        if self.model_params != None:
            self.feature_extractor.load_state_dict(self.model_params)  
            with torch.no_grad():
                old_feats = self.feature_extractor.log2feats(seq)
            self.feature_extractor.load_state_dict(model_params_temple)

            self.feature_extractor.train()  # enable model training
            for epoch in range(epoch_start_idx, self.args.num_epochs + 1):
                noise_std = self.args.rand  
                noise = torch.normal(mean=0.0,
                                     std=noise_std,
                                     size=self.other_feats.shape,
                                     device=self.args.device)
                other_feats_noisy = self.other_feats + noise

                pos_logits, neg_logits, new_feats = self.feature_extractor(u, seq, other_feats_noisy, pos, neg)
                pos_labels, neg_labels = (torch.ones(pos_logits.shape, device=self.args.device),
                                          torch.zeros(neg_logits.shape, device=self.args.device))
                assert not torch.isnan(pos_labels).any(), "pos_labels contains NaN!"
                assert not torch.isinf(pos_labels).any(), "pos_labels contains Inf!"
                assert (pos_labels >= 0).all() and (pos_labels <= 1).all(), "Invalid pos_labels!"
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                bce_loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                bce_loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in self.feature_extractor.item_emb.parameters(): bce_loss += self.args.l2_emb * torch.norm(
                    param)


                distillation_loss = mse_criterion(new_feats, old_feats.detach())  

                with torch.no_grad():
                    ratio = bce_loss.item() / (distillation_loss.item() + 1e-8) 
                scaled_distill_loss = min(ratio, self.args.distil_para) * distillation_loss
                total_loss = bce_loss + self.args.distillation_weight * scaled_distill_loss


                total_loss.backward()
        
                adam_optimizer.step()

                if epoch == 1 or epoch ==self.args.num_epochs:
            
                    print("loss in client{}, epoch {} session {}: {},{}".format(self.user_id, epoch, self.train_session_count,
                                                                         bce_loss.item(), distillation_loss))  # expected 0.4~0.6 after init few epochs

                if epoch == self.args.num_epochs:
                    self.com_feats = new_feats

        else:
            self.feature_extractor.train()  # enable model training
            for epoch in range(epoch_start_idx, self.args.num_epochs + 1):
                if self.other_feats == None:
                    avg_feats = torch.zeros(1, self.args.maxlen, self.args.hidden_units, dtype=torch.float32, device=self.args.device) 

                pos_logits, neg_logits, _ = self.feature_extractor(u, seq, avg_feats, pos, neg)
                pos_labels, neg_labels = (torch.ones(pos_logits.shape, device=self.args.device),
                                          torch.zeros(neg_logits.shape, device=self.args.device))
                
                assert not torch.isnan(pos_labels).any(), "pos_labels contains NaN!"
                assert not torch.isinf(pos_labels).any(), "pos_labels contains Inf!"
                
                assert (pos_labels >= 0).all() and (pos_labels <= 1).all(), "Invalid pos_labels!"
                
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
              
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in self.feature_extractor.item_emb.parameters(): loss += self.args.l2_emb * torch.norm(
                    param)
                loss.backward()
                adam_optimizer.step()
                if epoch == 1 or epoch == self.args.num_epochs:
                
                    print("loss in client{}, epoch {} session {}: {}".format(self.user_id, epoch,
                                                                         self.train_session_count,
                                                                         loss.item()))  # expected 0.4~0.6 after init few epochs
                if epoch == self.args.num_epochs:
                    with torch.no_grad():
                        self.com_feats = self.feature_extractor.log2feats(seq)

        self.feature_extractor.to("cpu")



        return

    def step_nom(self):  
        self.feature_extractor.to(self.args.device)




        epoch_start_idx = 1
        bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
        adam_optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=self.args.lr, betas=(0.9, 0.98))

        best_val_ndcg, best_val_hr = 0.0, 0.0
        best_test_ndcg, best_test_hr = 0.0, 0.0
        num_batch = (len(self.sampler.train['session_id'].unique()) - 1) // self.args.batch_size + 1


        T = 0.0
        t0 = time.time()
        self.session_id, seq, pos, neg = self.sampler.nextbatch(*self.sampler.get_nextsession(self.user_id))

        u = self.user_id
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        self.feature_extractor.train()  # enable model training
        for epoch in range(epoch_start_idx, self.args.num_epochs + 1):
        
            pos_logits, neg_logits, _ = self.feature_extractor(u, seq, pos, neg)
            pos_labels, neg_labels = (torch.ones(pos_logits.shape, device=self.args.device),
                                      torch.zeros(neg_logits.shape, device=self.args.device))
            assert not torch.isnan(pos_labels).any(), "pos_labels contains NaN!"
            assert not torch.isinf(pos_labels).any(), "pos_labels contains Inf!"
            assert (pos_labels >= 0).all() and (pos_labels <= 1).all(), "Invalid pos_labels!"
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in self.feature_extractor.item_emb.parameters(): loss += self.args.l2_emb * torch.norm(param)


            loss.backward()
            adam_optimizer.step()

            if epoch % 1 == 0:
                print("loss in client{}, epoch {} session {}: {}".format(self.user_id, epoch, self.train_session_count,
                                                                     loss.item()))  # expected 0.4~0.6 after init few epochs



    def write_logs(self):
        return












