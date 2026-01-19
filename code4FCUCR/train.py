import copy
from tqdm import tqdm
import time
import torch
import pandas as pd
import numpy as np
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
from utils.args import *
from client import *
from torch.utils.tensorboard import SummaryWriter
from aggreator import *
from model import *
import os
import logging



def init_client(args_):

    print("===> Building data iterators..")
    log, purchases, test = None, None, None
    
    test = pd.read_hdf("data/XING/session.hdf", key="test")
    train = pd.read_hdf("data/XING/session.hdf", key="train")
    valid = pd.read_hdf("data/XING/session.hdf", key="valid")


    train_item = set(train['item_id'].unique())
    valid_item = set(valid['item_id'].unique())
    test_item = set(test['item_id'].unique())

    all_item_ts = set(train_item | valid_item | test_item)
    print(len(all_item_ts))
    print(len(train))

    # train = train[:510]
    # print(
    #     train
    # )
    # time.sleep(5)





    print("======> Initializing clients..")
    clients = []
    for client_id in list(train.user_id.unique()):
        
        feature_extractor = SASRec(session_num=40, item_num=24390, args=args_)   # The current number of items is 24390
        # Model initialization
        for name, param in feature_extractor.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass
        feature_extractor.pos_emb.weight.data[0, :] = 0
        feature_extractor.item_emb.weight.data[0, :] = 0



        train_data = train[train["user_id"] == client_id]
        valid_data = valid[valid["user_id"] == client_id]
        test_data = test[test["user_id"] == client_id]

        data = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data
        }



        client_ = \
            client(
                data=data,
                all_item_ts= all_item_ts,
                user_id=client_id,
                feature_extractor=feature_extractor,
                local_steps=args_.local_steps,
                args= args_

            )        # Create several clients here

        





        clients.append(client_)
        print(f"Client{client_id} has been created" )
    print(f"A total of {train.user_id.nunique()} clients have been created")
    # time.sleep(10)

    # Create a list of clients here, which will be passed to the aggregator.
# Each time a client is selected from the aggregate class, it undergoes a round of training and is then passed back to the aggregate class.
    return clients

def run(args_):
    torch.manual_seed(args_.seed)

    print("==> Clients initialization..")
    
    clients = init_client(args_)

    print("==> Test Clients initialization..")

    global_model = SASRec(session_num=40, item_num=24390, args=args_)



    aggregator = CentralizedAggregator(
        clients=clients,
        log_freq=args_.log_freq,
        global_model=global_model,
        seed=42,
        args=args_
    )

    print("training now,,,,,,,,,")
    Pro_bar = tqdm(total=args_.n_rounds)
    valid_hr = 0.0
    valid_ndcg = 0.0
    test_hr = 0.0
    test_ndcg = 0.0
    logging.basicConfig(filename=args_.output, level=logging.INFO, format="%(asctime)s - %(message)s")

    whether_end = 0


    for current_round in range(args_.n_rounds):
        aggregator.mix()
        feats_dict = {}
        feats_vectors = []
        session_dict = {}

        # Prepare KB_feats
        for i, (session_id, state_dict) in enumerate(aggregator.KB_feats.items()):
            feats_vector = torch.cat([state_dict.view(-1)])
            feats_dict[session_id] = feats_vector
            session_dict[i] = session_id
            feats_vectors.append(feats_vector.cpu().numpy())

        # Prepare client valid/test prototype
        client_valid_vectors, client_test_vectors = [], []
        valid_clients, test_clients = [], []

        for client in aggregator.clients:
            if client.have_valid:
                client_feat = client.get_valid_feats().view(-1)
                client_valid_vectors.append(client_feat.cpu().numpy())
                valid_clients.append(client)
            if client.have_test:
                client_feat = client.get_test_feats().view(-1)
                client_test_vectors.append(client_feat.cpu().numpy())
                test_clients.append(client)
        # Similarity
        scores_valid = cosine_similarity(client_valid_vectors, feats_vectors) if client_valid_vectors else []
        scores_test = cosine_similarity(client_test_vectors, feats_vectors) if client_test_vectors else []
        # Traverse the client to evaluate valid/test
        cur_hr, cur_ndcg, cur_user = 0.0, 0.0, 0
        res_hr, res_ndcg, test_user = 0.0, 0.0, 0

        for idx, client in enumerate(aggregator.clients):
            client.feature_extractor.to(client.args.device)
            client.feature_extractor.eval()
               # valid
            if client in valid_clients:
                score = scores_valid[valid_clients.index(client)]
                sorted_idx = np.argsort(score)[::-1]
                top5_indices = [i for i in sorted_idx if session_dict[i] not in client.user_session][:args_.KB_len]

                if top5_indices:
                    session_top5 = [(session_dict[i], score[i]) for i in top5_indices]
                    avg_feats = torch.zeros_like(aggregator.KB_feats[session_top5[0][0]])
                    weights = torch.nn.functional.softmax(torch.tensor([s for _, s in session_top5]), dim=0)
                    for i, (session_id, _) in enumerate(session_top5):
                        avg_feats += weights[i] * aggregator.KB_feats[session_id]
                    valid_other_feats = avg_feats
                    if args_.pca:
                        feats_list = [aggregator.KB_feats[session_id] for session_id, _ in session_top5]
                        valid_other_feats = enhance_sessions_with_pca(feats_list)
                else:
                    valid_other_feats = None
                    print("valid none")
                noise_std = args_.rand  # Set the degree of privacy protection based on the actual situation of the task

                
                noise = torch.normal(mean=0.0,
                                     std=noise_std,
                                     size=valid_other_feats.shape,
                                     device=args_.device)


                
                valid_other_feats_noisy = valid_other_feats + noise
                res = evaluate_valid(client.feature_extractor, client.sampler, client.user_id, valid_other_feats_noisy, client.args)
                print(f"Valid: client{client.user_id}, NDCG@10: {res[0]:.4f}, HR@10: {res[1]:.4f}")
                cur_ndcg += res[0]
                cur_hr += res[1]
                cur_user += 1
                # test
            if client in test_clients:
                score = scores_test[test_clients.index(client)]
                sorted_idx = np.argsort(score)[::-1]
                top5_indices = [i for i in sorted_idx if session_dict[i] not in client.user_session][:args_.KB_len]

                if top5_indices:
                    session_top5 = [(session_dict[i], score[i]) for i in top5_indices]
                    avg_feats = torch.zeros_like(aggregator.KB_feats[session_top5[0][0]])
                    weights = torch.nn.functional.softmax(torch.tensor([s for _, s in session_top5]), dim=0)
                    for i, (session_id, _) in enumerate(session_top5):
                        avg_feats += weights[i] * aggregator.KB_feats[session_id]
                    test_other_feats = avg_feats
                    if args_.pca:
                        feats_list = [aggregator.KB_feats[session_id] for session_id, _ in session_top5]
                        test_other_feats = enhance_sessions_with_pca(feats_list)
                else:
                    test_other_feats = None
                    print("test none")
                noise_std = args_.rand  # Set the degree of privacy protection based on the actual situation of the task

                
                noise = torch.normal(mean=0.0,
                                     std=noise_std,
                                     size=test_other_feats.shape,
                                     device=args_.device)

                
                test_other_feats_noisy = test_other_feats + noise

                res = evaluate(client.feature_extractor, client.sampler, client.user_id, test_other_feats_noisy, client.args)
                print(f"Test: client{client.user_id}, NDCG@10: {res[0]:.4f}, HR@10: {res[1]:.4f}")
                res_ndcg += res[0]
                res_hr += res[1]
                test_user += 1

            client.feature_extractor.to('cpu')
            # Calculate average indicators
        cur_ndcg = cur_ndcg / cur_user if cur_user else 0.0
        cur_hr = cur_hr / cur_user if cur_user else 0.0
        res_ndcg = res_ndcg / test_user if test_user else 0.0
        res_hr = res_hr / test_user if test_user else 0.0

        # early stop
        if cur_hr > valid_hr or cur_ndcg > valid_ndcg:
            valid_hr, valid_ndcg = max(valid_hr, cur_hr), max(valid_ndcg, cur_ndcg)
            whether_end = 0
        else:
            whether_end += 1

        # output
        print(f"Valid AVG: NDCG@10:{cur_ndcg:.4f}, HR@10:{cur_hr:.4f}")
        print(f"Final test result: NDCG@10:{res_ndcg:.4f}, HR@10:{res_hr:.4f}")
        logging.info(f"The current round is {current_round}, and there have been {whether_end} rounds without growth.")
        logging.info(f"Valid AVG: NDCG@10:{cur_ndcg}, HR@10:{cur_hr}")
        logging.info(f"Test AVG: NDCG@10:{res_ndcg}, HR@10:{res_hr}")
        logging.info(f"Related parameters:{args_to_string(args_)}")





        Pro_bar.update(1)

    # The process is finished

    logging.info(f"Related parameters:{args_to_string(args_)}")




























if __name__ == "__main__":
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = parse_args()
    run(args)





















