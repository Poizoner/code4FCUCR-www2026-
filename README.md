# code4FCUCR-www2026-
implementation of "Learning Evolving Preferences: A Federated Continual Framework for User-Centric Recommendation" (WWW 2026).
# Learning Evolving Preferences: A Federated Continual Framework for User-Centric Recommendation
This repository is the implementation of *Learning Evolving Preferences: A Federated Continual Framework for User-Centric Recommendation*

## Abstract
User-centric recommendation has become essential for delivering personalized services, as it enables systems to adapt to users’ evolving behaviors while respecting their long-term preferences and privacy constraints. Although federated learning offers a promising alternative to centralized training, existing approaches largely overlook user behavior dynamics, leading to temporal forgetting and weakened collaborative personalization. In this work, we propose \model, the first federated continual recommendation framework designed to support long-term personalization in a privacy-preserving manner. To address temporal forgetting, we introduce a time-aware self-distillation strategy that implicitly retains historical preferences during local model updates. To tackle collaborative personalization under heterogeneous user data, we design an inter-user prototype transfer mechanism that enriches each client’s representation using knowledge from similar users while preserving individual decision logic. Extensive experiments on four public benchmarks demonstrate the superior effectiveness of our approach, along with strong compatibility and practical applicability.

## Requirement
To install requirements:
```
pip install -r requirements.txt
```
## Usage
We provide code to simulate federated training of machine learning. 
The core components of this framework are `Aggregator` and `Client`.  
Different federated learning algorithms can be implemented by modifying:

-   the local update behavior in `Client.step()`
    
-   and/or the aggregation protocol in `Aggregator.mix()` and `Aggregator.update_client()`.
    

In addition to traditional model communication, this framework also supports **knowledge-enhanced federated learning**:  
A global **Knowledge Base (KB)** is maintained on the server side to store domain-specific or user-related information. The server can utilize this KB during aggregation (e.g., prototype fusion, knowledge-guided model merging), and optionally deliver relevant knowledge snippets to clients to assist local personalization.

## Datasets
We evaluate our method on four widely-used real-world benchmark datasets:

-   **[XING]** : User-job interaction dataset released by the RecSys Challenge 2017.
    
-   **[RetailRocket]**: An e-commerce dataset containing user behavior logs from a real online retail platform.
    
-   **[Tmall]**: User purchase behavior data from Alibaba’s Tmall platform.
    
-   **[LastFM]**: Music listening history of users.
    

These datasets span different domains (e.g., job recommendation, e-commerce, music) and are commonly used in session-based or sequential recommendation research.

## Training
Run on XING dataset, and configure all other hyper-parameters. (see all hyper-parameters values in utils/args.py)
```
python train.py\
--logs_dir log\
--method my\
--KB_len 20\
--distillation_weight 1.0\
--distil_para 10.0\
--lr 1e-1\
--num_epochs 4\
--device cuda:0\
--output XING_result.txt
```

-   `--n_rounds`: Number of federated communication rounds.
    
-   `--batch_size`: Mini-batch size used for training on each client.
    
-   `--lr`: Learning rate for the optimizer.
    
-   `--maxlen`: Maximum sequence length for input data (e.g., user interaction history).
    
-   `--hidden_units`: Dimension of hidden representation in the model.
    
-   `--num_blocks`: Number of transformer blocks used in the model.
    
-   `--num_epochs`: Number of training epochs per client per round.
    
-   `--num_heads`: Number of attention heads in each transformer block.
    
-   `--dropout_rate`: Dropout probability used in model layers.
    
-   `--l2_emb`: L2 regularization strength applied to embedding weights.
    
-   `--distillation_weight`: Weight of the distillation loss in total loss function.
    
-   `--distil_para`: Hyperparameter controlling temperature or influence of knowledge distillation.
    
-   `--rand`: Random noise level or proportion used for ablation or perturbation.
    
-   `--device`: Computing device to use (`cuda` or `cpu`).
    
-   `--method`: Name of the training method or experiment variant to run.
    
-   `--KB_len`: Number of knowledge base entries to be retrieved or used per client.
    
-   `--KB_all`: If set to `0`, the full knowledge base is stored; if greater than `0`, limits the total size of the knowledge base to the given value.
    
-   `--update_round`: Frequency (in rounds) for updating auxiliary modules (e.g., KB or prompt).
    
-   `--pca`: Whether to apply PCA for dimensionality reduction (flag, default: `False`).
    
-   `--output`: Path to the file for saving evaluation results or logs.
