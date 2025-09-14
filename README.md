I. Training Time Predictor = "How long will this take?" 
II. Carbon Forecaster = "When is the grid cleanest?" 
III. GPU Optimizer = "What hardware is most efficient?"



1. Finding time required to train models on GPU's
    a. Chinchilla's Law states that time_to_train = (total_FLOPS)/ (GPU_FLOPS * efficiency) where total FlOPS is typically found as 6 * Parameters * Tokens * 3. However, this formula assumes efficiency is constant. Contrary to this, efficiency depends largely on memory utilization. In order to accurately measure time_to_train, we must account for this memory optimization. 

    forward_flops = 6 * params_b * 1e9
    backward_flops = 2 * forward_flops
    
    optimizer_flops = {
        'sgd': 1 * params_b * 1e9,
        'adam': 3 * params_b * 1e9,
        'adamw': 3 * params_b * 1e9
    }

    total_FLOPS = forward_flops + backward_flops + optimizer_flops

    b. Finding efficiency would rely on memory. The correlation is nearly exponential, so we can use a generic exponential model that matches real-life data on GPU efficiency and memory correlation
    
    c. We use the optimize and account for the Zero Redundancy Operator by estimating memory reduction factors across different stages of training
        Stage 0: No optimization (1.00x)
        Stage 1: Optimizer state partitioning (0.55x)
        Stage 2: Gradient + optimizer partitioning (0.40x)
        Stage 3: Full parameter partitioning (0.20x)
    
    d. Because accounting for memory to penalize efficiency doesn't take initiate that much of a change in our training time calculation, we can default to 0.2 and 0.5 for activation checkpointing to find model state bytes required. 

    e. Eventually, with we can determine an efficiency score that accounts for memory penalization and use our original formula, Chinchilla's law, to measure the estimated training time required. 


