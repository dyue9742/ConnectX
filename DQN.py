"""
1. Experience Replay
    § To help remove correlations, store dataset (called a replay buffer) D
    § To perform experience replay, repeat the following:
        ¶ (s,a,r,s')~D: sample an experience tuple from the dataset
        ¶ Compute the target value for the sampled s: r+γ*max(a')Q(s',a';w)
        ¶ Use stochastic gradient descent to update the network weights
            ∆w = α(r + γmax(a')Q¨(s',a';w) - Q¨(s,a;w))NablaQ¨(s,a;w)
2. Symmary
    § DQN use experience replay and fixed Q-targets
    § Store transition (s.t,a.t,r.(t+1),s.(t+1)) in replay memory D
    § Sample random mini-batch of transition (s,a,r,s') from D
    § Compute Q-learning targets w.r.t old, fixed parameters w-
    § Optimizes MSE between Q-network and Q-learning targets
    § Uses stochastic gradient descent
"""
