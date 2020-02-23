# Talk

## Contrastive Learning of Structured World Models

Authors: ['Thomas Kipf', 'Elise van der Pol', 'Max Welling']

Ratings: ['8: Accept', '8: Accept', '8: Accept']

Keywords: ['state representation learning', 'graph neural networks', 'model-based reinforcement learning', 'relational learning', 'object discovery']

A structured understanding of our world in terms of objects, relations, and hierarchies is an important component of human cognition. Learning such a structured world model from raw sensory data remains a challenge. As a step towards this goal, we introduce Contrastively-trained Structured World Models (C-SWMs). C-SWMs utilize a contrastive approach for representation learning in environments with compositional structure. We structure each state embedding as a set of object representations and their relations, modeled by a graph neural network. This allows objects to be discovered from raw pixel observations without direct supervision as part of the learning process. We evaluate C-SWMs on compositional environments involving multiple interacting objects that can be manipulated independently by an agent, simple Atari games, and a multi-object physics simulation. Our experiments demonstrate that C-SWMs can overcome limitations of models based on pixel reconstruction and outperform typical representatives of this model class in highly structured environments, while learning interpretable object-based representations.

## Dynamics-Aware Unsupervised Skill Discovery

Authors: ['Archit Sharma', 'Shixiang Gu', 'Sergey Levine', 'Vikash Kumar', 'Karol Hausman']

Ratings: ['8: Accept', '8: Accept', '8: Accept']

Keywords: ['reinforcement learning', 'unsupervised learning', 'model-based learning', 'deep learning', 'hierarchical reinforcement learning']

Conventionally, model-based reinforcement learning (MBRL) aims to learn a global model for the dynamics of the environment. A good model can potentially enable planning algorithms to generate a large variety of behaviors and solve diverse tasks. However, learning an accurate model for complex dynamical systems is difficult, and even then, the model might not generalize well outside the distribution of states on which it was trained. In this work, we combine model-based learning with model-free learning of primitives that make model-based planning easy. To that end, we aim to answer the question: how can we discover skills whose outcomes are easy to predict? We propose an unsupervised learning algorithm, Dynamics-Aware Discovery of Skills (DADS), which simultaneously discovers predictable behaviors and learns their dynamics. Our method can leverage continuous skill spaces, theoretically, allowing us to learn infinitely many behaviors even for high-dimensional state-spaces. We demonstrate that zero-shot planning in the learned latent space significantly outperforms standard MBRL and model-free goal-conditioned RL, can handle sparse-reward tasks, and substantially improves over prior hierarchical RL methods for unsupervised skill discovery.

## Harnessing Structures for Value-Based Planning and Reinforcement Learning

Authors: ['Yuzhe Yang', 'Guo Zhang', 'Zhi Xu', 'Dina Katabi']

Ratings: ['6: Weak Accept', '8: Accept', '8: Accept']

Keywords: ['Deep reinforcement learning', 'value-based reinforcement learning']

Value-based methods constitute a fundamental methodology in planning and deep reinforcement learning (RL). In this paper, we propose to exploit the underlying structures of the state-action value function, i.e., Q function, for both planning and deep RL. In particular, if the underlying system dynamics lead to some global structures of the Q function, one should be capable of inferring the function better by leveraging such structures. Specifically, we investigate the low-rank structure, which widely exists for big data matrices. We verify empirically the existence of low-rank Q functions in the context of control and deep RL tasks. As our key contribution, by leveraging Matrix Estimation (ME) techniques, we propose a general framework to exploit the underlying low-rank structure in Q functions. This leads to a more efficient planning procedure for classical control, and additionally, a simple scheme that can be applied to value-based RL techniques to consistently achieve better performance on "low-rank" tasks. Extensive experiments on control tasks and Atari games confirm the efficacy of our approach.

## Meta-Q-Learning

Authors: ['Rasool Fakoor', 'Pratik Chaudhari', 'Stefano Soatto', 'Alexander J. Smola']

Ratings: ['6: Weak Accept', '8: Accept', '8: Accept']

Keywords: ['meta reinforcement learning', 'propensity estimation', 'off-policy']

This paper introduces Meta-Q-Learning (MQL), a new off-policy algorithm for meta-Reinforcement Learning (meta-RL). MQL builds upon three simple ideas. First, we show that Q-learning is competitive with state-of-the-art meta-RL algorithms if given access to a context variable that is a representation of the past trajectory. Second, a multi-task objective to maximize the average reward across the training tasks is an effective method to meta-train RL policies. Third, past data from the meta-training replay buffer can be recycled to adapt the policy on a new task using off-policy updates. MQL draws upon ideas in propensity estimation to do so and thereby amplifies the amount of available data for adaptation. Experiments on standard continuous-control benchmarks suggest that MQL compares favorably with the state of the art in meta-RL.

## A Closer Look at Deep Policy Gradients

Authors: ['Andrew Ilyas', 'Logan Engstrom', 'Shibani Santurkar', 'Dimitris Tsipras', 'Firdaus Janoos', 'Larry Rudolph', 'Aleksander Madry']

Ratings: ['6: Weak Accept', '8: Accept', '8: Accept']

Keywords: ['deep policy gradient methods', 'deep reinforcement learning', 'trpo', 'ppo']

    We study how the behavior of deep policy gradient algorithms reflects the conceptual framework motivating their development. To this end, we propose a fine-grained analysis of state-of-the-art methods based on key elements of this framework: gradient estimation, value prediction, and optimization landscapes. Our results show that the behavior of deep policy gradient algorithms often deviates from what their motivating framework would predict: surrogate rewards do not match the true reward landscape, learned value estimators fail to fit the true value function, and gradient estimates poorly correlate with the "true" gradient. The mismatch between predicted and empirical behavior we uncover highlights our poor understanding of current methods, and indicates the need to move beyond current benchmark-centric evaluation methods.

## Implementation Matters in Deep RL: A Case Study on PPO and TRPO

Authors: ['Logan Engstrom', 'Andrew Ilyas', 'Shibani Santurkar', 'Dimitris Tsipras', 'Firdaus Janoos', 'Larry Rudolph', 'Aleksander Madry']

Ratings: ['8: Accept', '8: Accept', '8: Accept']

Keywords: ['deep policy gradient methods', 'deep reinforcement learning', 'trpo', 'ppo']

We study the roots of algorithmic progress in deep policy gradient algorithms through a case study on two popular algorithms, Proximal Policy Optimization and Trust Region Policy Optimization. We investigate the consequences of "code-level optimizations:" algorithm augmentations found only in implementations or described as auxiliary details to the core algorithm. Seemingly of secondary importance, such optimizations have a major impact on agent behavior. Our results show that they (a) are responsible for most of PPO's gain in cumulative reward over TRPO, and (b) fundamentally change how RL methods function. These insights show the difficulty, and importance, of attributing performance gains in deep reinforcement learning.

## Causal Discovery with Reinforcement Learning

Authors: ['Shengyu Zhu', 'Ignavier Ng', 'Zhitang Chen']

Ratings: ['8: Accept', '8: Accept', '8: Accept']

Keywords: ['causal discovery', 'structure learning', 'reinforcement learning', 'directed acyclic graph']

Discovering causal structure among a set of variables is a fundamental problem in many empirical sciences. Traditional score-based casual discovery methods rely on various local heuristics to search for a Directed Acyclic Graph (DAG) according to a predefined score function. While these methods, e.g., greedy equivalence search, may have attractive results with infinite samples and certain model assumptions, they are less satisfactory in practice due to finite data and possible violation of assumptions. Motivated by recent advances in neural combinatorial optimization, we propose to use Reinforcement Learning (RL) to search for the DAG with the best scoring. Our encoder-decoder model takes observable data as input and generates graph adjacency matrices that are used to compute rewards. The reward incorporates both the predefined score function and two penalty terms for enforcing acyclicity. In contrast with typical RL applications where the goal is to learn a policy, we use RL as a search strategy and our final output would be the graph, among all graphs generated during training, that achieves the best reward. We conduct experiments on both synthetic and real datasets, and show that the proposed approach not only has an improved search ability but also allows for a flexible score function under the acyclicity constraint. 

## SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference

Authors: ['Lasse Espeholt', 'Raphaël Marinier', 'Piotr Stanczyk', 'Ke Wang', 'Marcin Michalski\u200e']

Ratings: ['6: Weak Accept', '8: Accept', '8: Accept']

Keywords: ['machine learning', 'reinforcement learning', 'scalability', 'distributed', 'DeepMind Lab', 'ALE', 'Atari-57', 'Google Research Football']

We present a modern scalable reinforcement learning agent called SEED (Scalable, Efficient Deep-RL). By effectively utilizing modern accelerators, we show that it is not only possible to train on millions of frames per second but also to lower the cost. of experiments compared to current methods. We achieve this with a simple architecture that features centralized inference and an optimized communication layer. SEED adopts two state-of-the-art distributed algorithms, IMPALA/V-trace (policy gradients) and R2D2 (Q-learning), and is evaluated on Atari-57, DeepMind Lab and Google Research Football. We improve the state of the art on Football and are able to reach state of the art on Atari-57 twice as fast in wall-time. For the scenarios we consider, a 40% to 80% cost reduction for running experiments is achieved. The implementation along with experiments is open-sourced so results can be reproduced and novel ideas tried out.

