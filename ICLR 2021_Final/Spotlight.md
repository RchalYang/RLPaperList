# Spotlight

## Model-Based Visual Planning with Self-Supervised Functional Distances

Authors: ['Stephen Tian', 'Suraj Nair', 'Frederik Ebert', 'Sudeep Dasari', 'Benjamin Eysenbach', 'Chelsea Finn', 'Sergey Levine']

Ratings: ['7: Good paper, accept', '7: Good paper, accept', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['planning', 'model learning', 'distance learning', 'reinforcement learning', 'robotics']

A generalist robot must be able to complete a variety of tasks in its environment. One appealing way to specify each task is in terms of a goal observation. However, learning goal-reaching policies with reinforcement learning remains a challenging problem, particularly when hand-engineered reward functions are not available. Learned dynamics models are a promising approach for learning about the environment without rewards or task-directed data, but planning to reach goals with such a model requires a notion of functional similarity between observations and goal states. We present a self-supervised method for model-based visual goal reaching, which uses both a visual dynamics model as well as a dynamical distance function learned using model-free reinforcement learning. Our approach learns entirely using offline, unlabeled data, making it practical to scale to large and diverse datasets. In our experiments, we find that our method can successfully learn models that perform a variety of tasks at test-time, moving objects amid distractors with a simulated robotic arm and even learning to open and close a drawer using a real-world robot. In comparisons, we find that this approach substantially outperforms both model-free and model-based prior methods.

## Self-Supervised Policy Adaptation during Deployment

Authors: ['Nicklas Hansen', 'Rishabh Jangir', 'Yu Sun', 'Guillem Alenyà', 'Pieter Abbeel', 'Alexei A Efros', 'Lerrel Pinto', 'Xiaolong Wang']

Ratings: ['7: Good paper, accept', '7: Good paper, accept', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['reinforcement learning', 'robotics', 'self-supervised learning', 'generalization', 'sim2real']

In most real world scenarios, a policy trained by reinforcement learning in one environment needs to be deployed in another, potentially quite different environment. However, generalization across different environments is known to be hard. A natural solution would be to keep training after deployment in the new environment, but this cannot be done if the new environment offers no reward signal. Our work explores the use of self-supervision to allow the policy to continue training after deployment without using any rewards. While previous methods explicitly anticipate changes in the new environment, we assume no prior knowledge of those changes yet still obtain significant improvements. Empirical evaluations are performed on diverse simulation environments from DeepMind Control suite and ViZDoom, as well as real robotic manipulation tasks in  continuously changing environments, taking observations from an uncalibrated camera. Our method improves generalization in 31 out of 36 environments across various tasks and outperforms domain randomization on a majority of environments. Webpage and implementation: https://nicklashansen.github.io/PAD/.

## What are the Statistical Limits of Offline RL with Linear Function Approximation?

Authors: ['Ruosong Wang', 'Dean Foster', 'Sham M. Kakade']

Ratings: ['7: Good paper, accept', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['batch reinforcement learning', 'function approximation', 'lower bound', 'representation']

Offline reinforcement learning seeks to utilize offline (observational) data to guide the learning of (causal) sequential decision making strategies. The hope is that offline reinforcement learning coupled with function approximation methods (to deal with the curse of dimensionality) can provide a means to help alleviate the excessive sample complexity burden in modern sequential decision making problems. However, the extent to which this broader approach can be effective is not well understood, where the literature largely consists of sufficient conditions.

This work focuses on the basic question of what are necessary representational and distributional conditions that permit provable sample-efficient offline reinforcement learning. Perhaps surprisingly, our main result shows that even if: i) we have realizability in that the true value function of \emph{every} policy is linear in a given set of features and 2) our off-policy data has good  coverage over all features (under a strong spectral condition), any algorithm still (information-theoretically) requires a number of offline samples that is exponential in the problem horizon to non-trivially estimate the value of \emph{any} given policy. Our results highlight that sample-efficient offline policy evaluation is not possible unless significantly stronger conditions hold; such conditions include either having low distribution shift (where the offline data distribution is close to the distribution of the policy to be evaluated) or significantly stronger representational conditions (beyond realizability).

## Regularized Inverse Reinforcement Learning

Authors: ['Wonseok Jeon', 'Chen-Yang Su', 'Paul Barde', 'Thang Doan', 'Derek Nowrouzezahrai', 'Joelle Pineau']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['inverse reinforcement learning', 'reward learning', 'regularized markov decision processes', 'reinforcement learning']

Inverse Reinforcement Learning (IRL) aims to facilitate a learner’s ability to imitate expert behavior by acquiring reward functions that explain the expert’s decisions. Regularized IRLapplies strongly convex regularizers to the learner’s policy in order to avoid the expert’s behavior being rationalized by arbitrary constant rewards, also known as degenerate solutions. We propose tractable solutions, and practical methods to obtain them, for regularized IRL. Current methods are restricted to the maximum-entropy IRL framework, limiting them to Shannon-entropy regularizers, as well as proposing solutions that are intractable in practice.  We present theoretical backing for our proposed IRL method’s applicability to both discrete and continuous controls, empirically validating our performance on a variety of tasks.

## Correcting experience replay for multi-agent communication

Authors: ['Sanjeevan Ahilan', 'Peter Dayan']

Ratings: ['7: Good paper, accept', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['multi-agent reinforcement learning', 'experience replay', 'communication', 'relabelling']

We consider the problem of learning to communicate using multi-agent reinforcement learning (MARL). A common approach is to learn off-policy, using data sampled from a replay buffer. However, messages received in the past may not accurately reflect the current communication policy of each agent, and this complicates learning. We therefore introduce a 'communication correction' which accounts for the non-stationarity of observed communication induced by multi-agent learning. It works by relabelling the received message to make it likely under the communicator's current policy, and thus be a better reflection of the receiver's current environment. To account for cases in which agents are both senders and receivers, we introduce an ordered relabelling scheme. Our correction is computationally efficient and can be integrated with a range of off-policy algorithms. We find in our experiments that it substantially improves the ability of communicating MARL systems to learn across a variety of cooperative and competitive tasks.

## Winning the L2RPN Challenge: Power Grid Management via Semi-Markov Afterstate Actor-Critic

Authors: ['Deunsol Yoon', 'Sunghoon Hong', 'Byung-Jun Lee', 'Kee-Eung Kim']

Ratings: ['7: Good paper, accept', '7: Good paper, accept', '7: Good paper, accept', '9: Top 15% of accepted papers, strong accept']

Keywords: ['power grid management', 'deep reinforcement learning', 'graph neural network']

Safe and reliable electricity transmission in power grids is crucial for modern society. It is thus quite natural that there has been a growing interest in the automatic management of power grids, exempliﬁed by the Learning to Run a Power Network Challenge (L2RPN), modeling the problem as a reinforcement learning (RL) task. However, it is highly challenging to manage a real-world scale power grid, mostly due to the massive scale of its state and action space. In this paper, we present an off-policy actor-critic approach that effectively tackles the unique challenges in power grid management by RL, adopting the hierarchical policy together with the afterstate representation. Our agent ranked ﬁrst in the latest challenge (L2RPN WCCI 2020), being able to avoid disastrous situations while maintaining the highest level of operational efﬁciency in every test scenarios. This paper provides a formal description of the algorithmic aspect of our approach, as well as further experimental studies on diverse power grids.

## Self-supervised Visual Reinforcement Learning with Object-centric Representations

Authors: ['Andrii Zadaianchuk', 'Maximilian Seitzer', 'Georg Martius']

Ratings: ['5: Marginally below acceptance threshold', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept', '9: Top 15% of accepted papers, strong accept']

Keywords: ['self-supervision', 'autonomous learning', 'object-centric representations', 'visual reinforcement learning']

Autonomous agents need large repertoires of skills to act reasonably on new tasks that they have not seen before. However, acquiring these skills using only a stream of high-dimensional, unstructured, and unlabeled observations is a tricky challenge for any autonomous agent. Previous methods have used variational autoencoders to encode a scene into a low-dimensional vector that can be used as a goal for an agent to discover new skills. Nevertheless, in compositional/multi-object environments it is difficult to disentangle all the factors of variation into such a fixed-length representation of the whole scene. We propose to use object-centric representations as a modular and structured observation space, which is learned with a compositional generative world model.
We show that the structure in the representations in combination with goal-conditioned attention policies helps the autonomous agent to discover and learn useful skills. These skills can be further combined to address compositional tasks like the manipulation of several different objects.

## Quantifying Differences in Reward Functions

Authors: ['Adam Gleave', 'Michael D Dennis', 'Shane Legg', 'Stuart Russell', 'Jan Leike']

Ratings: ['6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['rl', 'irl', 'reward learning', 'distance', 'benchmarks']

For many tasks, the reward function is inaccessible to introspection or too complex to be specified procedurally, and must instead be learned from user data. Prior work has evaluated learned reward functions by evaluating policies optimized for the learned reward. However, this method cannot distinguish between the learned reward function failing to reflect user preferences and the policy optimization process failing to optimize the learned reward. Moreover, this method can only tell us about behavior in the evaluation environment, but the reward may incentivize very different behavior in even a slightly different deployment environment. To address these problems, we introduce the Equivalent-Policy Invariant Comparison (EPIC) distance to quantify the difference between two reward functions directly, without a policy optimization step. We prove EPIC is invariant on an equivalence class of reward functions that always induce the same optimal policy. Furthermore, we find EPIC can be efficiently approximated and is more robust than baselines to the choice of coverage distribution. Finally, we show that EPIC distance bounds the regret of optimal policies even under different transition dynamics, and we confirm empirically that it predicts policy training success. Our source code is available at https://github.com/HumanCompatibleAI/evaluating-rewards.

