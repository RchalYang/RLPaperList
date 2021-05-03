# Poster

## Optimism in Reinforcement Learning with Generalized Linear Function Approximation

Authors: ['Yining Wang', 'Ruosong Wang', 'Simon Shaolei Du', 'Akshay Krishnamurthy']

Ratings: ['5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept']

Keywords: ['reinforcement learning', 'optimism', 'exploration', 'function approximation', 'theory', 'regret analysis', 'provable sample efficiency']

We design a new provably efficient algorithm for episodic reinforcement learning with generalized linear function approximation. We analyze the algorithm under a new expressivity assumption that we call ``optimistic closure,'' which is strictly weaker than assumptions from prior analyses for the linear setting. With optimistic closure, we prove that our algorithm enjoys a regret bound of $\widetilde{O}\left(H\sqrt{d^3 T}\right)$ where $H$ is the horizon, $d$ is the dimensionality of the state-action features and $T$ is the number of episodes. This is the first statistically and computationally efficient algorithm for reinforcement learning with generalized linear functions.

## CausalWorld: A Robotic Manipulation Benchmark for Causal Structure and Transfer Learning

Authors: ['Ossama Ahmed', 'Frederik Träuble', 'Anirudh Goyal', 'Alexander Neitz', 'Manuel Wuthrich', 'Yoshua Bengio', 'Bernhard Schölkopf', 'Stefan Bauer']

Ratings: ['4: Ok but not good enough - rejection', '6: Marginally above acceptance threshold', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['reinforcement learning', 'transfer learning', 'sim2real transfer', 'domain adaptation', 'causality', 'generalization', 'robotics']

Despite recent successes of reinforcement learning (RL), it remains a challenge for agents to transfer learned skills to related environments. To facilitate research addressing this problem, we proposeCausalWorld, a benchmark for causal structure and transfer learning in a robotic manipulation environment. The environment is a simulation of an open-source robotic platform, hence offering the possibility of sim-to-real transfer. Tasks consist of constructing 3D shapes from a set of blocks - inspired by how children learn to build complex structures. The key strength of CausalWorld is that it provides a combinatorial family of such tasks with common causal structure and underlying factors (including, e.g., robot and object masses, colors, sizes). The user (or the agent) may intervene on all causal variables, which allows  for  fine-grained  control  over  how  similar  different  tasks  (or  task  distributions) are. One can thus easily define training and evaluation distributions of a desired difficulty level,  targeting a specific form of generalization (e.g.,  only changes in appearance or object mass). Further, this common parametrization facilitates defining curricula by interpolating between an initial and a target task. While users may define their own task distributions, we present eight meaningful distributions as concrete benchmarks, ranging from simple to very challenging, all of which require long-horizon planning as well as precise low-level motor control. Finally, we provide baseline results for a subset of these tasks on distinct training curricula and corresponding evaluation protocols, verifying the feasibility of the tasks in this benchmark.

## C-Learning: Learning to Achieve Goals via Recursive Classification

Authors: ['Benjamin Eysenbach', 'Ruslan Salakhutdinov', 'Sergey Levine']

Ratings: ['4: Ok but not good enough - rejection', '6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['reinforcement learning', 'goal reaching', 'density estimation', 'Q-learning', 'hindsight relabeling']

We study the problem of predicting and controlling the future state distribution of an autonomous agent. This problem, which can be viewed as a reframing of goal-conditioned reinforcement learning (RL), is centered around learning a conditional probability density function over future states. Instead of directly estimating this density function, we indirectly estimate this density function by training a classifier to predict whether an observation comes from the future. Via Bayes' rule, predictions from our classifier can be transformed into predictions over future states. Importantly, an off-policy variant of our algorithm allows us to predict the future state distribution of a new policy, without collecting new experience. This variant allows us to optimize functionals of a policy's future state distribution, such as the density of reaching a particular goal state. While conceptually similar to Q-learning, our work lays a principled foundation for goal-conditioned RL as density estimation, providing justification for goal-conditioned methods used in prior work. This foundation makes hypotheses about Q-learning, including the optimal goal-sampling ratio, which we confirm experimentally. Moreover, our proposed method is competitive with prior goal-conditioned RL methods.

## Robust Reinforcement Learning on State Observations with Learned Optimal Adversary

Authors: ['Huan Zhang', 'Hongge Chen', 'Duane S Boning', 'Cho-Jui Hsieh']

Ratings: ['6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['reinforcement learning', 'robustness', 'adversarial attacks', 'adversarial defense']

We study the robustness of reinforcement learning (RL) with adversarially perturbed state observations, which aligns with the setting of many adversarial attacks to deep reinforcement learning (DRL) and is also important for rolling out real-world RL agent under unpredictable sensing noise. With a fixed agent policy, we demonstrate that an optimal adversary to perturb state observations can be found, which is guaranteed to obtain the worst case agent reward. For DRL settings, this leads to a novel empirical adversarial attack to RL agents via a learned adversary that is much stronger than previous ones. To enhance the robustness of an agent, we propose a framework of alternating training with learned adversaries (ATLA), which trains an adversary online together with the agent using policy gradient following the optimal adversarial attack framework. Additionally, inspired by the analysis of state-adversarial Markov decision process (SA-MDP), we show that past states and actions (history) can be useful for learning a robust agent, and we empirically find a LSTM based policy can be more robust under adversaries. Empirical evaluations on a few continuous control environments show that ATLA achieves state-of-the-art performance under strong adversaries. Our code is available at https://github.com/huanzhang12/ATLA_robust_RL.

## Balancing Constraints and Rewards with Meta-Gradient D4PG

Authors: ['Dan A. Calian', 'Daniel J Mankowitz', 'Tom Zahavy', 'Zhongwen Xu', 'Junhyuk Oh', 'Nir Levine', 'Timothy Mann']

Ratings: ['6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['reinforcement learning', 'meta-gradients', 'constraints']

Deploying Reinforcement Learning (RL) agents to solve real-world applications often requires satisfying complex system constraints. Often the constraint thresholds are incorrectly set due to the complex nature of a system or the inability to verify the thresholds offline (e.g, no simulator or reasonable offline evaluation procedure exists). This results in solutions where a task cannot be solved without violating the constraints. However, in many real-world cases, constraint violations are undesirable yet they are not catastrophic, motivating the need for soft-constrained RL approaches. We present two soft-constrained RL approaches that utilize meta-gradients to find a good trade-off between expected return and minimizing constraint violations. We demonstrate the effectiveness of these approaches by showing that they consistently outperform the baselines across four different Mujoco domains.

## The Importance of Pessimism in Fixed-Dataset Policy Optimization

Authors: ['Jacob Buckman', 'Carles Gelada', 'Marc G Bellemare']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept']

Keywords: ['deep learning', 'reinforcement learning', 'offline reinforcement learning']

We study worst-case guarantees on the expected return of fixed-dataset policy optimization algorithms. Our core contribution is a unified conceptual and mathematical framework for the study of algorithms in this regime. This analysis reveals that for naive approaches, the possibility of erroneous value overestimation leads to a difficult-to-satisfy requirement: in order to guarantee that we select a policy which is near-optimal, we may need the dataset to be informative of the value of every policy. To avoid this, algorithms can follow the pessimism principle, which states that we should choose the policy which acts optimally in the worst possible world. We show why pessimistic algorithms can achieve good performance even when the dataset is not informative of every policy, and derive families of algorithms which follow this principle. These theoretical findings are validated by experiments on a tabular gridworld, and deep learning experiments on four MinAtar environments.

## Learning to Represent Action Values as a Hypergraph on the Action Vertices

Authors: ['Arash Tavakoli', 'Mehdi Fatemi', 'Petar Kormushev']

Ratings: ['5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['reinforcement learning', 'structural credit assignment', 'structural inductive bias', 'multi-dimensional discrete action spaces', 'learning action representations']

Action-value estimation is a critical component of many reinforcement learning (RL) methods whereby sample complexity relies heavily on how fast a good estimator for action value can be learned. By viewing this problem through the lens of representation learning, good representations of both state and action can facilitate action-value estimation. While advances in deep learning have seamlessly driven progress in learning state representations, given the specificity of the notion of agency to RL, little attention has been paid to learning action representations. We conjecture that leveraging the combinatorial structure of multi-dimensional action spaces is a key ingredient for learning good representations of action. To test this, we set forth the action hypergraph networks framework---a class of functions for learning action representations in multi-dimensional discrete action spaces with a structural inductive bias. Using this framework we realise an agent class based on a combination with deep Q-networks, which we dub hypergraph Q-networks. We show the effectiveness of our approach on a myriad of domains: illustrative prediction problems under minimal confounding effects, Atari 2600 games, and discretised physical control benchmarks.

## Mastering Atari with Discrete World Models

Authors: ['Danijar Hafner', 'Timothy P Lillicrap', 'Mohammad Norouzi', 'Jimmy Ba']

Ratings: ['4: Ok but not good enough - rejection', '5: Marginally below acceptance threshold', '8: Top 50% of accepted papers, clear accept', '9: Top 15% of accepted papers, strong accept']

Keywords: ['Atari', 'world models', 'model-based reinforcement learning', 'reinforcement learning', 'planning', 'actor critic']

Intelligent agents need to generalize from past experience to achieve goals in complex environments. World models facilitate such generalization and allow learning behaviors from imagined outcomes to increase sample-efficiency. While learning world models from image inputs has recently become feasible for some tasks, modeling Atari games accurately enough to derive successful behaviors has remained an open challenge for many years. We introduce DreamerV2, a reinforcement learning agent that learns behaviors purely from predictions in the compact latent space of a powerful world model. The world model uses discrete representations and is trained separately from the policy. DreamerV2 constitutes the first agent that achieves human-level performance on the Atari benchmark of 55 tasks by learning behaviors inside a separately trained world model. With the same computational budget and wall-clock time, Dreamer V2 reaches 200M frames and exceeds the final performance of the top single-GPU agents IQN and Rainbow.

## Efficient Reinforcement Learning in Factored MDPs with Application to Constrained RL

Authors: ['Xiaoyu Chen', 'Jiachen Hu', 'Lihong Li', 'Liwei Wang']

Ratings: ['6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['reinforcement learning', 'factored MDP', 'constrained RL', 'learning theory']

Reinforcement learning (RL) in episodic, factored Markov decision processes (FMDPs) is studied. We propose an algorithm called FMDP-BF, which leverages the factorization structure of FMDP.  The regret of FMDP-BF is shown to be exponentially smaller than that of optimal algorithms designed for non-factored MDPs, and improves on the best previous result for FMDPs~\citep{osband2014near} by a factor of $\sqrt{nH|\mathcal{S}_i|}$, where $|\mathcal{S}_i|$ is the cardinality of the factored state subspace, $H$ is the planning horizon and $n$ is the number of factored transition. To show the optimality of our bounds, we also provide a lower bound for FMDP, which indicates that our algorithm is near-optimal w.r.t. timestep $T$, horizon $H$ and factored state-action subspace cardinality. Finally, as an application, we study a new formulation of constrained RL, known as RL with knapsack constraints (RLwK), and provides the first sample-efficient algorithm based on FMDP-BF.

## Discovering Diverse Multi-Agent Strategic Behavior via Reward Randomization

Authors: ['Zhenggang Tang', 'Chao Yu', 'Boyuan Chen', 'Huazhe Xu', 'Xiaolong Wang', 'Fei Fang', 'Simon Shaolei Du', 'Yu Wang', 'Yi Wu']

Ratings: ['5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept']

Keywords: ['strategic behavior', 'multi-agent reinforcement learning', 'reward randomization', 'diverse strategies']

We propose a simple, general and effective technique, Reward Randomization for discovering diverse strategic policies in complex multi-agent games. Combining reward randomization and policy gradient, we derive a new algorithm, Reward-Randomized Policy Gradient (RPG). RPG is able to discover a set of multiple distinctive human-interpretable strategies in challenging temporal trust dilemmas, including grid-world games and a real-world game Agar.io, where multiple equilibria exist but standard multi-agent policy gradient algorithms always converge to a fixed one with a sub-optimal payoff for every player even using state-of-the-art exploration techniques. Furthermore, with the set of diverse strategies from RPG, we can (1) achieve higher payoffs by fine-tuning the best policy from the set; and (2) obtain an adaptive agent by using this set of strategies as its training opponents. 

## Learning with AMIGo: Adversarially Motivated Intrinsic Goals

Authors: ['Andres Campero', 'Roberta Raileanu', 'Heinrich Kuttler', 'Joshua B. Tenenbaum', 'Tim Rocktäschel', 'Edward Grefenstette']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['reinforcement learning', 'exploration', 'meta-learning']

A key challenge for reinforcement learning (RL) consists of learning in environments with sparse extrinsic rewards. In contrast to current RL methods, humans are able to learn new skills with little or no reward by using various forms of intrinsic motivation. We propose AMIGo, a novel agent incorporating -- as form of meta-learning -- a goal-generating teacher that proposes Adversarially Motivated Intrinsic Goals to train a goal-conditioned "student" policy in the absence of (or alongside) environment reward. Specifically, through a simple but effective "constructively adversarial" objective, the teacher learns to propose increasingly challenging -- yet achievable -- goals that allow the student to learn general skills for acting in a new environment, independent of the task to be solved. We show that our method generates a natural curriculum of self-proposed goals which ultimately allows the agent to solve challenging procedurally-generated tasks where other forms of intrinsic motivation and state-of-the-art RL methods fail.

## Hierarchical Reinforcement Learning by Discovering Intrinsic Options

Authors: ['Jesse Zhang', 'Haonan Yu', 'Wei Xu']

Ratings: ['4: Ok but not good enough - rejection', '4: Ok but not good enough - rejection', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['hierarchical reinforcement learning', 'reinforcement learning', 'options', 'unsupervised skill discovery', 'exploration']

We propose a hierarchical reinforcement learning method, HIDIO, that can learn task-agnostic options in a self-supervised manner while jointly learning to utilize them to solve sparse-reward tasks. Unlike current hierarchical RL approaches that tend to formulate goal-reaching low-level tasks or pre-define ad hoc lower-level policies, HIDIO encourages lower-level option learning that is independent of the task at hand, requiring few assumptions or little knowledge about the task structure. These options are learned through an intrinsic entropy minimization objective conditioned on the option sub-trajectories. The learned options are diverse and task-agnostic. In experiments on sparse-reward robotic manipulation and navigation tasks, HIDIO achieves higher success rates with greater sample efficiency than regular RL baselines and two state-of-the-art hierarchical RL methods. Code at: https://github.com/jesbu1/hidio.

## Off-Dynamics Reinforcement Learning: Training for Transfer with Domain Classifiers

Authors: ['Benjamin Eysenbach', 'Shreyas Chaudhari', 'Swapnil Asawa', 'Sergey Levine', 'Ruslan Salakhutdinov']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['reinforcement learning', 'transfer learning', 'domain adaptation']

We propose a simple, practical, and intuitive approach for domain adaptation in reinforcement learning. Our approach stems from the idea that the agent's experience in the source domain should look similar to its experience in the target domain. Building off of a probabilistic view of RL, we achieve this goal by compensating for the difference in dynamics by modifying the reward function. This modified reward function is simple to estimate by learning auxiliary classifiers that distinguish source-domain transitions from target-domain transitions. Intuitively, the agent is penalized for transitions that would indicate that the agent is interacting with the source domain, rather than the target domain. Formally, we prove that applying our method in the source domain is guaranteed to obtain a near-optimal policy for the target domain, provided that the source and target domains satisfy a lightweight assumption. Our approach is applicable to domains with continuous states and actions and does not require learning an explicit model of the dynamics. On discrete and continuous control tasks, we illustrate the mechanics of our approach and demonstrate its scalability to high-dimensional~tasks.

## Learning Safe Multi-agent Control with Decentralized Neural Barrier Certificates

Authors: ['Zengyi Qin', 'Kaiqing Zhang', 'Yuxiao Chen', 'Jingkai Chen', 'Chuchu Fan']

Ratings: ['4: Ok but not good enough - rejection', '6: Marginally above acceptance threshold', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['Multi-agent', 'safe', 'control barrier function', 'reinforcement learning']

We study the multi-agent safe control problem where agents should avoid collisions to static obstacles and collisions with each other while reaching their goals. Our core idea is to learn the multi-agent control policy jointly with  learning the control barrier functions as safety certificates. We propose a new joint-learning framework that can be implemented in a decentralized fashion, which can adapt to an arbitrarily large number of agents. Building upon this framework, we further improve the scalability by  incorporating neural network architectures  that are invariant to the quantity and permutation of neighboring agents. In addition, we propose a new spontaneous policy refinement method to further enforce the certificate condition during testing. We provide extensive experiments to demonstrate that our method significantly outperforms other leading multi-agent control approaches in terms of maintaining safety and completing original tasks. Our approach also shows substantial generalization capability in that the control policy can be trained with 8 agents in one scenario, while being used on other scenarios with up to 1024 agents in complex multi-agent environments and dynamics. Videos and source code can be found at https://realm.mit.edu/blog/learning-safe-multi-agent-control-decentralized-neural-barrier-certificates.

## Policy-Driven Attack: Learning to Query for Hard-label Black-box Adversarial Examples

Authors: ['Ziang Yan', 'Yiwen Guo', 'Jian Liang', 'Changshui Zhang']

Ratings: ['6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['hard-label attack', 'black-box attack', 'adversarial attack', 'reinforcement learning']

To craft black-box adversarial examples, adversaries need to query the victim model and take proper advantage of its feedback. Existing black-box attacks generally suffer from high query complexity, especially when only the top-1 decision (i.e., the hard-label prediction) of the victim model is available.  In this paper, we propose a novel hard-label black-box attack named Policy-Driven Attack, to reduce the query complexity. Our core idea is to learn promising search directions of the adversarial examples using a well-designed policy network in a novel reinforcement learning formulation, in which the queries become more sensible. Experimental results demonstrate that our method can significantly reduce the query complexity in comparison with existing state-of-the-art hard-label black-box attacks on various image classification benchmark datasets. Code and models for reproducing our results are available at https://github.com/ZiangYan/pda.pytorch

## Autoregressive Dynamics Models for Offline Policy Evaluation and Optimization

Authors: ['Michael R Zhang', 'Thomas Paine', 'Ofir Nachum', 'Cosmin Paduraru', 'George Tucker', 'ziyu wang', 'Mohammad Norouzi']

Ratings: ['6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['Off-policy policy evaluation', 'autoregressive models', 'offline reinforcement learning', 'policy optimization']

Standard dynamics models for continuous control make use of feedforward computation to predict the conditional distribution of next state and reward given current state and action using a multivariate Gaussian with a diagonal covariance structure. This modeling choice assumes that different dimensions of the next state and reward are conditionally independent given the current state and action and may be driven by the fact that fully observable physics-based simulation environments entail deterministic transition dynamics. In this paper, we challenge this conditional independence assumption and propose a family of expressive autoregressive dynamics models that generate different dimensions of the next state and reward sequentially conditioned on previous dimensions. We demonstrate that autoregressive dynamics models indeed outperform standard feedforward models in log-likelihood on heldout transitions. Furthermore, we compare different model-based and model-free off-policy evaluation (OPE) methods on RL Unplugged, a suite of offline MuJoCo datasets, and find that autoregressive dynamics models consistently outperform all baselines, achieving a new state-of-the-art. Finally, we show that autoregressive dynamics models are useful for offline policy optimization by serving as a way to enrich the replay buffer through data augmentation and improving performance using model-based planning.


## Learning Robust State Abstractions for Hidden-Parameter Block MDPs

Authors: ['Amy Zhang', 'Shagun Sodhani', 'Khimya Khetarpal', 'Joelle Pineau']

Ratings: ['6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['multi-task reinforcement learning', 'bisimulation', 'hidden-parameter mdp', 'block mdp']

Many control tasks exhibit similar dynamics that can be modeled as having common latent structure. Hidden-Parameter Markov Decision Processes (HiP-MDPs) explicitly model this structure to improve sample efficiency in multi-task settings.
However, this setting makes strong assumptions on the observability of the state that limit its application in real-world scenarios with rich observation spaces.  In this work, we leverage ideas of common structure from the HiP-MDP setting, and extend it to enable robust state abstractions inspired by Block MDPs. We  derive instantiations of this new framework for  both multi-task reinforcement learning (MTRL) and  meta-reinforcement learning (Meta-RL) settings. Further, we provide transfer and generalization bounds based on task and state similarity, along with sample complexity bounds that depend on the aggregate number of samples across tasks, rather than the number of tasks, a significant improvement over prior work. To further demonstrate efficacy of the proposed method, we empirically compare and show improvement over multi-task and meta-reinforcement learning baselines.

## Blending MPC & Value Function Approximation for Efficient Reinforcement Learning

Authors: ['Mohak Bhardwaj', 'Sanjiban Choudhury', 'Byron Boots']

Ratings: ['5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept']

Keywords: ['reinforcement learning', 'model-predictive control']

Model-Predictive Control (MPC) is a powerful tool for controlling complex, real-world systems that uses a model to make predictions about future behavior. For each state encountered, MPC solves an online optimization problem to choose a control action that will minimize future cost. This is a surprisingly effective strategy, but real-time performance requirements warrant the use of simple models. If the model is not sufficiently accurate, then the resulting controller can be biased, limiting performance. We present a framework for improving on MPC with model-free reinforcement learning (RL). The key insight is to view MPC as constructing a series of local Q-function approximations. We show that by using a parameter $\lambda$, similar to the trace decay parameter in TD($\lambda$), we can systematically trade-off learned value estimates against the local Q-function approximations. We present a theoretical analysis that shows how error from inaccurate models in MPC and value function estimation in RL can be balanced. We further propose an algorithm that changes $\lambda$ over time to reduce the dependence on MPC as our estimates of the value function improve, and test the efficacy our approach on challenging high-dimensional manipulation tasks with biased models in simulation. We demonstrate that our approach can obtain performance comparable with MPC with access to true dynamics even under severe model bias and is more sample efficient as compared to model-free RL.

## Model-based micro-data reinforcement learning: what are the crucial model properties and which model to choose?

Authors: ['Balázs Kégl', 'Gabriel Hurtado', 'Albert Thomas']

Ratings: ['5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['model-based reinforcement learning', 'generative models', 'mixture density nets', 'dynamic systems', 'heteroscedasticity']

We contribute to micro-data model-based reinforcement learning (MBRL) by rigorously comparing popular generative models using a fixed (random shooting) control agent. We find that on an environment that requires multimodal posterior predictives, mixture density nets outperform all other models by a large margin. When multimodality is not required, our surprising finding is that we do not need probabilistic posterior predictives: deterministic models are on par, in fact they consistently (although non-significantly) outperform their probabilistic counterparts. We also found that heteroscedasticity at training time, perhaps acting as a regularizer, improves predictions at longer horizons. At the methodological side, we design metrics and an experimental protocol which can be used to evaluate the various models, predicting their asymptotic performance when using them on the control problem. Using this framework, we improve the state-of-the-art sample complexity of MBRL on Acrobot by two to four folds, using an aggressive training schedule which is outside of the hyperparameter interval usually considered.

## Enforcing robust control guarantees within neural network policies

Authors: ['Priya L. Donti', 'Melrose Roderick', 'Mahyar Fazlyab', 'J Zico Kolter']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold']

Keywords: ['robust control', 'reinforcement learning', 'differentiable optimization']

When designing controllers for safety-critical systems, practitioners often face a challenging tradeoff between robustness and performance. While robust control methods provide rigorous guarantees on system stability under certain worst-case disturbances, they often yield simple controllers that perform poorly in the average (non-worst) case. In contrast, nonlinear control methods trained using deep learning have achieved state-of-the-art performance on many control tasks, but often lack robustness guarantees. In this paper, we propose a technique that combines the strengths of these two approaches: constructing a generic nonlinear control policy class, parameterized by neural networks, that nonetheless enforces the same provable robustness criteria as robust control. Specifically, our approach entails integrating custom convex-optimization-based projection layers into a neural network-based policy. We demonstrate the power of this approach on several domains, improving in average-case performance over existing robust control methods and in worst-case stability over (non-robust) deep RL methods.

## Efficient Wasserstein Natural Gradients for Reinforcement Learning

Authors: ['Ted Moskovitz', 'Michael Arbel', 'Ferenc Huszar', 'Arthur Gretton']

Ratings: ['5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '8: Top 50% of accepted papers, clear accept']

Keywords: ['reinforcement learning', 'optimization']

A novel optimization approach is proposed for application to policy gradient methods and evolution strategies for reinforcement learning (RL). The procedure uses a computationally efficient \emph{Wasserstein natural gradient} (WNG) descent that takes advantage of the geometry induced by a Wasserstein penalty to speed optimization. This method follows the recent theme in RL of including divergence penalties in the objective to establish trust regions. Experiments on challenging tasks demonstrate improvements in both computational cost and performance over advanced baselines. 


## Learning What To Do by Simulating the Past

Authors: ['David Lindner', 'Rohin Shah', 'Pieter Abbeel', 'Anca Dragan']

Ratings: ['5: Marginally below acceptance threshold', '5: Marginally below acceptance threshold', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['imitation learning', 'reward learning', 'reinforcement learning']

Since reward functions are hard to specify, recent work has focused on learning policies from human feedback. However, such approaches are impeded by the expense of acquiring such feedback. Recent work proposed that agents have access to a source of information that is effectively free: in any environment that humans have acted in, the state will already be optimized for human preferences, and thus an agent can extract information about what humans want from the state. Such learning is possible in principle, but requires simulating all possible past trajectories that could have led to the observed state. This is feasible in gridworlds, but how do we scale it to complex tasks? In this work, we show that by combining a learned feature encoder with learned inverse models, we can enable agents to simulate human actions backwards in time to infer what they must have done. The resulting algorithm is able to reproduce a specific skill in MuJoCo environments given a single state sampled from the optimal policy for that skill.

## Discovering Non-monotonic Autoregressive Orderings with Variational Inference

Authors: ['Xuanlin Li', 'Brandon Trabucco', 'Dong Huk Park', 'Michael Luo', 'Sheng Shen', 'Trevor Darrell', 'Yang Gao']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['variational inference', 'unsupervised learning', 'computer vision', 'natural language processing', 'optimization', 'reinforcement learning']

The predominant approach for language modeling is to encode a sequence of tokens from left to right, but this eliminates a source of information: the order by which the sequence was naturally generated. One strategy to recover this information is to decode both the content and ordering of tokens. Some prior work supervises content and ordering with hand-designed loss functions to encourage specific orders or bootstraps from a predefined ordering. These approaches require domain-specific insight. Other prior work searches over valid insertion operations that lead to ground truth sequences during training, which has high time complexity and cannot be efficiently parallelized. We address these limitations with an unsupervised learner that can be trained in a fully-parallelizable manner to discover high-quality autoregressive orders in a data driven way without a domain-specific prior. The learner is a neural network that performs variational inference with the autoregressive ordering as a latent variable. Since the corresponding variational lower bound is not differentiable, we develop a practical algorithm for end-to-end optimization using policy gradients. Strong empirical results with our solution on sequence modeling tasks suggest that our algorithm is capable of discovering various autoregressive orders for different sequences that are competitive with or even better than fixed orders.

## Differentiable Trust Region Layers for Deep Reinforcement Learning

Authors: ['Fabian Otto', 'Philipp Becker', 'Vien Anh Ngo', 'Hanna Carolin Maria Ziesche', 'Gerhard Neumann']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept']

Keywords: ['reinforcement learning', 'trust region', 'policy gradient', 'projection', 'Wasserstein distance', 'Kullback-Leibler divergence', 'Frobenius norm']

Trust region methods are a popular tool in reinforcement learning as they yield robust policy updates in continuous and discrete action spaces. However, enforcing such trust regions in deep reinforcement learning is difficult. Hence, many approaches, such as Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO), are based on approximations. Due to those approximations, they violate the constraints or fail to find the optimal solution within the trust region. Moreover, they are difficult to implement, often lack sufficient exploration, and have been shown to depend on seemingly unrelated implementation choices. In this work, we propose differentiable neural network layers to enforce trust regions for deep Gaussian policies via closed-form projections. Unlike existing methods, those layers formalize trust regions for each state individually and can complement existing reinforcement learning algorithms. We derive trust region projections based on the Kullback-Leibler divergence, the Wasserstein L2 distance, and the Frobenius norm for Gaussian distributions. We empirically demonstrate that those projection layers achieve similar or better results than existing methods while being almost agnostic to specific implementation choices. The code is available at https://git.io/Jthb0.


## Plan-Based Relaxed Reward Shaping for Goal-Directed Tasks

Authors: ['Ingmar Schubert', 'Ozgur S Oguz', 'Marc Toussaint']

Ratings: ['3: Clear rejection', '6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['reinforcement learning', 'reward shaping', 'plan-based reward shaping', 'robotics', 'robotic manipulation']

In high-dimensional state spaces, the usefulness of Reinforcement Learning (RL) is limited by the problem of exploration. This issue has been addressed using potential-based reward shaping (PB-RS) previously. In the present work, we introduce Final-Volume-Preserving Reward Shaping (FV-RS). FV-RS relaxes the strict optimality guarantees of PB-RS to a guarantee of preserved long-term behavior. Being less restrictive, FV-RS allows for reward shaping functions that are even better suited for improving the sample efficiency of RL algorithms. In particular, we consider settings in which the agent has access to an approximate plan. Here, we use examples of simulated robotic manipulation tasks to demonstrate that plan-based FV-RS can indeed significantly improve the sample efficiency of RL over plan-based PB-RS.

## Symmetry-Aware Actor-Critic for 3D Molecular Design

Authors: ['Gregor N. C. Simm', 'Robert Pinsler', 'Gábor Csányi', 'José Miguel Hernández-Lobato']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '8: Top 50% of accepted papers, clear accept']

Keywords: ['deep reinforcement learning', 'molecular design', 'covariant neural networks']

Automating molecular design using deep reinforcement learning (RL) has the potential to greatly accelerate the search for novel materials. Despite recent progress on leveraging graph representations to design molecules, such methods are fundamentally limited by the lack of three-dimensional (3D) information. In light of this, we propose a novel actor-critic architecture for 3D molecular design that can generate molecular structures unattainable with previous approaches. This is achieved by exploiting the symmetries of the design process through a rotationally covariant state-action representation based on a spherical harmonics series expansion. We demonstrate the benefits of our approach on several 3D molecular design tasks, where we find that building in such symmetries significantly improves generalization and the quality of generated molecules.

## Meta-Learning of Structured Task Distributions in Humans and Machines

Authors: ['Sreejan Kumar', 'Ishita Dasgupta', 'Jonathan Cohen', 'Nathaniel Daw', 'Thomas Griffiths']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['meta-learning', 'human cognition', 'reinforcement learning', 'compositionality']

In recent years, meta-learning, in which a model is trained on a family of tasks (i.e. a task distribution), has emerged as an approach to training neural networks to perform tasks that were previously assumed to require structured representations, making strides toward closing the gap between humans and machines. However, we argue that evaluating meta-learning remains a challenge, and can miss whether meta-learning actually uses the structure embedded within the tasks. These meta-learners might therefore still be significantly different from humans learners. To demonstrate this difference, we first define a new meta-reinforcement learning task in which a structured task distribution is generated using a compositional grammar. We then introduce a novel approach to constructing a "null task distribution" with the same statistical complexity as this structured task distribution but without the explicit rule-based structure used to generate the structured task. We train a standard meta-learning agent, a recurrent network trained with model-free reinforcement learning, and compare it with human performance across the two task distributions. We find a double dissociation in which humans do better in the structured task distribution whereas agents do better in the null task distribution -- despite comparable statistical complexity. This work highlights that multiple strategies can achieve reasonable meta-test performance, and that careful construction of control task distributions is a valuable way to understand which strategies meta-learners acquire, and how they might differ from humans. 

## Benchmarks for Deep Off-Policy Evaluation

Authors: ['Justin Fu', 'Mohammad Norouzi', 'Ofir Nachum', 'George Tucker', 'ziyu wang', 'Alexander Novikov', 'Mengjiao Yang', 'Michael R Zhang', 'Yutian Chen', 'Aviral Kumar', 'Cosmin Paduraru', 'Sergey Levine', 'Thomas Paine']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['reinforcement learning', 'off-policy evaluation', 'benchmarks']

Off-policy evaluation (OPE) holds the promise of being able to leverage large, offline datasets for both evaluating and selecting complex policies for decision making. The ability to learn offline is particularly important in many real-world domains, such as in healthcare, recommender systems, or robotics, where online data collection is an expensive and potentially dangerous process. Being able to accurately evaluate and select high-performing policies without requiring online interaction could yield significant benefits in safety, time, and cost for these applications. While many OPE methods have been proposed in recent years, comparing results between papers is difficult because currently there is a lack of a comprehensive and unified benchmark, and measuring algorithmic progress has been challenging due to the lack of difficult evaluation tasks. In order to address this gap, we present a collection of policies that in conjunction with existing offline datasets can be used for benchmarking off-policy evaluation. Our tasks include a range of challenging high-dimensional continuous control problems, with wide selections of datasets and policies for performing policy selection. The goal of our benchmark is to provide a standardized measure of progress that is motivated from a set of principles designed to challenge and test the limits of existing OPE methods. We perform an evaluation of state-of-the-art algorithms and provide open-source access to our data and code to foster future research in this area.    

## Planning from Pixels using Inverse Dynamics Models

Authors: ['Keiran Paster', 'Sheila A. McIlraith', 'Jimmy Ba']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold']

Keywords: ['model based reinforcement learning', 'deep reinforcement learning', 'multi-task learning', 'deep learning', 'goal-conditioned reinforcement learning']

Learning dynamics models in high-dimensional observation spaces can be challenging for model-based RL agents. We propose a novel way to learn models in a latent space by learning to predict sequences of future actions conditioned on task completion. These models track task-relevant environment dynamics over a distribution of tasks, while simultaneously serving as an effective heuristic for planning with sparse rewards. We evaluate our method on challenging visual goal completion tasks and show a substantial increase in performance compared to prior model-free approaches.

## Adaptive Procedural Task Generation for Hard-Exploration Problems

Authors: ['Kuan Fang', 'Yuke Zhu', 'Silvio Savarese', 'Fei-Fei Li']

Ratings: ['4: Ok but not good enough - rejection', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept']

Keywords: ['reinforcement learning', 'curriculum learning', 'procedural generation', 'task generation']

We introduce Adaptive Procedural Task Generation (APT-Gen), an approach to progressively generate a sequence of tasks as curricula to facilitate reinforcement learning in hard-exploration problems. At the heart of our approach, a task generator learns to create tasks from a parameterized task space via a black-box procedural generation module. To enable curriculum learning in the absence of a direct indicator of learning progress, we propose to train the task generator by balancing the agent's performance in the generated tasks and the similarity to the target tasks. Through adversarial training, the task similarity is adaptively estimated by a task discriminator defined on the agent's experiences, allowing the generated tasks to approximate target tasks of unknown parameterization or outside of the predefined task space. Our experiments on the grid world and robotic manipulation task domains show that APT-Gen achieves substantially better performance than various existing baselines by generating suitable tasks of rich variations.

## Risk-Averse Offline Reinforcement Learning

Authors: ['Núria Armengol Urpí', 'Sebastian Curi', 'Andreas Krause']

Ratings: ['5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['offline', 'reinforcement learning', 'risk-averse', 'risk sensitive', 'robust', 'safety', 'safe']

Training Reinforcement Learning (RL) agents in high-stakes applications might be too prohibitive due to the risk associated to exploration. Thus, the agent can only use data previously collected by safe policies. While previous work considers optimizing the average performance using offline data, we focus on optimizing a risk-averse criteria, namely the CVaR. In particular, we present the Offline Risk-Averse Actor-Critic (O-RAAC), a model-free RL algorithm that is able to learn risk-averse policies in a fully offline setting. We show that O-RAAC learns policies with higher CVaR than risk-neutral approaches in different robot control tasks. Furthermore, considering risk-averse criteria guarantees distributional robustness of the average performance with respect to particular distribution shifts. We demonstrate empirically that in the presence of natural distribution-shifts, O-RAAC learns policies with good average performance. 


## Monte-Carlo Planning and Learning with Language Action Value Estimates

Authors: ['Youngsoo Jang', 'Seokin Seo', 'Jongmin Lee', 'Kee-Eung Kim']

Ratings: ['4: Ok but not good enough - rejection', '6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['natural language processing', 'Monte-Carlo tree search', 'reinforcement learning', 'interactive fiction']

Interactive Fiction (IF) games provide a useful testbed for language-based reinforcement learning agents, posing significant challenges of natural language understanding, commonsense reasoning, and non-myopic planning in the combinatorial search space. Agents based on standard planning algorithms struggle to play IF games due to the massive search space of language actions. Thus, language-grounded planning is a key ability of such agents, since inferring the consequence of language action based on semantic understanding can drastically improve search. In this paper, we introduce Monte-Carlo planning with Language Action Value Estimates (MC-LAVE) that combines a Monte-Carlo tree search with language-driven exploration. MC-LAVE invests more search effort into semantically promising language actions using locally optimistic language value estimates, yielding a significant reduction in the effective search space of language actions. We then present a reinforcement learning approach via MC-LAVE, which alternates between MC-LAVE planning and supervised learning of the self-generated language actions. In the experiments, we demonstrate that our method achieves new high scores in various IF games.

## Learning to Sample with Local and Global Contexts  in Experience Replay Buffer

Authors: ['Youngmin Oh', 'Kimin Lee', 'Jinwoo Shin', 'Eunho Yang', 'Sung Ju Hwang']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept']

Keywords: ['reinforcement learning', 'experience replay buffer', 'off-policy RL']

Experience replay, which enables the agents to remember and reuse experience from the past, has played a significant role in the success of off-policy reinforcement learning (RL). To utilize the experience replay efficiently, the existing sampling methods allow selecting out more meaningful experiences by imposing priorities on them based on certain metrics (e.g. TD-error). However, they may result in sampling highly biased, redundant transitions since they compute the sampling rate for each transition independently, without consideration of its importance in relation to other transitions. In this paper, we aim to address the issue by proposing a new learning-based sampling method that can compute the relative importance of transition. To this end, we design a novel permutation-equivariant neural architecture that takes contexts from not only features of each transition (local) but also those of others (global) as inputs. We validate our framework, which we refer to as Neural Experience Replay Sampler (NERS), on multiple benchmark tasks for both continuous and discrete control tasks and show that it can significantly improve the performance of various off-policy RL methods. Further analysis confirms that the improvements of the sample efficiency indeed are due to sampling diverse and meaningful transitions by NERS that considers both local and global contexts. 

## Variational Intrinsic Control Revisited

Authors: ['Taehwan Kwon']

Ratings: ['5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold']

Keywords: ['Unsupervised reinforcement learning', 'Information theory']

In this paper, we revisit variational intrinsic control (VIC), an unsupervised reinforcement learning method for finding the largest set of intrinsic options available to an agent. In the original work by Gregor et al. (2016), two VIC algorithms were proposed: one that represents the options explicitly, and the other that does it implicitly. We show that the intrinsic reward used in the latter is subject to bias in stochastic environments, causing convergence to suboptimal solutions. To correct this behavior, we propose two methods respectively based on the transitional probability model and Gaussian Mixture Model. We substantiate our claims through rigorous mathematical derivations and experimental analyses. 

## Return-Based Contrastive Representation Learning for Reinforcement  Learning

Authors: ['Guoqing Liu', 'Chuheng Zhang', 'Li Zhao', 'Tao Qin', 'Jinhua Zhu', 'Li Jian', 'Nenghai Yu', 'Tie-Yan Liu']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['reinforcement learning', 'auxiliary task', 'representation learning', 'contrastive learning']

Recently, various auxiliary tasks have been proposed to accelerate representation learning and improve sample efficiency in deep reinforcement learning (RL). However, existing auxiliary tasks do not take the characteristics of RL problems into consideration and are unsupervised. By leveraging returns, the most important feedback signals in RL, we propose a novel auxiliary task that forces the learnt representations to discriminate state-action pairs with different returns. Our auxiliary loss is theoretically justified to learn representations that capture the structure of a new form of state-action abstraction, under which state-action pairs with similar return distributions are aggregated together. Empirically, our algorithm outperforms strong baselines on complex tasks in Atari games and DeepMind Control suite, and achieves even better performance when combined with existing auxiliary tasks.

## Scalable Bayesian Inverse Reinforcement Learning

Authors: ['Alex James Chan', 'Mihaela van der Schaar']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['Bayesian', 'Inverse reinforcement learning', 'Imitation Learning']

Bayesian inference over the reward presents an ideal solution to the ill-posed nature of the inverse reinforcement learning problem. Unfortunately current methods generally do not scale well beyond the small tabular setting due to the need for an inner-loop MDP solver, and even non-Bayesian methods that do themselves scale often require extensive interaction with the environment to perform well, being inappropriate for high stakes or costly applications such as healthcare. In this paper we introduce our method, Approximate Variational Reward Imitation Learning (AVRIL), that addresses both of these issues by jointly learning an approximate posterior distribution over the reward that scales to arbitrarily complicated state spaces alongside an appropriate policy in a completely offline manner through a variational approach to said latent reward. Applying our method to real medical data alongside classic control simulations, we demonstrate Bayesian reward inference in environments beyond the scope of current methods, as well as task performance competitive with focused offline imitation learning algorithms.

## Simple Augmentation Goes a Long Way: ADRL for DNN Quantization

Authors: ['Lin Ning', 'Guoyang Chen', 'Weifeng Zhang', 'Xipeng Shen']

Ratings: ['6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept']

Keywords: ['Reinforcement Learning', 'Quantization', 'mixed precision', 'augmented deep reinforcement learning', 'DNN']

Mixed precision quantization improves DNN performance by assigning different layers with different bit-width values. Searching for the optimal bit-width for each layer, however, remains a challenge. Deep Reinforcement Learning (DRL) shows some recent promise. It however suffers instability due to function approximation errors, causing large variances in the early training stages, slow convergence, and suboptimal policies in the mixed-precision quantization problem. This paper proposes augmented DRL (ADRL) as a way to alleviate these issues. This new strategy augments the neural networks in DRL with a complementary scheme to boost the performance of learning. The paper examines the effectiveness of ADRL both analytically and empirically, showing that it can produce more accurate quantized models than the state of the art DRL-based quantization while improving the learning speed by 4.5-64 times. 

## C-Learning: Horizon-Aware Cumulative Accessibility Estimation

Authors: ['Panteha Naderian', 'Gabriel Loaiza-Ganem', 'Harry J. Braviner', 'Anthony L. Caterini', 'Jesse C. Cresswell', 'Tong Li', 'Animesh Garg']

Ratings: ['5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold']

Keywords: ['reinforcement learning', 'goal reaching', 'Q-learning']

Multi-goal reaching is an important problem in reinforcement learning needed to achieve algorithmic generalization. Despite recent advances in this field, current algorithms suffer from three major challenges: high sample complexity, learning only a single way of reaching the goals,  and difficulties in solving complex motion planning tasks. In order to address these limitations, we introduce the concept of cumulative accessibility functions, which measure the reachability of a goal from a given state within a specified horizon. We show that these functions obey a recurrence relation, which enables learning from offline interactions. We also prove that optimal cumulative accessibility functions are monotonic in the planning horizon. Additionally, our method can trade off speed and reliability in goal-reaching by suggesting multiple paths to a single goal depending on the provided horizon. We evaluate our approach on a set of multi-goal discrete and continuous control tasks. We show that our method outperforms state-of-the-art goal-reaching algorithms in success rate, sample complexity, and path optimality. Our code is available at https://github.com/layer6ai-labs/CAE, and additional visualizations can be found at https://sites.google.com/view/learning-cae/.

## Grounding Language to Autonomously-Acquired Skills via Goal Generation

Authors: ['Ahmed Akakzia', 'Cédric Colas', 'Pierre-Yves Oudeyer', 'Mohamed CHETOUANI', 'Olivier Sigaud']

Ratings: ['4: Ok but not good enough - rejection', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept']

Keywords: ['Deep reinforcement learning', 'intrinsic motivations', 'symbolic representations', 'autonomous learning']

We are interested in the autonomous acquisition of repertoires of skills. Language-conditioned reinforcement learning (LC-RL) approaches are great tools in this quest, as they allow to express abstract goals as sets of constraints on the states. However, most LC-RL agents are not autonomous and cannot learn without external instructions and feedback. Besides, their direct language condition cannot account for the goal-directed behavior of pre-verbal infants and strongly limits the expression of behavioral diversity for a given language input. To resolve these issues, we propose a new conceptual approach to language-conditioned RL: the Language-Goal-Behavior architecture (LGB). LGB decouples skill learning and language grounding via an intermediate semantic representation of the world. To showcase the properties of LGB, we present a specific implementation called DECSTR. DECSTR is an intrinsically motivated learning agent endowed with an innate semantic representation describing spatial relations between physical objects. In a first stage G -> B, it freely explores its environment and targets self-generated semantic configurations. In a second stage (L -> G), it trains a language-conditioned  goal generator to generate semantic goals that match the constraints expressed in language-based inputs. We showcase the additional properties of LGB w.r.t. both an end-to-end LC-RL approach and a similar approach leveraging non-semantic, continuous intermediate representations. Intermediate semantic representations help satisfy language commands in a diversity of ways, enable strategy switching after a failure and facilitate language grounding.

## Extracting Strong Policies for Robotics Tasks from Zero-Order Trajectory Optimizers

Authors: ['Cristina Pinneri', 'Shambhuraj Sawant', 'Sebastian Blaes', 'Georg Martius']

Ratings: ['5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold']

Keywords: ['reinforcement learning', 'zero-order optimization', 'policy learning', 'model-based learning', 'robotics', 'model predictive control']

Solving high-dimensional, continuous robotic tasks is a challenging optimization problem. Model-based methods that rely on zero-order optimizers like the cross-entropy method (CEM) have so far shown strong performance and are considered state-of-the-art in the model-based reinforcement learning community. However, this success comes at the cost of high computational complexity, being therefore not suitable for real-time control. In this paper, we propose a technique to jointly optimize the trajectory and distill a policy, which is essential for fast execution in real robotic systems. Our method builds upon standard approaches, like guidance cost and dataset aggregation, and introduces a novel adaptive factor which prevents the optimizer from collapsing to the learner's behavior at the beginning of the training. The extracted policies reach unprecedented performance on challenging tasks as making a humanoid stand up and opening a door without reward shaping

## Temporally-Extended ε-Greedy Exploration

Authors: ['Will Dabney', 'Georg Ostrovski', 'Andre Barreto']

Ratings: ['5: Marginally below acceptance threshold', '5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '8: Top 50% of accepted papers, clear accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['reinforcement learning', 'exploration']

Recent work on exploration in reinforcement learning (RL) has led to a series of increasingly complex solutions to the problem. This increase in complexity often comes at the expense of generality. Recent empirical studies suggest that, when applied to a broader set of domains, some sophisticated exploration methods are outperformed by simpler counterparts, such as ε-greedy. In this paper we propose an exploration algorithm that retains the simplicity of ε-greedy while reducing dithering. We build on a simple hypothesis: the main limitation of ε-greedy exploration is its lack of temporal persistence, which limits its ability to escape local optima. We propose a temporally extended form of ε-greedy that simply repeats the sampled action for a random duration. It turns out that, for many duration distributions, this suffices to improve exploration on a large set of domains. Interestingly, a class of distributions inspired by ecological models of animal foraging behaviour yields particularly strong performance.

## Rapid Task-Solving in Novel Environments

Authors: ['Samuel Ritter', 'Ryan Faulkner', 'Laurent Sartran', 'Adam Santoro', 'Matthew Botvinick', 'David Raposo']

Ratings: ['4: Ok but not good enough - rejection', '7: Good paper, accept', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['deep reinforcement learning', 'meta learning', 'deep learning', 'exploration', 'planning']

We propose the challenge of rapid task-solving in novel environments (RTS), wherein an agent must solve a series of tasks as rapidly as possible in an unfamiliar environment. An effective RTS agent must balance between exploring the unfamiliar environment and solving its current task, all while building a model of the new environment over which it can plan when faced with later tasks. While modern deep RL agents exhibit some of these abilities in isolation, none are suitable for the full RTS challenge. To enable progress toward RTS, we introduce two challenge domains: (1) a minimal RTS challenge called the Memory&Planning Game and (2) One-Shot StreetLearn Navigation, which introduces scale and complexity from real-world data. We demonstrate that state-of-the-art deep RL agents fail at RTS in both domains, and that this failure is due to an inability to plan over gathered knowledge. We develop Episodic Planning Networks (EPNs) and show that deep-RL agents with EPNs excel at RTS, outperforming the nearest baseline by factors of 2-3 and learning to navigate held-out StreetLearn maps within a single episode. We show that EPNs learn to execute a value iteration-like planning algorithm and that they generalize to situations beyond their training experience.

## X2T: Training an X-to-Text Typing Interface with Online Learning from User Feedback

Authors: ['Jensen Gao', 'Siddharth Reddy', 'Glen Berseth', 'Nicholas Hardy', 'Nikhilesh Natraj', 'Karunesh Ganguly', 'Anca Dragan', 'Sergey Levine']

Ratings: ['4: Ok but not good enough - rejection', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['reinforcement learning', 'human-computer interaction']

We aim to help users communicate their intent to machines using flexible, adaptive interfaces that translate arbitrary user input into desired actions. In this work, we focus on assistive typing applications in which a user cannot operate a keyboard, but can instead supply other inputs, such as webcam images that capture eye gaze or neural activity measured by a brain implant. Standard methods train a model on a fixed dataset of user inputs, then deploy a static interface that does not learn from its mistakes; in part, because extracting an error signal from user behavior can be challenging. We investigate a simple idea that would enable such interfaces to improve over time, with minimal additional effort from the user: online learning from user feedback on the accuracy of the interface's actions. In the typing domain, we leverage backspaces as feedback that the interface did not perform the desired action. We propose an algorithm called x-to-text (X2T) that trains a predictive model of this feedback signal, and uses this model to fine-tune any existing, default interface for translating user input into actions that select words or characters. We evaluate X2T through a small-scale online user study with 12 participants who type sentences by gazing at their desired words, a large-scale observational study on handwriting samples from 60 users, and a pilot study with one participant using an electrocorticography-based brain-computer interface. The results show that X2T learns to outperform a non-adaptive default interface, stimulates user co-adaptation to the interface, personalizes the interface to individual users, and can leverage offline data collected from the default interface to improve its initial performance and accelerate online learning.

## QPLEX: Duplex Dueling Multi-Agent Q-Learning

Authors: ['Jianhao Wang', 'Zhizhou Ren', 'Terry Liu', 'Yang Yu', 'Chongjie Zhang']

Ratings: ['4: Ok but not good enough - rejection', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept']

Keywords: ['Multi-agent reinforcement learning', 'Value factorization', 'Dueling structure']

We explore value-based multi-agent reinforcement learning (MARL) in the popular paradigm of centralized training with decentralized execution (CTDE). CTDE has an important concept, Individual-Global-Max (IGM) principle, which requires the consistency between joint and local action selections to support efficient local decision-making. However, in order to achieve scalability, existing MARL methods either limit representation expressiveness of their value function classes or relax the IGM consistency, which may suffer from instability risk or may not perform well in complex domains. This paper presents a novel MARL approach, called duPLEX dueling multi-agent Q-learning (QPLEX), which takes a duplex dueling network architecture to factorize the joint value function. This duplex dueling structure encodes the IGM principle into the neural network architecture and thus enables efficient value function learning. Theoretical analysis shows that QPLEX achieves a complete IGM function class. Empirical experiments on StarCraft II micromanagement tasks demonstrate that QPLEX significantly outperforms state-of-the-art baselines in both online and offline data collection settings, and also reveal that QPLEX achieves high sample efficiency and can benefit from offline datasets without additional online exploration.

## Model-Based Offline Planning

Authors: ['Arthur Argenson', 'Gabriel Dulac-Arnold']

Ratings: ['5: Marginally below acceptance threshold', '5: Marginally below acceptance threshold', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['off-line reinforcement learning', 'model-based reinforcement learning', 'model-based control', 'reinforcement learning', 'model predictive control', 'robotics']

Offline learning is a key part of making reinforcement learning (RL) useable in real systems. Offline RL looks at scenarios where there is data from a system's operation, but no direct access to the system when learning a policy. Recent work on training RL policies from offline data has shown results both with model-free policies learned directly from the data, or with planning on top of learnt models of the data. Model-free policies tend to be more performant, but are more opaque, harder to command externally, and less easy to integrate into larger systems. We propose an offline learner that generates a model that can be used to control the system directly through planning. This allows us to have easily controllable policies directly from data, without ever interacting with the system. We show the performance of our algorithm, Model-Based Offline Planning (MBOP) on a series of robotics-inspired tasks, and demonstrate its ability leverage planning to respect environmental constraints. We are able to find near-optimal polices for certain simulated systems from as little as 50 seconds of real-time system interaction, and create zero-shot goal-conditioned policies on a series of environments.

## Batch Reinforcement Learning Through Continuation Method

Authors: ['Yijie Guo', 'Shengyu Feng', 'Nicolas Le Roux', 'Ed Chi', 'Honglak Lee', 'Minmin Chen']

Ratings: ['4: Ok but not good enough - rejection', '6: Marginally above acceptance threshold', '7: Good paper, accept', '9: Top 15% of accepted papers, strong accept']

Keywords: ['batch reinforcement learning', 'continuation method', 'relaxed regularization']

Many real-world applications of reinforcement learning (RL) require the agent to learn from a fixed set of trajectories, without collecting new interactions.  Policy optimization under this setting is extremely challenging as: 1) the geometry of the objective function is hard to optimize efficiently; 2) the shift of data distributions causes high noise in the value estimation. In this work, we propose a simple yet effective policy iteration approach to batch RL using global optimization techniques known as continuation.  By constraining the difference between the learned policy and the behavior policy that generates the fixed trajectories, and continuously relaxing the constraint, our method 1) helps the agent escape local optima; 2) reduces the error in policy evaluation in the optimization procedure.   We present results on a variety of control tasks, game environments, and a recommendation task to empirically demonstrate the efficacy of our proposed method.

## Large Batch Simulation for Deep Reinforcement Learning

Authors: ['Brennan Shacklett', 'Erik Wijmans', 'Aleksei Petrenko', 'Manolis Savva', 'Dhruv Batra', 'Vladlen Koltun', 'Kayvon Fatahalian']

Ratings: ['4: Ok but not good enough - rejection', '5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept', '7: Good paper, accept']

Keywords: ['reinforcement learning', 'simulation']

We accelerate deep reinforcement learning-based training in visually complex 3D environments by two orders of magnitude over prior work, realizing end-to-end training speeds of over 19,000 frames of experience per second on a single GPU and up to 72,000 frames per second on a single eight-GPU machine. The key idea of our approach is to design a 3D renderer and embodied navigation simulator around the principle of “batch simulation”: accepting and executing large batches of requests simultaneously.  Beyond exposing large amounts of work at once, batch simulation allows implementations to amortize in-memory storage of scene assets, rendering work, data loading, and synchronization costs across many simulation requests, dramatically improving the number of simulated agents per GPU and overall simulation throughput.  To balance DNN inference and training costs with faster simulation, we also build a computationally efficient policy DNN that maintains high task performance, and modify training algorithms to maintain sample efficiency when training with large mini-batches. By combining batch simulation and DNN performance optimizations, we demonstrate that PointGoal navigation agents can be trained in complex 3D environments on a single GPU in 1.5 days to 97% of the accuracy of agents trained on a prior state-of-the-art system using a 64-GPU cluster over three days.  We provide open-source reference implementations of our batch 3D renderer and simulator to facilitate incorporation of these ideas into RL systems.

## Reset-Free Lifelong Learning with Skill-Space Planning

Authors: ['Kevin Lu', 'Aditya Grover', 'Pieter Abbeel', 'Igor Mordatch']

Ratings: ['5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept']

Keywords: ['reset-free', 'lifelong', 'reinforcement learning']

The objective of \textit{lifelong} reinforcement learning (RL) is to optimize agents which can continuously adapt and interact in changing environments. However, current RL approaches fail drastically when environments are non-stationary and interactions are non-episodic. We propose \textit{Lifelong Skill Planning} (LiSP), an algorithmic framework for lifelong RL based on planning in an abstract space of higher-order skills. We learn the skills in an unsupervised manner using intrinsic rewards and plan over the learned skills using a learned dynamics model. Moreover, our framework permits skill discovery even from offline data, thereby reducing the need for excessive real-world interactions. We demonstrate empirically that LiSP successfully enables long-horizon planning and learns agents that can avoid catastrophic failures even in challenging non-stationary and non-episodic environments derived from gridworld and MuJoCo benchmarks.

## FOCAL: Efficient Fully-Offline Meta-Reinforcement Learning via Distance Metric Learning and Behavior Regularization

Authors: ['Lanqing Li', 'Rui Yang', 'Dijun Luo']

Ratings: ['5: Marginally below acceptance threshold', '5: Marginally below acceptance threshold', '7: Good paper, accept']

Keywords: ['offline/batch reinforcement learning', 'meta-reinforcement learning', 'multi-task reinforcement learning', 'distance metric learning', 'contrastive learning']

We study the offline meta-reinforcement learning (OMRL) problem, a paradigm which enables reinforcement learning (RL) algorithms to quickly adapt to unseen tasks without any interactions with the environments, making RL truly practical in many real-world applications. This problem is still not fully understood, for which two major challenges need to be addressed. First, offline RL usually suffers from bootstrapping errors of out-of-distribution state-actions which leads to divergence of value functions. Second, meta-RL requires efficient and robust task inference learned jointly with control policy. In this work, we enforce behavior regularization on learned policy as a general approach to offline RL, combined with a deterministic context encoder for efficient task inference. We propose a novel negative-power distance metric on bounded context embedding space, whose gradients propagation is detached from the Bellman backup. We provide analysis and insight showing that some simple design choices can yield substantial improvements over recent approaches involving meta-RL and distance metric learning. To the best of our knowledge, our method is the first model-free and end-to-end OMRL algorithm, which is computationally efficient and demonstrated to outperform prior algorithms on several meta-RL benchmarks.

## Communication in Multi-Agent Reinforcement Learning: Intention Sharing

Authors: ['Woojun Kim', 'Jongeui Park', 'Youngchul Sung']

Ratings: ['4: Ok but not good enough - rejection', '5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold']

Keywords: ['Multi-agent reinforcement learning', 'communication', 'intention', 'attention']

Communication is one of the core components for learning coordinated behavior in multi-agent systems.
In this paper, we propose a new communication scheme named  Intention Sharing (IS) for multi-agent reinforcement learning in order to enhance the coordination among agents. In the proposed IS scheme, each agent generates an imagined trajectory by modeling the environment dynamics and other agents' actions. The imagined trajectory is the simulated future trajectory of each agent based on the learned model of the environment dynamics and other agents and represents each agent's future action plan. Each agent compresses this imagined trajectory capturing its future action plan to generate its intention message for communication by applying an attention mechanism to learn the relative importance of the components in the imagined trajectory based on the received message from other agents. Numeral results show that the proposed IS scheme outperforms other communication schemes in multi-agent reinforcement learning.

## Solving Compositional Reinforcement Learning Problems via Task Reduction

Authors: ['Yunfei Li', 'Yilin Wu', 'Huazhe Xu', 'Xiaolong Wang', 'Yi Wu']

Ratings: ['3: Clear rejection', '5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '7: Good paper, accept']

Keywords: ['compositional task', 'sparse reward', 'reinforcement learning', 'task reduction', 'imitation learning']

We propose a novel learning paradigm, Self-Imitation via Reduction (SIR), for solving compositional reinforcement learning problems. SIR is based on two core ideas: task reduction and self-imitation. Task reduction tackles a hard-to-solve task by actively reducing it to an easier task whose solution is known by the RL agent. Once the original hard task is successfully solved by task reduction, the agent naturally obtains a self-generated solution trajectory to imitate. By continuously collecting and imitating such demonstrations, the agent is able to progressively expand the solved subspace in the entire task space. Experiment results show that SIR can significantly accelerate and improve learning on a variety of challenging sparse-reward continuous-control problems with compositional structures. Code and videos are available at https://sites.google.com/view/sir-compositional.

## Acting in Delayed Environments with Non-Stationary Markov Policies

Authors: ['Esther Derman', 'Gal Dalal', 'Shie Mannor']

Ratings: ['5: Marginally below acceptance threshold', '6: Marginally above acceptance threshold', '6: Marginally above acceptance threshold', '8: Top 50% of accepted papers, clear accept']

Keywords: ['reinforcement learning', 'delay']

The standard Markov Decision Process (MDP) formulation hinges on the assumption that an action is executed immediately after it was chosen. However, assuming it is often unrealistic and can lead to catastrophic failures in applications such as robotic manipulation, cloud computing, and finance. We introduce a framework for learning and planning in MDPs where the decision-maker commits actions that are executed with a delay of $m$ steps. The brute-force state augmentation baseline where the state is concatenated to the last $m$ committed actions suffers from an exponential complexity in $m$, as we show for policy iteration. We then prove that with execution delay, deterministic Markov policies in the original state-space are sufficient for attaining maximal reward, but need to be non-stationary. As for stationary Markov policies, we show they are sub-optimal in general. Consequently, we devise a non-stationary Q-learning style model-based algorithm that solves delayed execution tasks without resorting to state-augmentation. Experiments on tabular, physical, and Atari domains reveal that it converges quickly to high performance even for substantial delays, while standard approaches that either ignore the delay or rely on state-augmentation struggle or fail due to divergence. The code is available at \url{https://github.com/galdl/rl_delay_basic.git}.

