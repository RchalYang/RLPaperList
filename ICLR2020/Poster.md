# Poster

## Reinforcement Learning Based Graph-to-Sequence Model for Natural Question Generation

Authors: ['Yu Chen', 'Lingfei Wu', 'Mohammed J. Zaki']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['deep learning', 'reinforcement learning', 'graph neural networks', 'natural language processing', 'question generation']

Natural question generation (QG) aims to generate questions from a passage and an answer. Previous works on QG either (i) ignore the rich structure information hidden in text, (ii) solely rely on cross-entropy loss that leads to issues like exposure bias and inconsistency between train/test measurement, or (iii) fail to fully exploit the answer information. To address these limitations, in this paper, we propose a reinforcement learning (RL) based graph-to-sequence (Graph2Seq) model for QG. Our model consists of a Graph2Seq generator with a novel Bidirectional Gated Graph Neural Network based encoder to embed the passage, and a hybrid evaluator with a mixed objective combining both cross-entropy and RL losses to ensure the generation of syntactically and semantically valid text. We also introduce an effective Deep Alignment Network for incorporating the answer information into the passage at both the word and contextual levels. Our model is end-to-end trainable and achieves new state-of-the-art scores, outperforming existing methods by a significant margin on the standard SQuAD benchmark.

## Maxmin Q-learning: Controlling the Estimation Bias of Q-learning

Authors: ['Qingfeng Lan', 'Yangchen Pan', 'Alona Fyshe', 'Martha White']

Ratings: ['3: Weak Reject', '6: Weak Accept', '8: Accept']

Keywords: ['reinforcement learning', 'bias and variance reduction']

Q-learning suffers from overestimation bias, because it approximates the maximum action value using the maximum estimated action value. Algorithms have been proposed to reduce overestimation bias, but we lack an understanding of how bias interacts with performance, and the extent to which existing algorithms mitigate bias. In this paper, we 1) highlight that the effect of overestimation bias on learning efficiency is environment-dependent; 2) propose a generalization of Q-learning, called \emph{Maxmin Q-learning}, which provides a parameter to flexibly control bias; 3) show theoretically that there exists a parameter choice for Maxmin Q-learning that leads to unbiased estimation with a lower approximation variance than Q-learning; and 4) prove the convergence of our algorithm in the tabular case, as well as convergence of several previous Q-learning variants, using a novel Generalized Q-learning framework. We empirically verify that our algorithm better controls estimation bias in toy environments, and that it achieves superior performance on several benchmark problems. 

## A Learning-based Iterative Method for Solving Vehicle Routing Problems

Authors: ['Hao Lu', 'Xingwen Zhang', 'Shuang Yang']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['vehicle routing', 'reinforcement learning', 'optimization', 'heuristics']

This paper is concerned with solving combinatorial optimization problems, in particular, the capacitated vehicle routing problems (CVRP). Classical Operations Research (OR) algorithms such as LKH3 (Helsgaun, 2017) are extremely inefficient (e.g., 13 hours on CVRP of only size 100) and difficult to scale to larger-size problems. Machine learning based approaches have recently shown to be promising, partly because of their efficiency (once trained, they can perform solving within minutes or even seconds). However, there is still a considerable gap between the quality of a machine learned solution and what OR methods can offer (e.g., on CVRP-100, the best result of learned solutions is between 16.10-16.80, significantly worse than LKH3's 15.65). In this paper, we present ’‘learn to Improve’‘ (L2I), the first learning based approach for CVRP that is efficient in solving speed and at the same time outperforms OR methods. Starting with a random initial solution, L2I learns to iteratively refine the solution with an improvement operator, selected by a reinforcement learning based controller. The improvement operator is selected from a pool of powerful operators that are customized for routing problems. By combining the strengths of the two worlds, our approach achieves the new state-of-the-art results on CVRP, e.g., an average cost of 15.57 on CVRP-100.

## SVQN: Sequential Variational Soft Q-Learning Networks

Authors: ['Shiyu Huang', 'Hang Su', 'Jun Zhu', 'Ting Chen']

Ratings: ['3: Weak Reject', '8: Accept']

Keywords: ['reinforcement learning', 'POMDP', 'variational inference', 'generative model']

Partially Observable Markov Decision Processes (POMDPs) are popular and flexible models for real-world decision-making applications that demand the information from past observations to make optimal decisions. Standard reinforcement learning algorithms for solving Markov Decision Processes (MDP) tasks are not applicable, as they cannot infer the unobserved states. In this paper, we propose a novel algorithm for POMDPs, named sequential variational soft Q-learning networks (SVQNs), which formalizes the inference of hidden states and maximum entropy reinforcement learning (MERL) under a unified graphical model and optimizes the two modules jointly. We further design a deep recurrent neural network to reduce the computational complexity of the algorithm. Experimental results show that SVQNs can utilize past information to help decision making for efficient inference, and outperforms other baselines on several challenging tasks. Our ablation study shows that SVQNs have the generalization ability over time and are robust to the disturbance of the observation.

## Ranking Policy Gradient

Authors: ['Kaixiang Lin', 'Jiayu Zhou']

Ratings: ['3: Weak Reject', '6: Weak Accept', '6: Weak Accept']

Keywords: ['Sample-efficient reinforcement learning', 'off-policy learning.']

Sample inefficiency is a long-lasting problem in reinforcement learning (RL). The state-of-the-art estimates the optimal action values while it usually involves an extensive search over the state-action space and unstable optimization. Towards the sample-efficient RL, we propose ranking policy gradient (RPG), a policy gradient method that learns the optimal rank of a set of discrete actions. To accelerate the learning of policy gradient methods, we establish the equivalence between maximizing the lower bound of return and imitating a near-optimal policy without accessing any oracles. These results lead to a general off-policy learning framework, which preserves the optimality, reduces variance, and improves the sample-efficiency. We conduct extensive experiments showing that when consolidating with the off-policy learning framework, RPG substantially reduces the sample complexity, comparing to the state-of-the-art.

## Hierarchical Foresight: Self-Supervised Learning of Long-Horizon Tasks via Visual Subgoal Generation

Authors: ['Suraj Nair', 'Chelsea Finn']

Ratings: ['6: Weak Accept', '6: Weak Accept']

Keywords: ['video prediction', 'reinforcement learning', 'planning']

Video prediction models combined with planning algorithms have shown promise in enabling robots to learn to perform many vision-based tasks through only self-supervision, reaching novel goals in cluttered scenes with unseen objects. However, due to the compounding uncertainty in long horizon video prediction and poor scalability of sampling-based planning optimizers, one significant limitation of these approaches is the ability to plan over long horizons to reach distant goals. To that end, we propose a framework for subgoal generation and planning, hierarchical visual foresight (HVF), which generates subgoal images conditioned on a goal image, and uses them for planning. The subgoal images are directly optimized to decompose the task into easy to plan segments, and as a result, we observe that the method naturally identifies semantically meaningful states as subgoals. Across three out of four simulated vision-based manipulation tasks, we find that our method achieves more than 20% absolute performance improvement over planning without subgoals and model-free RL approaches. Further, our experiments illustrate that our approach extends to real, cluttered visual scenes.

## Multi-agent Reinforcement Learning for Networked System Control

Authors: ['Tianshu Chu', 'Sandeep Chinchali', 'Sachin Katti']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['deep reinforcement learning', 'multi-agent reinforcement learning', 'decision and control']

This paper considers multi-agent reinforcement learning (MARL) in networked system control. Specifically, each agent learns a decentralized control policy based on local observations and messages from connected neighbors. We formulate such a networked MARL (NMARL) problem as a spatiotemporal Markov decision process and introduce a spatial discount factor to stabilize the training of each local agent. Further, we propose a new differentiable communication protocol, called NeurComm, to reduce information loss and non-stationarity in NMARL. Based on experiments in realistic NMARL scenarios of adaptive traffic signal control and cooperative adaptive cruise control, an appropriate spatial discount factor effectively enhances the learning curves of non-communicative MARL algorithms, while NeurComm outperforms existing communication protocols in both learning efficiency and control performance.

## Logic and the 2-Simplicial Transformer

Authors: ['James Clift', 'Dmitry Doryn', 'Daniel Murfet', 'James Wallbridge']

Ratings: ['3: Weak Reject', '3: Weak Reject', '8: Accept']

Keywords: ['transformer', 'logic', 'reinforcement learning', 'reasoning']

We introduce the 2-simplicial Transformer, an extension of the Transformer which includes a form of higher-dimensional attention generalising the dot-product attention, and uses this attention to update entity representations with tensor products of value vectors. We show that this architecture is a useful inductive bias for logical reasoning in the context of deep reinforcement learning.


## Watch, Try, Learn: Meta-Learning from Demonstrations and Rewards

Authors: ['Allan Zhou', 'Eric Jang', 'Daniel Kappler', 'Alex Herzog', 'Mohi Khansari', 'Paul Wohlhart', 'Yunfei Bai', 'Mrinal Kalakrishnan', 'Sergey Levine', 'Chelsea Finn']

Ratings: ['3: Weak Reject', '6: Weak Accept', '8: Accept']

Keywords: ['meta-learning', 'reinforcement learning', 'imitation learning']

Imitation learning allows agents to learn complex behaviors from demonstrations. However, learning a complex vision-based task may require an impractical number of demonstrations. Meta-imitation learning is a promising approach towards enabling agents to learn a new task from one or a few demonstrations by leveraging experience from learning similar tasks. In the presence of task ambiguity or unobserved dynamics, demonstrations alone may not provide enough information; an agent must also try the task to successfully infer a policy. In this work, we propose a method that can learn to learn from both demonstrations and trial-and-error experience with sparse reward feedback. In comparison to meta-imitation, this approach enables the agent to effectively and efficiently improve itself autonomously beyond the demonstration data. In comparison to meta-reinforcement learning, we can scale to substantially broader distributions of tasks, as the demonstration reduces the burden of exploration. Our experiments show that our method significantly outperforms prior approaches on a set of challenging, vision-based control tasks.

## V-MPO: On-Policy Maximum a Posteriori Policy Optimization for Discrete and Continuous Control

Authors: ['H. Francis Song', 'Abbas Abdolmaleki', 'Jost Tobias Springenberg', 'Aidan Clark', 'Hubert Soyer', 'Jack W. Rae', 'Seb Noury', 'Arun Ahuja', 'Siqi Liu', 'Dhruva Tirumala', 'Nicolas Heess', 'Dan Belov', 'Martin Riedmiller', 'Matthew M. Botvinick']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['reinforcement learning', 'policy iteration', 'multi-task learning', 'continuous control']

Some of the most successful applications of deep reinforcement learning to challenging domains in discrete and continuous control have used policy gradient methods in the on-policy setting. However, policy gradients can suffer from large variance that may limit performance, and in practice require carefully tuned entropy regularization to prevent policy collapse. As an alternative to policy gradient algorithms, we introduce V-MPO, an on-policy adaptation of Maximum a Posteriori Policy Optimization (MPO) that performs policy iteration based on a learned state-value function. We show that V-MPO surpasses previously reported scores for both the Atari-57 and DMLab-30 benchmark suites in the multi-task setting, and does so reliably without importance weighting, entropy regularization, or population-based tuning of hyperparameters. On individual DMLab and Atari levels, the proposed algorithm can achieve scores that are substantially higher than has previously been reported. V-MPO is also applicable to problems with high-dimensional, continuous action spaces, which we demonstrate in the context of learning to control simulated humanoids with 22 degrees of freedom from full state observations and 56 degrees of freedom from pixel observations, as well as example OpenAI Gym tasks where V-MPO achieves substantially higher asymptotic scores than previously reported.

## RTFM: Generalising to New Environment Dynamics via Reading

Authors: ['Victor Zhong', 'Tim Rocktäschel', 'Edward Grefenstette']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['reinforcement learning', 'policy learning', 'reading comprehension', 'generalisation']

Obtaining policies that can generalise to new environments in reinforcement learning is challenging. In this work, we demonstrate that language understanding via a reading policy learner is a promising vehicle for generalisation to new environments. We propose a grounded policy learning problem, Read to Fight Monsters (RTFM), in which the agent must jointly reason over a language goal, relevant dynamics described in a document, and environment observations. We procedurally generate environment dynamics and corresponding language descriptions of the dynamics, such that agents must read to understand new environment dynamics instead of memorising any particular information. In addition, we propose txt2π, a model that captures three-way interactions between the goal, document, and observations. On RTFM, txt2π generalises to new environments with dynamics not seen during training via reading. Furthermore, our model outperforms baselines such as FiLM and language-conditioned CNNs on RTFM. Through curriculum learning, txt2π produces policies that excel on complex RTFM tasks requiring several reasoning and coreference steps.

## Exploring Model-based Planning with Policy Networks

Authors: ['Tingwu Wang', 'Jimmy Ba']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['reinforcement learning', 'model-based reinforcement learning', 'planning']

Model-based reinforcement learning (MBRL) with model-predictive control or
online planning has shown great potential for locomotion control tasks in both
sample efficiency and asymptotic performance. Despite the successes, the existing
planning methods search from candidate sequences randomly generated in the
action space, which is inefficient in complex high-dimensional environments. In
this paper, we propose a novel MBRL algorithm, model-based policy planning
(POPLIN), that combines policy networks with online planning. More specifically,
we formulate action planning at each time-step as an optimization problem using
neural networks. We experiment with both optimization w.r.t. the action sequences
initialized from the policy network, and also online optimization directly w.r.t. the
parameters of the policy network. We show that POPLIN obtains state-of-the-art
performance in the MuJoCo benchmarking environments, being about 3x more
sample efficient than the state-of-the-art algorithms, such as PETS, TD3 and SAC.
To explain the effectiveness of our algorithm, we show that the optimization surface
in parameter space is smoother than in action space. Further more, we found the
distilled policy network can be effectively applied without the expansive model
predictive control during test time for some environments such as Cheetah. Code
is released.

## Geometric Insights into the Convergence of Nonlinear TD Learning

Authors: ['David Brandfonbrener', 'Joan Bruna']

Ratings: ['3: Weak Reject', '6: Weak Accept', '8: Accept', '8: Accept']

Keywords: ['TD', 'nonlinear', 'convergence', 'value estimation', 'reinforcement learning']

While there are convergence guarantees for temporal difference (TD) learning when using linear function approximators, the situation for nonlinear models is far less understood, and divergent examples are known. Here we take a first step towards extending theoretical convergence guarantees to TD learning with nonlinear function approximation. More precisely, we consider the expected learning dynamics of the TD(0) algorithm for value estimation. As the step-size converges to zero, these dynamics are defined by a nonlinear ODE which depends on the geometry of the space of function approximators, the structure of the underlying Markov chain, and their interaction. We find a set of function approximators that includes ReLU networks and has geometry amenable to TD learning regardless of environment, so that the solution performs about as well as linear TD in the worst case. Then, we show how environments that are more reversible induce dynamics that are better for TD learning and prove global convergence to the true value function for well-conditioned function approximators. Finally, we generalize a divergent counterexample to a family of divergent problems to demonstrate how the interaction between approximator and environment can go wrong and to motivate the assumptions needed to prove convergence. 

## Q-learning with UCB Exploration is Sample Efficient for Infinite-Horizon MDP

Authors: ['Yuanhao Wang', 'Kefan Dong', 'Xiaoyu Chen', 'Liwei Wang']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['theory', 'reinforcement learning', 'sample complexity']

A fundamental question in reinforcement learning is whether model-free algorithms are sample efficient. Recently,  Jin et al. (2018) proposed a Q-learning algorithm with UCB exploration policy, and proved it has nearly optimal regret bound for finite-horizon episodic MDP. In this paper, we adapt Q-learning with UCB-exploration bonus to infinite-horizon MDP with discounted rewards \emph{without} accessing a generative model. We show that the \textit{sample complexity of exploration} of our algorithm is bounded by $\tilde{O}({\frac{SA}{\epsilon^2(1-\gamma)^7}})$. This improves the previously best known result of $\tilde{O}({\frac{SA}{\epsilon^4(1-\gamma)^8}})$ in this setting achieved by delayed Q-learning (Strehlet al., 2006),, and matches the lower bound in terms of $\epsilon$ as well as $S$ and $A$ up to logarithmic factors.

## Dynamical Distance Learning for Semi-Supervised and Unsupervised Skill Discovery

Authors: ['Kristian Hartikainen', 'Xinyang Geng', 'Tuomas Haarnoja', 'Sergey Levine']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['reinforcement learning', 'semi-supervised learning', 'unsupervised learning', 'robotics', 'deep learning']

Reinforcement learning requires manual specification of a reward function to learn a task. While in principle this reward function only needs to specify the task goal, in practice reinforcement learning can be very time-consuming or even infeasible unless the reward function is shaped so as to provide a smooth gradient towards a successful outcome. This shaping is difficult to specify by hand, particularly when the task is learned from raw observations, such as images. In this paper, we study how we can automatically learn dynamical distances: a measure of the expected number of time steps to reach a given goal state from any other state. These dynamical distances can be used to provide well-shaped reward functions for reaching new goals, making it possible to learn complex tasks efficiently. We show that dynamical distances can be used in a semi-supervised regime, where unsupervised interaction with the environment is used to learn the dynamical distances, while a small amount of preference supervision is used to determine the task goal, without any manually engineered reward function or goal examples. We evaluate our method both on a real-world robot and in simulation. We show that our method can learn to turn a valve with a real-world 9-DoF hand, using raw image observations and just ten preference labels, without any other supervision. Videos of the learned skills can be found on the project website: https://sites.google.com/view/dynamical-distance-learning

## Reinforced active learning for image segmentation

Authors: ['Arantxa Casanova', 'Pedro O. Pinheiro', 'Negar Rostamzadeh', 'Christopher J. Pal']

Ratings: ['6: Weak Accept', '6: Weak Accept']

Keywords: ['semantic segmentation', 'active learning', 'reinforcement learning']

Learning-based approaches for semantic segmentation have two inherent challenges. First, acquiring pixel-wise labels is expensive and time-consuming. Second, realistic segmentation datasets are highly unbalanced: some categories are much more abundant than others, biasing the performance to the most represented ones. In this paper, we are interested in focusing human labelling effort on a small subset of a larger pool of data, minimizing this effort while maximizing performance of a segmentation model on a hold-out set. We present a new active learning strategy for semantic segmentation based on deep reinforcement learning (RL). An agent learns a policy to select a subset of small informative image regions -- opposed to entire images -- to be labeled, from a pool of unlabeled data. The region selection decision is made based on predictions and uncertainties of the segmentation model being trained. Our method proposes a new modification of the deep Q-network (DQN) formulation for active learning, adapting it to the large-scale nature of semantic segmentation problems. We test the proof of concept in CamVid and provide results in the large-scale dataset Cityscapes. On Cityscapes, our deep RL region-based DQN approach requires roughly 30% less additional labeled data than our most competitive baseline to reach the same performance. Moreover, we find that our method asks for more labels of under-represented categories compared to the baselines, improving their performance and helping to mitigate class imbalance.

## Imitation Learning via Off-Policy Distribution Matching

Authors: ['Ilya Kostrikov', 'Ofir Nachum', 'Jonathan Tompson']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['reinforcement learning', 'deep learning', 'imitation learning', 'adversarial learning']

When performing imitation learning from expert demonstrations, distribution matching is a popular approach, in which one alternates between estimating distribution ratios and then using these ratios as rewards in a standard reinforcement learning (RL) algorithm. Traditionally, estimation of the distribution ratio requires on-policy data, which has caused previous work to either be exorbitantly data- inefficient or alter the original objective in a manner that can drastically change its optimum. In this work, we show how the original distribution ratio estimation objective may be transformed in a principled manner to yield a completely off-policy objective. In addition to the data-efficiency that this provides, we are able to show that this objective also renders the use of a separate RL optimization unnecessary. Rather, an imitation policy may be learned directly from this objective without the use of explicit rewards. We call the resulting algorithm ValueDICE and evaluate it on a suite of popular imitation learning benchmarks, finding that it can achieve state-of-the-art sample efficiency and performance.

## AMRL: Aggregated Memory For Reinforcement Learning

Authors: ['Jacob Beck', 'Kamil Ciosek', 'Sam Devlin', 'Sebastian Tschiatschek', 'Cheng Zhang', 'Katja Hofmann']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['deep learning', 'reinforcement learning', 'rl', 'memory', 'noise', 'machine learning']

In many partially observable scenarios, Reinforcement Learning (RL) agents must rely on long-term memory in order to learn an optimal policy. We demonstrate that using techniques from NLP and supervised learning fails at RL tasks due to stochasticity from the environment and from exploration. Utilizing our insights on the limitations of traditional memory methods in RL, we propose AMRL, a class of models that can learn better policies with greater sample efficiency and are resilient to noisy inputs. Specifically, our models use a standard memory module to summarize short-term context, and then aggregate all prior states from the standard model without respect to order. We show that this provides advantages both in terms of gradient decay and signal-to-noise ratio over time. Evaluating in Minecraft and maze environments that test long-term memory, we find that our model improves average return by 19% over a baseline that has the same number of parameters and by 9% over a stronger baseline that has far more parameters.

## CM3: Cooperative Multi-goal Multi-stage Multi-agent Reinforcement Learning

Authors: ['Jiachen Yang', 'Alireza Nakhaei', 'David Isele', 'Kikuo Fujimura', 'Hongyuan Zha']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['multi-agent reinforcement learning']

A variety of cooperative multi-agent control problems require agents to achieve individual goals while contributing to collective success. This multi-goal multi-agent setting poses difficulties for recent algorithms, which primarily target settings with a single global reward, due to two new challenges: efficient exploration for learning both individual goal attainment and cooperation for others' success, and credit-assignment for interactions between actions and goals of different agents. To address both challenges, we restructure the problem into a novel two-stage curriculum, in which single-agent goal attainment is learned prior to learning multi-agent cooperation, and we derive a new multi-goal multi-agent policy gradient with a credit function for localized credit assignment. We use a function augmentation scheme to bridge value and policy functions across the curriculum. The complete architecture, called CM3, learns significantly faster than direct adaptations of existing algorithms on three challenging multi-goal multi-agent problems: cooperative navigation in difficult formations, negotiating multi-vehicle lane changes in the SUMO traffic simulator, and strategic cooperation in a Checkers environment.

## Playing the lottery with rewards and multiple languages: lottery tickets in RL and NLP

Authors: ['Haonan Yu', 'Sergey Edunov', 'Yuandong Tian', 'Ari S. Morcos']

Ratings: ['3: Weak Reject', '3: Weak Reject', '6: Weak Accept']

Keywords: ['lottery tickets', 'nlp', 'transformer', 'rl', 'reinforcement learning']

The lottery ticket hypothesis proposes that over-parameterization of deep neural networks (DNNs) aids training by increasing the probability of a “lucky” sub-network initialization being present rather than by helping the optimization process (Frankle& Carbin, 2019). Intriguingly, this phenomenon suggests that initialization strategies for DNNs can be improved substantially, but the lottery ticket hypothesis has only previously been tested in the context of supervised learning for natural image tasks. Here, we evaluate whether “winning ticket” initializations exist in two different domains: natural language processing (NLP) and reinforcement learning (RL).For NLP, we examined both recurrent LSTM models and large-scale Transformer models (Vaswani et al., 2017). For RL, we analyzed a number of discrete-action space tasks, including both classic control and pixel control. Consistent with workin supervised image classification, we confirm that winning ticket initializations generally outperform parameter-matched random initializations, even at extreme pruning rates for both NLP and RL. Notably, we are able to find winning ticket initializations for Transformers which enable models one-third the size to achieve nearly equivalent performance. Together, these results suggest that the lottery ticket hypothesis is not restricted to supervised learning of natural images, but rather represents a broader phenomenon in DNNs.

## Intrinsic Motivation for Encouraging Synergistic Behavior

Authors: ['Rohan Chitnis', 'Shubham Tulsiani', 'Saurabh Gupta', 'Abhinav Gupta']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['reinforcement learning', 'intrinsic motivation', 'synergistic', 'robot manipulation']

We study the role of intrinsic motivation as an exploration bias for reinforcement learning in sparse-reward synergistic tasks, which are tasks where multiple agents must work together to achieve a goal they could not individually. Our key idea is that a good guiding principle for intrinsic motivation in synergistic tasks is to take actions which affect the world in ways that would not be achieved if the agents were acting on their own. Thus, we propose to incentivize agents to take (joint) actions whose effects cannot be predicted via a composition of the predicted effect for each individual agent. We study two instantiations of this idea, one based on the true states encountered, and another based on a dynamics model trained concurrently with the policy. While the former is simpler, the latter has the benefit of being analytically differentiable with respect to the action taken. We validate our approach in robotic bimanual manipulation and multi-agent locomotion tasks with sparse rewards; we find that our approach yields more efficient learning than both 1) training with only the sparse reward and 2) using the typical surprise-based formulation of intrinsic motivation, which does not bias toward synergistic behavior. Videos are available on the project webpage: https://sites.google.com/view/iclr2020-synergistic.

## DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames

Authors: ['Erik Wijmans', 'Abhishek Kadian', 'Ari Morcos', 'Stefan Lee', 'Irfan Essa', 'Devi Parikh', 'Manolis Savva', 'Dhruv Batra']

Ratings: ['3: Weak Reject', '8: Accept', '8: Accept']

Keywords: ['autonomous navigation', 'habitat', 'embodied AI', 'pointgoal navigation', 'reinforcement learning']

We present Decentralized Distributed Proximal Policy Optimization (DD-PPO), a method for distributed reinforcement learning in resource-intensive simulated environments. DD-PPO is distributed (uses multiple machines), decentralized (lacks a centralized server), and synchronous (no computation is ever "stale"), making it conceptually simple and easy to implement. In our experiments on training virtual robots to navigate in Habitat-Sim, DD-PPO exhibits near-linear scaling -- achieving a speedup of 107x on 128 GPUs over a serial implementation. We leverage this scaling to train an agent for 2.5 Billion steps of experience (the equivalent of 80 years of human experience) -- over 6 months of GPU-time training in under 3 days of wall-clock time with 64 GPUs. 

This massive-scale training not only sets the state of art on Habitat Autonomous Navigation Challenge 2019, but essentially "solves" the task -- near-perfect autonomous navigation in an unseen environment without access to a map, directly from an RGB-D camera and a GPS+Compass sensor.  Fortuitously, error vs computation exhibits a power-law-like distribution; thus, 90% of peak performance is obtained relatively early (at 100 million steps) and relatively cheaply (under 1 day with 8 GPUs). Finally, we show that the scene understanding and navigation policies learned can be transferred to other navigation tasks -- the analog of "ImageNet pre-training + task-specific fine-tuning" for embodied AI. Our model outperforms ImageNet pre-trained CNNs on these transfer tasks and can serve as a universal resource (all models and code are publicly available). 

## PAC Confidence Sets for Deep Neural Networks via Calibrated Prediction

Authors: ['Sangdon Park', 'Osbert Bastani', 'Nikolai Matni', 'Insup Lee']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['PAC', 'confidence sets', 'classification', 'regression', 'reinforcement learning']

We propose an algorithm combining calibrated prediction and generalization bounds from learning theory to construct confidence sets for deep neural networks with PAC guarantees---i.e., the confidence set for a given input contains the true label with high probability. We demonstrate how our approach can be used to construct PAC confidence sets on ResNet for ImageNet, a visual object tracking model, and a dynamics model for the half-cheetah reinforcement learning problem.

## Graph Constrained Reinforcement Learning for Natural Language Action Spaces

Authors: ['Prithviraj Ammanabrolu', 'Matthew Hausknecht']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['natural language generation', 'deep reinforcement learning', 'knowledge graphs', 'interactive fiction']

Interactive Fiction games are text-based simulations in which an agent interacts with the world purely through natural language. They are ideal environments for studying how to extend reinforcement learning agents to meet the challenges of natural language understanding, partial observability, and action generation in combinatorially-large text-based action spaces. We present KG-A2C, an agent that builds a dynamic knowledge graph while exploring and generates actions using a template-based action space. We contend that the dual uses of the knowledge graph to reason about game state and to constrain natural language generation are the keys to scalable exploration of combinatorially large natural language actions. Results across a wide variety of IF games show that KG-A2C outperforms current IF agents despite the exponential increase in action space size.

## Composing Task-Agnostic Policies with Deep Reinforcement Learning

Authors: ['Ahmed H. Qureshi', 'Jacob J. Johnson', 'Yuzhe Qin', 'Taylor Henderson', 'Byron Boots', 'Michael C. Yip']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['composition', 'transfer learning', 'deep reinforcement learning']

The composition of elementary behaviors to solve challenging transfer learning problems is one of the key elements in building intelligent machines. To date, there has been plenty of work on learning task-specific policies or skills but almost no focus on composing necessary, task-agnostic skills to find a solution to new problems. In this paper, we propose a novel deep reinforcement learning-based skill transfer and composition method that takes the agent's primitive policies to solve unseen tasks. We evaluate our method in difficult cases where training policy through standard reinforcement learning (RL) or even hierarchical RL is either not feasible or exhibits high sample complexity. We show that our method not only transfers skills to new problem settings but also solves the challenging environments requiring both task planning and motion control with high data efficiency.

## Single Episode Policy Transfer in Reinforcement Learning

Authors: ['Jiachen Yang', 'Brenden Petersen', 'Hongyuan Zha', 'Daniel Faissol']

Ratings: ['3: Weak Reject', '8: Accept', '8: Accept']

Keywords: ['transfer learning', 'reinforcement learning']

Transfer and adaptation to new unknown environmental dynamics is a key challenge for reinforcement learning (RL). An even greater challenge is performing near-optimally in a single attempt at test time, possibly without access to dense rewards, which is not addressed by current methods that require multiple experience rollouts for adaptation. To achieve single episode transfer in a family of environments with related dynamics, we propose a general algorithm that optimizes a probe and an inference model to rapidly estimate underlying latent variables of test dynamics, which are then immediately used as input to a universal control policy. This modular approach enables integration of state-of-the-art algorithms for variational inference or RL. Moreover, our approach does not require access to rewards at test time, allowing it to perform in settings where existing adaptive approaches cannot. In diverse experimental domains with a single episode test constraint, our method significantly outperforms existing adaptive approaches and shows favorable performance against baselines for robust transfer.

## Synthesizing Programmatic Policies that Inductively Generalize

Authors: ['Jeevana Priya Inala', 'Osbert Bastani', 'Zenna Tavares', 'Armando Solar-Lezama']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['Program synthesis', 'reinforcement learning', 'inductive generalization']

Deep reinforcement learning has successfully solved a number of challenging control tasks. However, learned policies typically have difficulty generalizing to novel environments. We propose an algorithm for learning programmatic state machine policies that can capture repeating behaviors. By doing so, they have the ability to generalize to instances requiring an arbitrary number of repetitions, a property we call inductive generalization. However, state machine policies are hard to learn since they consist of a combination of continuous and discrete structures. We propose a learning framework called adaptive teaching, which learns a state machine policy by imitating a teacher; in contrast to traditional imitation learning, our teacher adaptively updates itself based on the structure of the student. We show that our algorithm can be used to learn policies that inductively generalize to novel environments, whereas traditional neural network policies fail to do so. 

## Model-Augmented Actor-Critic: Backpropagating through Paths

Authors: ['Ignasi Clavera', 'Yao Fu', 'Pieter Abbeel']

Ratings: ['3: Weak Reject', '6: Weak Accept', '8: Accept']

Keywords: ['reinforcement learning', 'model-based', 'actor-critic', 'pathwise']

Current model-based reinforcement learning approaches use the model simply as a learned black-box simulator to augment the data for policy optimization or value function learning. In this paper, we show how to make more effective use of the model by exploiting its differentiability. We construct a policy optimization algorithm that uses the pathwise derivative of the learned model and policy across future timesteps. Instabilities of learning across many timesteps are prevented by using a terminal value function, learning the policy in an actor-critic fashion. Furthermore, we present a derivation on the monotonic improvement of our objective in terms of the gradient error in the model and value function. We show that our approach (i) is consistently more sample efficient than existing state-of-the-art model-based algorithms, (ii) matches the asymptotic performance of model-free algorithms, and (iii) scales to long horizons, a regime where typically past model-based approaches have struggled.

## Robust Reinforcement Learning for Continuous Control with Model Misspecification

Authors: ['Daniel J. Mankowitz', 'Nir Levine', 'Rae Jeong', 'Abbas Abdolmaleki', 'Jost Tobias Springenberg', 'Yuanyuan Shi', 'Jackie Kay', 'Todd Hester', 'Timothy Mann', 'Martin Riedmiller']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['reinforcement learning', 'robustness']

We provide a framework for incorporating robustness -- to perturbations in the transition dynamics which we refer to as model misspecification -- into continuous control Reinforcement Learning (RL) algorithms. We specifically focus on incorporating robustness into a state-of-the-art continuous control RL algorithm called Maximum a-posteriori Policy Optimization (MPO). We achieve this by learning a policy that optimizes for a worst case, entropy-regularized, expected return objective and derive a corresponding robust entropy-regularized Bellman contraction operator. In addition, we introduce a less conservative, soft-robust, entropy-regularized objective with a corresponding Bellman operator. We show that both, robust and soft-robust policies, outperform their non-robust counterparts in nine Mujoco domains with environment perturbations. In addition, we show improved robust performance on a challenging, simulated, dexterous robotic hand. Finally, we present multiple investigative experiments that provide a deeper insight into the robustness framework; including an adaptation to another continuous control RL algorithm. Performance videos can be found online at https://sites.google.com/view/robust-rl.

## Frequency-based Search-control in Dyna

Authors: ['Yangchen Pan', 'Jincheng Mei', 'Amir-massoud Farahmand']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['Model-based reinforcement learning', 'search-control', 'Dyna', 'frequency of a signal']

Model-based reinforcement learning has been empirically demonstrated as a successful strategy to improve sample efficiency. In particular, Dyna is an elegant model-based architecture integrating learning and planning that provides huge flexibility of using a model. One of the most important components in Dyna is called search-control, which refers to the process of generating state or state-action pairs from which we query the model to acquire simulated experiences. Search-control is critical in improving learning efficiency. In this work, we propose a simple and novel search-control strategy by searching high frequency regions of the value function. Our main intuition is built on Shannon sampling theorem from signal processing, which indicates that a high frequency signal requires more samples to reconstruct. We empirically show that a high frequency function is more difficult to approximate. This suggests a search-control strategy: we should use states from high frequency regions of the value function to query the model to acquire more samples. We develop a simple strategy to locally measure the frequency of a function by gradient and hessian norms, and provide theoretical justification for this approach. We then apply our strategy to search-control in Dyna, and conduct experiments to show its property and effectiveness on benchmark domains.

## Adaptive Correlated Monte Carlo for Contextual Categorical Sequence Generation

Authors: ['Xinjie Fan', 'Yizhe Zhang', 'Zhendong Wang', 'Mingyuan Zhou']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['binary softmax', 'discrete variables', 'policy gradient', 'pseudo actions', 'reinforcement learning', 'variance reduction']

Sequence generation models are commonly refined with reinforcement learning over user-defined metrics. However, high gradient variance hinders the practical use of this method. To stabilize this method, we adapt to contextual generation of categorical sequences a policy gradient estimator, which evaluates a set of correlated Monte Carlo (MC) rollouts for variance control. Due to the correlation, the number of unique rollouts is random and adaptive to model uncertainty; those rollouts naturally become baselines for each other, and hence are combined to effectively reduce gradient variance. We also demonstrate the use of correlated MC rollouts for binary-tree softmax models, which reduce the high generation cost in large vocabulary scenarios by decomposing each categorical action into a sequence of binary actions. We evaluate our methods on both neural program synthesis and image captioning. The proposed methods yield lower gradient variance and consistent improvement over related baselines. 

## Black-box Off-policy Estimation for Infinite-Horizon Reinforcement Learning

Authors: ['Ali Mousavi', 'Lihong Li', 'Qiang Liu', 'Denny Zhou']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['reinforcement learning', 'off-policy estimation', 'importance sampling', 'propensity score']

Off-policy estimation for long-horizon problems is important in many real-life applications such as healthcare and robotics, where high-fidelity simulators may not be available and on-policy evaluation is expensive or impossible.  Recently, \citet{liu18breaking} proposed an approach that avoids the curse of horizon suffered by typical importance-sampling-based methods. While showing promising results, this approach is limited in practice as it requires data being collected by a known behavior policy. In this work, we propose a novel approach that eliminates such limitations. In particular, we formulate the problem as solving for the fixed point of a "backward flow" operator and show that the fixed point solution gives the desired importance ratios of stationary distributions between the target and behavior policies.  We analyze its asymptotic consistency and finite-sample
generalization. Experiments on benchmarks verify the effectiveness of our proposed approach.


## The Gambler's Problem and Beyond

Authors: ['Baoxiang Wang', 'Shuai Li', 'Jiajin Li', 'Siu On Chan']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ["the gambler's problem", 'reinforcement learning', 'fractal', 'self-similarity', 'Bellman equation']

We analyze the Gambler's problem, a simple reinforcement learning problem where the gambler has the chance to double or lose their bets until the target is reached. This is an early example introduced in the reinforcement learning textbook by Sutton and Barto (2018), where they mention an interesting pattern of the optimal value function with high-frequency components and repeating non-smooth points. It is however without further investigation. We provide the exact formula for the optimal value function for both the discrete and the continuous cases. Though simple as it might seem, the value function is pathological: fractal, self-similar, derivative taking either zero or infinity, not smooth on any interval, and not written as elementary functions. It is in fact one of the generalized Cantor functions, where it holds a complexity that has been uncharted thus far. Our analyses could lead insights into improving value function approximation, gradient-based algorithms, and Q-learning, in real applications and implementations.

## Exploratory Not Explanatory: Counterfactual Analysis of Saliency Maps for Deep Reinforcement Learning

Authors: ['Akanksha Atrey', 'Kaleigh Clary', 'David Jensen']

Ratings: ['1: Reject', '3: Weak Reject', '8: Accept']

Keywords: ['explainability', 'saliency maps', 'representations', 'deep reinforcement learning']

Saliency maps are frequently used to support explanations of the behavior of deep reinforcement learning (RL) agents. However, a review of how saliency maps are used in practice indicates that the derived explanations are often unfalsifiable and can be highly subjective. We introduce an empirical approach grounded in counterfactual reasoning to test the hypotheses generated from saliency maps and assess the degree to which they correspond to the semantics of RL environments. We use Atari games, a common benchmark for deep RL, to evaluate three types of saliency maps. Our results show the extent to which existing claims about Atari games can be evaluated and suggest that saliency maps are best viewed as an exploratory tool rather than an explanatory tool.

## Multi-Agent Interactions Modeling with Correlated Policies

Authors: ['Minghuan Liu', 'Ming Zhou', 'Weinan Zhang', 'Yuzheng Zhuang', 'Jun Wang', 'Wulong Liu', 'Yong Yu']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['Multi-agent reinforcement learning', 'Imitation learning']

In multi-agent systems, complex interacting behaviors arise due to the high correlations among agents. However, previous work on modeling multi-agent interactions from demonstrations is primarily constrained by assuming the independence among policies and their reward structures. 
In this paper, we cast the multi-agent interactions modeling problem into a multi-agent imitation learning framework with explicit modeling of correlated policies by approximating opponents’ policies, which can recover agents' policies that can regenerate similar interactions. Consequently, we develop a Decentralized Adversarial Imitation Learning algorithm with Correlated policies (CoDAIL), which allows for decentralized training and execution. Various experiments demonstrate that CoDAIL can better regenerate complex interactions close to the demonstrators and outperforms state-of-the-art multi-agent imitation learning methods. Our code is available at \url{https://github.com/apexrl/CoDAIL}.

## Implementing Inductive bias for different navigation tasks through diverse RNN attrractors

Authors: ['Tie XU', 'Omri Barak']

Ratings: ['3: Weak Reject', '6: Weak Accept', '6: Weak Accept']

Keywords: ['navigation', 'Recurrent Neural Networks', 'dynamics', 'inductive bias', 'pre-training', 'reinforcement learning']

Navigation is crucial for animal behavior and is assumed to require an internal representation of the external environment, termed a cognitive map. The precise form of this representation is often considered to be a metric representation of space. An internal representation, however, is judged by its contribution to performance on a given task, and may thus vary between different types of navigation tasks. Here we train a recurrent neural network that controls an agent performing several navigation tasks in a simple environment. To focus on internal representations, we split learning into a task-agnostic pre-training stage that modifies internal connectivity and a task-specific Q learning stage that controls the network's output. We show that pre-training shapes the attractor landscape of the networks, leading to either a continuous attractor, discrete attractors or a disordered state. These structures induce bias onto the Q-Learning phase, leading to a performance pattern across the tasks corresponding to metric and topological regularities. Our results show that, in recurrent networks, inductive bias takes the form of attractor landscapes -- which can be shaped by pre-training and analyzed using dynamical systems methods. Furthermore, we demonstrate that non-metric representations are useful for navigation tasks.  

## Thinking While Moving: Deep Reinforcement Learning with Concurrent Control

Authors: ['Ted Xiao', 'Eric Jang', 'Dmitry Kalashnikov', 'Sergey Levine', 'Julian Ibarz', 'Karol Hausman', 'Alexander Herzog']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['deep reinforcement learning', 'continuous-time', 'robotics']

We study reinforcement learning in settings where sampling an action from the policy must be done concurrently with the time evolution of the controlled system, such as when a robot must decide on the next action while still performing the previous action. Much like a person or an animal, the robot must think and move at the same time, deciding on its next action before the previous one has completed. In order to develop an algorithmic framework for such concurrent control problems, we start with a continuous-time formulation of the Bellman equations, and then discretize them in a way that is aware of system delays. We instantiate this new class of approximate dynamic programming methods via a simple architectural extension to existing value-based deep reinforcement learning algorithms. We evaluate our methods on simulated benchmark tasks and a large-scale robotic grasping task where the robot must "think while moving."

## Evolutionary Population Curriculum for Scaling Multi-Agent Reinforcement Learning

Authors: ['Qian Long*', 'Zihan Zhou*', 'Abhinav Gupta', 'Fei Fang', 'Yi Wu†', 'Xiaolong Wang†']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['multi-agent reinforcement learning', 'evolutionary learning', 'curriculum learning']

In multi-agent games, the complexity of the environment can grow exponentially as the number of agents increases, so it is particularly challenging to learn good policies when the agent population is large. In this paper, we introduce Evolutionary Population Curriculum (EPC), a curriculum learning paradigm that scales up Multi-Agent Reinforcement Learning (MARL) by progressively increasing the population of training agents in a stage-wise manner. Furthermore, EPC uses an evolutionary approach to fix an objective misalignment issue throughout the curriculum: agents successfully trained in an early stage with a small population are not necessarily the best candidates for adapting to later stages with scaled populations. Concretely, EPC maintains multiple sets of agents in each stage, performs mix-and-match and fine-tuning over these sets and promotes the sets of agents with the best adaptability to the next stage. We implement EPC on a popular MARL algorithm, MADDPG, and empirically show that our approach consistently outperforms baselines by a large margin as the number of agents grows exponentially. The source code and videos can be found at https://sites.google.com/view/epciclr2020.

## Network Randomization: A Simple Technique for Generalization in Deep Reinforcement Learning

Authors: ['Kimin Lee', 'Kibok Lee', 'Jinwoo Shin', 'Honglak Lee']

Ratings: ['3: Weak Reject', '6: Weak Accept', '8: Accept']

Keywords: ['Deep reinforcement learning', 'Generalization in visual domains']

Deep reinforcement learning (RL) agents often fail to generalize to unseen environments (yet semantically similar to trained agents), particularly when they are trained on high-dimensional state spaces, such as images. In this paper, we propose a simple technique to improve a generalization ability of deep RL agents by introducing a randomized (convolutional) neural network that randomly perturbs input observations. It enables trained agents to adapt to new domains by learning robust features invariant across varied and randomized environments. Furthermore, we consider an inference method based on the Monte Carlo approximation to reduce the variance induced by this randomization. We demonstrate the superiority of our method across 2D CoinRun, 3D DeepMind Lab exploration and 3D robotics control tasks: it significantly outperforms various regularization and data augmentation methods for the same purpose.

## Reinforced Genetic Algorithm Learning for Optimizing Computation Graphs

Authors: ['Aditya Paliwal', 'Felix Gimeno', 'Vinod Nair', 'Yujia Li', 'Miles Lubin', 'Pushmeet Kohli', 'Oriol Vinyals']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['reinforcement learning', 'learning to optimize', 'combinatorial optimization', 'computation graphs', 'model parallelism', 'learning for systems']

We present a deep reinforcement learning approach to minimizing the execution cost of neural network computation graphs in an optimizing compiler. Unlike earlier learning-based works that require training the optimizer on the same graph to be optimized, we propose a learning approach that trains an optimizer offline and then generalizes to previously unseen graphs without further training. This allows our approach to produce high-quality execution decisions on real-world TensorFlow graphs in seconds instead of hours. We consider two optimization tasks for computation graphs: minimizing running time and peak memory usage. In comparison to an extensive set of baselines, our approach achieves significant improvements over classical and other learning-based methods on these two tasks. 

## RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments

Authors: ['Roberta Raileanu', 'Tim Rocktäschel']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['reinforcement learning', 'exploration', 'curiosity']

Exploration in sparse reward environments remains one of the key challenges of model-free reinforcement learning. Instead of solely relying on extrinsic rewards provided by the environment, many state-of-the-art methods use intrinsic rewards to encourage exploration. However, we show that existing methods fall short in procedurally-generated environments where an agent is unlikely to visit a state more than once. We propose a novel type of intrinsic reward which encourages the agent to take actions that lead to significant changes in its learned state representation. We evaluate our method on multiple challenging procedurally-generated tasks in MiniGrid, as well as on tasks with high-dimensional observations used in prior work. Our experiments demonstrate that this approach is more sample efficient than existing exploration methods, particularly for procedurally-generated MiniGrid environments. Furthermore, we analyze the learned behavior as well as the intrinsic reward received by our agent. In contrast to previous approaches, our intrinsic reward does not diminish during the course of training and it rewards the agent substantially more for interacting with objects that it can control.

## Projection-Based Constrained Policy Optimization

Authors: ['Tsung-Yen Yang', 'Justinian Rosca', 'Karthik Narasimhan', 'Peter J. Ramadge']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['Reinforcement learning with constraints', 'Safe reinforcement learning']

We consider the problem of learning control policies that optimize a reward function while satisfying constraints due to considerations of safety, fairness, or other costs. We propose a new algorithm - Projection-Based Constrained Policy Optimization (PCPO), an iterative method for optimizing policies in a two-step process - the first step performs an unconstrained update while the second step reconciles the constraint violation by projecting the policy back onto the constraint set. We theoretically analyze PCPO and provide a lower bound on reward improvement, as well as an upper bound on constraint violation for each policy update. We further characterize the convergence of PCPO with projection based on two different metrics - L2 norm and Kullback-Leibler divergence. Our empirical results over several control tasks demonstrate that our algorithm achieves superior performance, averaging more than 3.5 times less constraint violation and around 15% higher reward compared to state-of-the-art methods.

## Toward Evaluating Robustness of Deep Reinforcement Learning with Continuous Control

Authors: ['Tsui-Wei Weng', 'Krishnamurthy (Dj) Dvijotham*', 'Jonathan Uesato*', 'Kai Xiao*', 'Sven Gowal*', 'Robert Stanforth*', 'Pushmeet Kohli']

Ratings: ['3: Weak Reject', '6: Weak Accept', '6: Weak Accept']

Keywords: ['deep learning', 'reinforcement learning', 'robustness', 'adversarial examples']

Deep reinforcement learning has achieved great success in many previously difficult reinforcement learning tasks, yet recent studies show that deep RL agents are also unavoidably susceptible to adversarial perturbations, similar to deep neural networks in classification tasks. Prior works mostly focus on model-free adversarial attacks and agents with discrete actions. In this work, we study the problem of continuous control agents in deep RL with adversarial attacks and propose the first two-step algorithm based on learned model dynamics. Extensive experiments on various MuJoCo domains (Cartpole, Fish, Walker, Humanoid) demonstrate that our proposed framework is much more effective and efficient than model-free based attacks baselines in degrading agent performance as well as driving agents to unsafe states. 

## Structured Object-Aware Physics Prediction for Video Modeling and Planning

Authors: ['Jannik Kossen', 'Karl Stelzner', 'Marcel Hussing', 'Claas Voelcker', 'Kristian Kersting']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['self-supervised learning', 'probabilistic deep learning', 'structured models', 'video prediction', 'physics prediction', 'planning', 'variational auteoncoders', 'model-based reinforcement learning', 'VAEs', 'unsupervised', 'variational', 'graph neural networks', 'tractable probabilistic models', 'attend-infer-repeat', 'relational learning', 'AIR', 'sum-product networks', 'object-oriented', 'object-centric', 'object-aware', 'MCTS']

When humans observe a physical system, they can easily locate components, understand their interactions, and anticipate future behavior, even in settings with complicated and previously unseen interactions. For computers, however, learning such models from videos in an unsupervised fashion is an unsolved research problem.  In this paper, we present STOVE, a novel state-space model for  videos, which explicitly reasons about objects and their positions, velocities, and interactions. It is constructed by combining an image model and a dynamics model in compositional manner and improves on previous work by reusing the dynamics model for inference, accelerating and regularizing training. STOVE predicts videos with convincing physical behavior over hundreds of timesteps, outperforms previous unsupervised models, and even approaches the performance of supervised baselines. We further demonstrate the strength of our model as a simulator for sample efficient model-based control, in a task with heavily interacting objects.


## Making Efficient Use of Demonstrations to Solve Hard Exploration Problems

Authors: ['Caglar Gulcehre', 'Tom Le Paine', 'Bobak Shahriari', 'Misha Denil', 'Matt Hoffman', 'Hubert Soyer', 'Richard Tanburn', 'Steven Kapturowski', 'Neil Rabinowitz', 'Duncan Williams', 'Gabriel Barth-Maron', 'Ziyu Wang', 'Nando de Freitas', 'Worlds Team']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['imitation learning', 'deep learning', 'reinforcement learning']

This paper introduces R2D3, an agent that makes efficient use of demonstrations to solve hard exploration problems in partially observable environments with highly variable initial conditions. We also introduce a suite of eight tasks that combine these three properties, and show that R2D3 can solve several of the tasks where other state of the art methods (both with and without demonstrations) fail to see even a single successful trajectory after tens of billions of steps of exploration.

## Model-based reinforcement learning for biological sequence design

Authors: ['Christof Angermueller', 'David Dohan', 'David Belanger', 'Ramya Deshpande', 'Kevin Murphy', 'Lucy Colwell']

Ratings: ['3: Weak Reject', '6: Weak Accept', '6: Weak Accept']

Keywords: ['reinforcement learning', 'blackbox optimization', 'molecule design']

The ability to design biological structures such as DNA or proteins would have considerable medical and industrial impact. Doing so presents a challenging black-box optimization problem characterized by the large-batch, low round setting due to the need for labor-intensive wet lab evaluations. In response, we propose using reinforcement learning (RL) based on proximal-policy optimization (PPO) for biological sequence design. RL provides a flexible framework for optimization generative sequence models to achieve specific criteria, such as diversity among the high-quality sequences discovered. We propose a model-based variant of PPO, DyNA-PPO, to improve sample efficiency, where the policy for a new round is trained offline using a simulator fit on functional measurements from prior rounds. To accommodate the growing number of observations across rounds, the simulator model is automatically selected at each round from a pool of diverse models of varying capacity.  On the tasks of designing DNA transcription factor binding sites, designing antimicrobial proteins, and optimizing the energy of Ising models based on protein structure, we find that DyNA-PPO performs significantly better than existing methods in settings in which modeling is feasible, while still not performing worse in situations in which a reliable model cannot be learned.

## Meta Reinforcement Learning with Autonomous Inference of Subtask Dependencies

Authors: ['Sungryull Sohn', 'Hyunjae Woo', 'Jongwook Choi', 'Honglak Lee']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['Meta reinforcement learning', 'subtask graph']

We propose and address a novel few-shot RL problem, where a task is characterized by a subtask graph which describes a set of subtasks and their dependencies that are unknown to the agent. The agent needs to quickly adapt to the task over few episodes during adaptation phase to maximize the return in the test phase. Instead of directly learning a meta-policy, we develop a Meta-learner with Subtask Graph Inference (MSGI), which infers the latent parameter of the task by interacting with the environment and maximizes the return given the latent parameter. To facilitate learning, we adopt an intrinsic reward inspired by upper confidence bound (UCB) that encourages efficient exploration. Our experiment results on two grid-world domains and StarCraft II environments show that the proposed method is able to accurately infer the latent task parameter, and to adapt more efficiently than existing meta RL and hierarchical RL methods.

## Hypermodels for Exploration

Authors: ['Vikranth Dwaracherla', 'Xiuyuan Lu', 'Morteza Ibrahimi', 'Ian Osband', 'Zheng Wen', 'Benjamin Van Roy']

Ratings: ['3: Weak Reject', '6: Weak Accept', '8: Accept']

Keywords: ['exploration', 'hypermodel', 'reinforcement learning']

We study the use of hypermodels to represent epistemic uncertainty and guide exploration.
This generalizes and extends the use of ensembles to approximate Thompson sampling. The computational cost of training an ensemble grows with its size, and as such, prior work has typically been limited to ensembles with tens of elements. We show that alternative hypermodels can enjoy dramatic efficiency gains, enabling behavior that would otherwise require hundreds or thousands of elements, and even succeed in situations where ensemble methods fail to learn regardless of size.
This allows more accurate approximation of Thompson sampling as well as use of more sophisticated exploration schemes.  In particular, we consider an approximate form of information-directed sampling and demonstrate performance gains relative to Thompson sampling.  As alternatives to ensembles, we consider linear and neural network hypermodels, also known as hypernetworks.
We prove that, with neural network base models, a linear hypermodel can represent essentially any distribution over functions, and as such, hypernetworks do not extend what can be represented.

## Dynamics-Aware Embeddings

Authors: ['William Whitney', 'Rajat Agarwal', 'Kyunghyun Cho', 'Abhinav Gupta']

Ratings: ['3: Weak Reject', '6: Weak Accept', '8: Accept', '8: Accept']

Keywords: ['representation learning', 'reinforcement learning', 'rl']

In this paper we consider self-supervised representation learning to improve sample efficiency in reinforcement learning (RL). We propose a forward prediction objective for simultaneously learning embeddings of states and actions. These embeddings capture the structure of the environment's dynamics, enabling efficient policy learning. We demonstrate that our action embeddings alone improve the sample efficiency and peak performance of model-free RL on control from low-dimensional states. By combining state and action embeddings, we achieve efficient learning of high-quality policies on goal-conditioned continuous control from pixel observations in only 1-2 million environment steps.

## Never Give Up: Learning Directed Exploration Strategies

Authors: ['Adrià Puigdomènech Badia', 'Pablo Sprechmann', 'Alex Vitvitskyi', 'Daniel Guo', 'Bilal Piot', 'Steven Kapturowski', 'Olivier Tieleman', 'Martin Arjovsky', 'Alexander Pritzel', 'Andrew Bolt', 'Charles Blundell']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['deep reinforcement learning', 'exploration', 'intrinsic motivation']

We propose a reinforcement learning agent to solve hard exploration games by learning a range of directed exploratory policies. We construct an episodic memory-based intrinsic reward using k-nearest neighbors over the agent's recent experience to train the directed exploratory policies, thereby encouraging the agent to repeatedly revisit all states in its environment. A self-supervised inverse dynamics model is used to train the embeddings of the nearest neighbour lookup, biasing the novelty signal towards what the agent can control. We employ the framework of Universal Value Function Approximators to simultaneously learn many directed exploration policies with the same neural network, with different trade-offs between exploration and exploitation. By using the same neural network for different degrees of exploration/exploitation, transfer is demonstrated from predominantly exploratory policies yielding effective exploitative policies. The proposed method can be incorporated to run with modern distributed RL agents that collect large amounts of experience from many actors running in parallel on separate environment instances. Our method doubles the performance of the base agent in all hard exploration in the Atari-57 suite while maintaining a very high score across the remaining games, obtaining a median human normalised score of 1344.0%. Notably, the proposed method is the first algorithm to achieve non-zero rewards (with a mean score of 8,400) in the game of Pitfall! without using demonstrations or hand-crafted features.

## Learning Nearly Decomposable Value Functions Via Communication Minimization

Authors: ['Tonghan Wang*', 'Jianhao Wang*', 'Chongyi Zheng', 'Chongjie Zhang']

Ratings: ['3: Weak Reject', '6: Weak Accept', '6: Weak Accept']

Keywords: ['Multi-agent reinforcement learning', 'Nearly decomposable value function', 'Minimized communication']

Reinforcement learning encounters major challenges in multi-agent settings, such as scalability and non-stationarity. Recently, value function factorization learning emerges as a promising way to address these challenges in collaborative multi-agent systems. However, existing methods have been focusing on learning fully decentralized value functions, which are not efficient for tasks requiring communication. To address this limitation, this paper presents a novel framework for learning nearly decomposable Q-functions (NDQ) via communication minimization, with which agents act on their own most of the time but occasionally send messages to other agents in order for effective coordination. This framework hybridizes value function factorization learning and communication learning by introducing two information-theoretic regularizers. These regularizers are maximizing mutual information between agents' action selection and communication messages while minimizing the entropy of messages between agents. We show how to optimize these regularizers in a way that is easily integrated with existing value function factorization methods such as QMIX. Finally, we demonstrate that, on the StarCraft unit micromanagement benchmark, our framework significantly outperforms baseline methods and allows us to cut off more than $80\%$ of communication without sacrificing the performance. The videos of our experiments are available at https://sites.google.com/view/ndq.

## Learning to Coordinate Manipulation Skills via Skill Behavior Diversification

Authors: ['Youngwoon Lee', 'Jingyun Yang', 'Joseph J. Lim']

Ratings: ['6: Weak Accept', '6: Weak Accept', '6: Weak Accept']

Keywords: ['reinforcement learning', 'hierarchical reinforcement learning', 'modular framework', 'skill coordination', 'bimanual manipulation']

When mastering a complex manipulation task, humans often decompose the task into sub-skills of their body parts, practice the sub-skills independently, and then execute the sub-skills together. Similarly, a robot with multiple end-effectors can perform complex tasks by coordinating sub-skills of each end-effector. To realize temporal and behavioral coordination of skills, we propose a modular framework that first individually trains sub-skills of each end-effector with skill behavior diversification, and then learns to coordinate end-effectors using diverse behaviors of the skills. We demonstrate that our proposed framework is able to efficiently coordinate skills to solve challenging collaborative control tasks such as picking up a long bar, placing a block inside a container while pushing the container with two robot arms, and pushing a box with two ant agents. Videos and code are available at https://clvrai.com/coordination

