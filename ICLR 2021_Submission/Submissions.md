# Submissions

## Optimism in Reinforcement Learning with Generalized Linear Function Approximation

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'optimism', 'exploration', 'function approximation', 'theory', 'regret analysis', 'provable sample efficiency']

We design a new provably efficient algorithm for episodic reinforcement learning with generalized linear function approximation. We analyze the algorithm under a new expressivity assumption that we call "optimistic closure," which is strictly weaker than assumptions from prior analyses for the linear setting. With optimistic closure, we prove that our algorithm enjoys a regret bound of $\tilde{O}(\sqrt{d^3T})$ where d is the dimensionality of the state-action features and T is the number of episodes. This is the first statistically and computationally efficient algorithm for reinforcement learning with generalized linear functions.

## Hellinger Distance Constrained Regression

Authors: ['Anonymous']

Keywords: ['offline', 'Reinforcement Learning', 'off-policy', 'control']

This paper introduces the off-policy reinforcement learning method that uses the Hellinger distance between sampling policy and current policy as a constraint. Hellinger distance squared multiplied by two is greater than or equal to total variation distance squared and less than or equal to Kullback-Leibler divergence, therefore lower bound for expected discounted return for the new policy is improved comparing to KL. Also, Hellinger distance is less than or equal to 1, so there is a policy-independent lower bound for expected discounted return. HDCR is capable of training with Experience Replay, a common setting for distributed RL when collecting trajectories using different policies and learning from this data centralized. HDCR shows results comparable to or better than Advantage-weighted Behavior Model and Advantage-Weighted Regression on MuJoCo tasks using offline datasets collected by random agents and datasets obtained during the first iterations of online training of HDCR agent. 

## Episodic Memory for Learning Subjective-Timescale Models

Authors: ['Anonymous']

Keywords: ['Episodic Memory', 'Time Perception', 'Active Inference', 'Model-based Reinforcement Learning']

In model-based learning, an agent’s model is commonly defined over transitions between consecutive states of an environment even though planning often requires reasoning over multi-step timescales, with intermediate states either unnecessary, or worse, accumulating prediction error. In contrast, intelligent behaviour in biological organisms is characterised by the ability to plan over varying temporal scales depending on the context. Inspired by the recent works on human time perception, we devise a novel approach to learning a transition dynamics model, based on the sequences of episodic memories that define the agent's subjective timescale – over which it learns world dynamics and over which future planning is performed. We implement this in the framework of active inference and demonstrate that the resulting subjective-timescale model (STM) can systematically vary the temporal extent of its predictions while preserving the same computational efficiency. Additionally, we show that STM predictions are more likely to introduce future salient events (for example new objects coming into view), incentivising exploration of new areas of the environment. As a result, STM produces more informative action-conditioned roll-outs that assist the agent in making better decisions. We validate significant improvement in our STM agent's performance in the Animal-AI environment against a baseline system, trained using the environment's objective-timescale dynamics.

## Robust Reinforcement Learning on State Observations with Learned Optimal Adversary

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'robustness', 'perturbations on state observations']

We study the robustness of reinforcement learning (RL) with adversarially perturbed state observations,  which aligns with the setting of many adversarial at-tacks to deep reinforcement learning (DRL) and is also important for rolling out real-world RL agent under unpredictable sensing noise.  With a fixed agent pol-icy, we demonstrate that an optimal adversary to perturb state observations can be found, which is guaranteed to obtain worst-case agent reward.  For DRL set-tings, this leads to a novel empirical adversarial attack to RL agents via a learned adversary that is much stronger than previous ones. To enhance the robustness of an agent, we propose a framework of alternating training with learned adversaries (ATLA), which trains an adversary online together with the agent using policy gradient following the optimal adversarial attack framework.  Additionally, inspired by the analysis of state-adversarial Markov decision process (SA-MDP), we show that past states and actions (history) can be useful for learning a robust agent, and we empirically find an LSTM based policy can be more robust under adversaries. Empirical evaluation on a few continuous control environments shows that ATLA achieves state-of-the-art performance under strong adversaries.

## PERIL: Probabilistic Embeddings for hybrid Meta-Reinforcement and Imitation Learning

Authors: ['Anonymous']

Keywords: ['Meta-learning', 'Imitation Learning', 'Reinforcement Learning']

Imitation learning is a natural way for a human to describe a task to an agent, and it can be combined with reinforcement learning to enable the agent to solve that task through exploration. However, traditional methods which combine imitation learning and reinforcement learning require a very large amount of interaction data to learn each new task, even when bootstrapping from a demonstration. One solution to this is to use meta reinforcement learning (meta-RL) to enable an agent to quickly adapt to new tasks at test time. In this work, we introduce a new method to combine imitation learning with meta reinforcement learning, Probabilistic Embeddings for hybrid meta-Reinforcement and Imitation Learning (PERIL). Dual inference strategies allow PERIL to precondition exploration policies on demonstrations, which greatly improves adaptation rates in unseen tasks. In contrast to pure imitation learning, our approach is capable of exploring beyond the demonstration, making it robust to task alterations and uncertainties. By exploiting the flexibility of meta-RL, we show how PERIL is capable of interpolating from within previously learnt dynamics to adapt to unseen tasks, as well as unseen task families, within a set of meta-RL benchmarks under sparse rewards.

## Monotonic Robust Policy Optimization with Model Discrepancy

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'generalization']

State-of-the-art deep reinforcement learning (DRL) algorithms tend to overfit in some specific environments due to the lack of data diversity in training. To mitigate the model discrepancy between training and target (testing) environments, domain randomization (DR) can generate plenty of environments with a sufficient diversity by randomly sampling environment parameters in simulator. Though standard DR using a uniform distribution improves the average performance on the whole range of environments, the worst-case environment is usually neglected without any performance guarantee. Since the average and worst-case performance are equally important for the generalization in RL, in this paper, we propose a policy optimization approach for concurrently improving the policy's performance in the average case (i.e., over all possible environments) and the worst-case environment. We theoretically derive a lower bound for the worst-case performance of a given policy over all environments. Guided by this lower bound, we formulate an optimization problem which aims to optimize the policy and sampling distribution together, such that the constrained expected performance of all environments is maximized. We prove that the worst-case performance is monotonically improved by iteratively solving this optimization problem. Based on the proposed lower bound, we develop a practical algorithm, named monotonic robust policy optimization (MRPO), and validate MRPO on several robot control tasks. By modifying the environment parameters in simulation, we obtain environments for the same task but with different transition dynamics for training and testing. We demonstrate that MRPO can improve both the average and worst-case performance in the training environments, and facilitate the learned policy with a better generalization capability in unseen testing environments.

## Structure and randomness in planning and reinforcement learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'uncertainty', 'model-based', 'MCTS']

Planning in large state spaces inevitably needs to balance depth and breadth of the search. It has a crucial impact on planners performance and most manage this interplay implicitly. We present a novel method $\textit{Shoot Tree Search (STS)}$, which makes it possible to control this trade-off more explicitly. Our algorithm can be understood as an interpolation between two celebrated search mechanisms: MCTS and random shooting. It also lets the user control the bias-variance trade-off, akin to $TD(n)$, but in the tree search context.

In experiments on challenging domains, we show that STS can get the best of both worlds consistently achieving higher scores. 

## Optimizing Memory Placement using Evolutionary Graph Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Memory Mapping', 'Device Placement', 'Evolutionary Algorithms']

For deep neural network accelerators, memory movement is both energetically expensive and can bound computation. Therefore, optimal mapping of tensors to memory hierarchies is critical to performance. The growing complexity of neural networks calls for automated memory mapping instead of manual heuristic approaches; yet the search space of neural network computational graphs have previously been prohibitively large. We introduce Evolutionary Graph Reinforcement Learning (EGRL), a method designed for large search spaces, that combines graph neural networks, reinforcement learning, and evolutionary search. A set of fast, stateless policies guide the evolutionary search to improve its sample-efficiency. We train and validate our approach directly on the Intel NNP-I chip for inference. EGRL outperforms policy-gradient, evolutionary search and dynamic programming baselines on BERT, ResNet-101 and ResNet-50. We additionally achieve 28-78% speed-up compared to the native NNP-I compiler on all three workloads.  

## EpidemiOptim: A Toolbox for the Optimization of Control Policies in Epidemiological Models

Authors: ['Anonymous']

Keywords: ['epidemiology', 'covid19', 'reinforcement learning', 'evolutionary algorithms', 'multi-objective optimization', 'decision-making', 'toolbox']

Epidemiologists model the dynamics of epidemics in order to propose control strategies based on pharmaceutical and non-pharmaceutical interventions (contact limitation, lock down, vaccination, etc). Hand-designing such strategies is not trivial because of the number of possible interventions and the difficulty to predict long-term effects. This task can be cast as an optimization problem where state-of-the-art machine learning algorithms such as deep reinforcement learning might bring significant value. However, the specificity of each domain - epidemic modelling or solving optimization problems - requires strong collaborations between researchers from different fields of expertise.
This is why we introduce EpidemiOptim, a Python toolbox that facilitates collaborations between researchers in epidemiology and optimization. EpidemiOptim turns epidemiological models and cost functions into optimization problems via a standard interface commonly used by optimization practitioners (OpenAI Gym). Reinforcement learning algorithms based on Q-Learning with deep neural networks (DQN) and evolutionary algorithms (NSGA-II) are already implemented. We illustrate the use of EpidemiOptim to find optimal policies for dynamical on-off lock-down control under the optimization of death toll and economic recess using a Susceptible-Exposed-Infectious-Removed (SEIR) model for SARS-CoV-2/COVID-19. 
Using EpidemiOptim and its interactive visualization platform in Jupyter notebooks, epidemiologists, optimization practitioners and others (e.g. economists) can easily compare epidemiological models, costs functions and optimization algorithms to address important choices to be made by health decision-makers.

## Data-efficient Hindsight Off-policy Option Learning

Authors: ['Anonymous']

Keywords: ['Hierarchical Reinforcement Learning', 'Off-Policy', 'Abstractions', 'Data-Efficiency']

Hierarchical approaches for reinforcement learning aim to improve data efficiency and accelerate learning by incorporating different abstractions. We introduce Hindsight Off-policy Options (HO2), an efficient off-policy option learning algorithm, and isolate the impact of action and temporal abstraction in the option framework by comparing flat policies, mixture policies without temporal abstraction, and finally option policies; all with comparable policy optimization. When aiming for data efficiency, we demonstrate the importance of off-policy optimization, as even flat policies trained off-policy can outperform on-policy option methods. In addition, off-policy training and backpropagation through a dynamic programming inference procedure -- through time and through the policy components for every time-step -- enable us to train all components' parameters independently of the data-generating behavior policy. We continue to illustrate challenges in off-policy option learning and the related importance of trust-region constraints. Experimentally, we demonstrate that HO2 outperforms existing option learning methods and that both action and temporal abstraction provide strong benefits in particular in more demanding simulated robot manipulation tasks from raw pixel inputs. Finally, we develop an intuitive extension to encourage temporal abstraction and investigate differences in its impact between learning from scratch and using pre-trained options. 

## Hindsight Curriculum Generation Based Multi-Goal Experience Replay

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'multi-goal task', 'experience replay']

In multi-goal tasks with sparse rewards, it is challenging to learn from tons of experiences with zero rewards. Hindsight experience replay (HER), which replays past experiences with additional heuristic goals, has shown it possible for off-policy reinforcement learning (RL) to make use of failed experiences. However, the replayed experiences may not lead to well-explored state-action pairs, especially for a pseudo goal, which instead results in a poor estimate of the value function. To tackle the problem, we propose to resample hindsight experiences based on their likelihood under the current policy and the overall distribution. Based on the hindsight strategy, we introduce a novel multi-goal experience replay method that automatically generates a training curriculum, namely Hindsight Curriculum Generation (HCG). As the range of experiences expands, the generated curriculum strikes a dynamic balance between exploiting and exploring. We implement HCG with the vanilla Deep Deterministic Policy Gradient(DDPG), and experiments on several tasks with sparse binary rewards demonstrate that HCG improves sample efficiency of the state of the art.

## CausalWorld: A Robotic Manipulation Benchmark for Causal Structure and Transfer Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'transfer learning', 'sim2real transfer', 'domain adaptation', 'causality', 'generalization', 'robotics']

Despite recent successes of reinforcement learning (RL), it remains a challenge for agents to transfer learned skills to related environments. To facilitate research addressing this, we propose CausalWorld, a benchmark for causal structure and transfer learning in a robotic manipulation environment. The environment is a simulation of an open-source robotic platform, hence offering the possibility of sim-to-real transfer.  Tasks consist of constructing 3D shapes from a given set of blocks - inspired by how children learn to build complex structures.  The key  strength of CausalWorld is that it provides a combinatorial family of such tasks with  common causal structure and underlying factors (including, e.g., robot and object masses, colors, sizes). The user (or the agent) may intervene on all causal variables, which allows for fine-grained control over how similar different  tasks (or task distributions) are. One can thus easily define training and evaluation distributions of a desired difficulty level, targeting a specific form of generalization (e.g., only changes in appearance or object mass).  Further, this common parametrization facilitates defining curricula by interpolating between an initial and a target task. While users may define their own task distributions, we present eight meaningful distributions as concrete benchmarks, ranging from simple to very challenging, all of which require long-horizon planning and precise low-level motor control at the same time. Finally, we provide baseline results for a subset of these tasks on distinct training curricula and corresponding evaluation protocols, verifying the feasibility of the tasks in this benchmark.

## Addressing Extrapolation Error in Deep Offline Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Addressing Extrapolation Error in Deep Offline Reinforcement Learning']

Reinforcement learning (RL) encompasses both online and offline regimes.  Unlike its online counterpart, offline RL agents are trained using logged-data only, without interaction with the environment.  Therefore, offline RL is a promising direction for real-world applications, such as healthcare, where repeated interaction with environments is prohibitive.  However, since offline RL losses often involve evaluating state-action pairs not well-covered by training data, they can suffer due to the errors introduced when the function approximator attempts to extrapolate those pairs' value. These errors can be compounded by bootstrapping when the function approximator overestimates, leading the value function to *grow unbounded*, thereby crippling learning. In this paper, we introduce a three-part solution to combat extrapolation errors: (i) behavior value estimation, (ii) ranking regularization, and (iii) reparametrization of the value function.  We provide ample empirical evidence on the effectiveness of our method, showing state of the art performance on the RL Unplugged (RLU) ATARI dataset. Furthermore, we introduce new datasets for bsuite as well as partially observable DeepMind Lab environments, on which our method outperforms state of the art offline RL algorithms. 


## Interpretable Reinforcement Learning With Neural Symbolic Logic

Authors: ['Anonymous']

Keywords: ['Interpretable Reinforcement Learning', 'Neural Symbolic Logic']

Recent progress in deep reinforcement learning (DRL) can be largely attributed to the use of neural networks.  However, this black-box approach fails to explain the learned policy in a human understandable way.  To address this challenge and improve the transparency, we introduce symbolic logic into DRL and propose a Neural Symbolic Reinforcement Learning framework, in which states and actions are represented in an interpretable way using first-order logic.  This framework features a relational reasoning module, which performs on task-level in Hierarchical Reinforcement Learning, enabling end-to-end learning with prior symbolic knowledge.  Moreover, interpretability is enabled by extracting the logical rules learned by the reasoning module in a symbolic rule space, providing explainability on task level. Experimental results demonstrate better interpretability of subtasks, along with competing performance compared with existing approaches.

## Deep Reinforcement Learning for Optimal Stopping with Application in Financial Engineering

Authors: ['Anonymous']

Keywords: ['Reinforcement learning', 'deep learning', 'financial engineering', 'optimal stopping.']

Optimal stopping is the problem of deciding the right time at which to take a particular action in a stochastic system, in order to maximize an expected reward. It has many applications in areas such as finance, healthcare, and statistics. In this paper, we employ deep Reinforcement Learning (RL) to learn optimal stopping policies in two financial engineering applications: namely option pricing, and optimal option exercise. We present for the first time a comprehensive empirical evaluation of the quality of optimal stopping policies identified by three state of the art deep RL algorithms: double deep Q-learning (DDQN), categorical distributional RL (C51), and Implicit Quantile Networks (IQN). In the case of option pricing, our findings indicate that in a theoretical Black-Schole environment, IQN successfully identifies nearly optimal prices. On the other hand, it is slightly outperformed by C51 when confronted to real stock data movements in a put option exercise problem that involves assets from the S&P500 index. More importantly, the C51 algorithm is able to identify an optimal stopping policy that achieves 8% more out-of-sample returns than the best of four natural benchmark policies. We conclude with a discussion of our findings which should pave the way for relevant future research.

## Convergence Proof for Actor-Critic Methods Applied to PPO and RUDDER

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'actor critic algorithms', 'policy gradient methods', 'stochastic approximation', 'PPO', 'RUDDER']

We prove under commonly used assumptions the convergence of actor-critic reinforcement learning algorithms, which simultaneously learn a policy function, the actor, and a value function, the critic. Both functions can be deep neural networks of arbitrary complexity. Our framework allows showing convergence of the well known Proximal Policy Optimization (PPO) and of the recently introduced RUDDER. For the convergence proof  we employ recently introduced techniques from the two time-scale stochastic approximation theory. Our results are valid for actor-critic methods that use episodic samples and that have a policy that becomes more greedy during learning. Previous convergence proofs assume linear function approximation, cannot treat episodic examples, or do not consider that policies become greedy. The latter is relevant since optimal policies are typically deterministic. 

## DECSTR: Learning Goal-Directed Abstract Behaviors using Pre-Verbal Spatial Predicates in Intrinsically Motivated Agents

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement Learning']

Past research showed that preverbal infants represent abstract spatial relations between physical objects and demonstrate goal-oriented sensorimotor behavior before they even start acquiring language. Inspired by these findings, we introduce DECSTR, an intrinsically motivated agent endowed with a high-level representation of spatial relations between objects that freely explores its environment and sets its own goals as target constraints on these spatial relations. In a robotic manipulation environment, DECSTR explores the corresponding representation space by manipulating objects, and efficiently learns to achieve any reachable configuration within it. It does so by leveraging an object-centered modular architecture, a symmetry inductive bias, and a new form of automatic curriculum learning for goal selection and policy learning. We explicitly compare the use of this abstract state and goal representations to the traditional goal-as-state framework and show that DECSTR can fulfill goals in an opportunistic manner, making profit of the current configuration of objects to achieve its goals. Additionally, by contrast with traditional instruction-following agents which are trained simultaneously to understand language and to act, DECSTR decouples goal-oriented sensorimotor learning from language acquisition, as in infants. This decoupling is facilitated by the abstract spatial relations, as they better match the natural semantics of linguistic inputs than traditional continuous representations. We show this behavior in a proof-of-concept setup, where trained agents learn to associate their behavioral repertoire with simple language commands via a language-conditioned goal generator. We show that this goal generator increases the behavioral diversity of a given language instruction and enables retry behaviors when the agent fails.

## Network Reusability Analysis for Multi-Joint Robot Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Machine Learning', 'Robot Motion Learning', 'DQN', 'Robot Manipulator', 'Target Reaching', 'Network Pruning']

Recent experiments indicate that pre-training of end-to-end Reinforcement Learning neural networks on general tasks can speed up the training process for a specific application. However, it remains open if these networks form general feature extractors that are reused and a hierarchical organization as apparent e.g. in Convolutional Neural Networks.  In this paper we analyze the intrinsic neuron activation in networks trained for target reaching of planar robot manipulators with increasing joint number. We analyze the individual neuron activity distribution in the network, introduce a pruning algorithm to reduce the network size, and with these dense networks we spot correlations of neuron activity patterns among networks trained for robot manipulators with different joint number. We show that the input and output network layers have more distinct neuron activation in contrast to hidden layers. Our pruning algorithm reduces the network size significantly, increases the euclidean distance of neuron activation, but keeps a high performance in training and evaluation. Our results demonstrate correlations of neuron activity among networks trained for robots with different complexity. Hereby, robots with small joint number difference show higher layer-wise correlations whereas more different robots mostly show correlations to the first input layer.

## Differentiable Trust Region Layers for Deep Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'trust region', 'policy gradient', 'projection', 'Wasserstein distance', 'Kullback-Leibler divergence', 'Frobenius norm']

Trust region methods are a popular tool in reinforcement learning as they yield robust policy updates in continuous and discrete action spaces. However, enforcing such trust regions in deep reinforcement learning is difficult.  Hence, many approaches, such as Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO), are based on approximations. Due to those approximations,they violate the constraints or fail to find the optimal solution within the trust region.  Moreover, they are difficult to implement, lack sufficient exploration, and have been shown to depend on code-level optimizations.  In this work, we pro-pose differentiable neural network layers to enforce trust regions for deep Gaussian policies via closed-form projections.  Unlike existing methods, those layers formalize trust regions for each state individually and can complement existing reinforcement learning algorithms.  We derive trust region projections based on the Kullback-Leibler divergence, the Wasserstein L2 distance, and the Frobenius norm for Gaussian distributions and empirically demonstrate that those projection layers achieve similar or better results than existing methods while being almost agnostic to specific implementation choices.

## Self-supervised Visual Reinforcement Learning with Object-centric Representations

Authors: ['Anonymous']

Keywords: ['self-supervision', 'autonomous learning', 'object-centric representations', 'visual reinforcement learning']

Autonomous agents need large repertoires of skills to act reasonably on new tasks that they have not seen before.
However, acquiring these skills using only a stream of high-dimensional, unstructured, and unlabeled observations is a tricky challenge for any autonomous agent. Previous methods have used variational autoencoders to encode a scene into a low-dimensional vector that can be used as a goal for an agent to discover new skills. Nevertheless, in compositional/multi-object environments it is difficult to disentangle all the factors of variation into such a fixed-length representation of the whole scene. We propose to use emph{object-centric representations as a modular and structured observation space, which is learned with a compositional generative world model. We show that the structure in the representations in combination with goal-conditioned attention policies helps the autonomous agent to discover and learn useful skills. And these skills can be further combined to solve complex compositional tasks like the manipulation of several different objects.

## Temporally-Extended ε-Greedy Exploration

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'exploration']

Recent work on exploration in reinforcement learning (RL) has led to a series of increasingly complex solutions to the problem. This increase in complexity often comes at the expense of generality. Recent empirical studies suggest that, when applied to a broader set of domains, some sophisticated exploration methods are outperformed by simpler counterparts, such as ε-greedy. In this paper we propose an exploration algorithm that retains the simplicity of ε-greedy while reducing dithering. We build on a simple hypothesis: the main limitation of ε-greedy exploration is its lack of temporal persistence, which limits its ability to escape local optima. We propose a temporally extended form of ε-greedy that simply repeats the sampled action for a random duration. It turns out that, for many duration distributions, this suffices to improve exploration on a large set of domains. Interestingly, a class of distributions inspired by ecological models of animal foraging behaviour yields particularly strong performance.

## Learning to Sample with Local and Global Contexts  in Experience Replay Buffer

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'experience replay buffer', 'off-policy RL']

Experience replay, which enables the agents to remember and reuse experience from the past, has played a significant role in the success of off-policy reinforcement learning (RL). To utilize the experience replay efficiently, the existing sampling methods allow selecting out more meaningful experiences by imposing priorities on them based on certain metrics (e.g. TD-error). However, they may result in sampling highly biased, redundant transitions since they compute the sampling rate for each transition independently, without consideration of its importance in relation to other transitions. In this paper, we aim to address the issue by proposing a new learning-based sampling method that can compute the relative importance of transition. To this end, we design a novel permutation-equivariant neural architecture that takes contexts from not only features of each transition (local) but also those of others (global) as inputs. We validate our framework, which we refer to as Neural Experience Replay Sampler (NERS), on multiple benchmark tasks for both continuous and discrete control tasks and show that it can significantly improve the performance of various off-policy RL methods. Further analysis confirms that the improvements of the sample efficiency indeed are due to sampling diverse and meaningful transitions by NERS that considers both local and global contexts.

## Using Deep Reinforcement Learning to Train and Evaluate Instructional Sequencing Policies for an Intelligent Tutoring System

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement Learning', 'Intelligent Tutoring Systems', 'Adaptive policy', 'Instructional Sequencing']

We present STEP, a novel Deep Reinforcement Learning solution to the problem of learning instructional sequencing.  STEP has three components: 1. Simulate the student by fitting a knowledge tracing model to data logged by an intelligent tutoring system. 2. Train instructional sequencing policies by using Proximal Policy Optimization. 3. Evaluate the learned instructional policies by estimating their local and global impact on learning gains.  STEP leverages the student model by representing the student’s knowledge state as a probability vector of knowing each skill and using the student’s estimated learning gains as its reward function to evaluate candidate policies.  A learned policy represents a mapping from each state to an action that maximizes the reward, i.e. the upward distance to the next state in the multi-dimensional space. We use STEP to discover and evaluate potential improvements to a literacy and numeracy tutor used by hundreds of children in Tanzania.

## Parameter-based Value Functions

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Off-Policy Reinforcement Learning']

Traditional off-policy actor-critic Reinforcement Learning (RL) algorithms learn value functions of a single target policy. However, when value functions are updated to track the learned policy, they forget potentially useful information about old policies. We introduce a class of value functions called Parameter-based Value Functions (PVFs) whose inputs include the policy parameters. They can generalize across different policies. PVFs can evaluate the performance of any policy given a state, a state-action pair, or a distribution over the RL agent's initial states. First we show how PVFs yield novel off-policy policy gradient theorems. Then we derive off-policy actor-critic algorithms based on PVFs trained by Monte Carlo or Temporal Difference methods. We show how learned PVFs can zero-shot learn new policies that outperform any policy seen during training. Finally our algorithms are evaluated on a selection of discrete and continuous control tasks using shallow policies and deep neural networks. Their performance is comparable to the one of state-of-the-art methods.

## Deep Coherent Exploration For Continuous Control

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'exploration', 'latent variable models']

In policy search methods for reinforcement learning (RL), exploration is often performed by injecting noise either in action space at each step independently or in parameter space over each full trajectory. In prior work, it has been shown that with linear policies, a more balanced trade-off between these two exploration strategies is beneficial. However, that method did not scale to policies using deep neural networks. In this paper, we introduce Deep Coherent Exploration, a general and scalable exploration framework for deep RL algorithms on continuous control, that generalizes step-based and trajectory-based exploration. This framework models the last layer parameters of the policy network as latent variables and uses a recursive inference step within the policy update to handle these latent variables in a scalable manner. We find that Deep Coherent Exploration improves the speed and stability of learning of A2C, PPO, and SAC on several continuous control tasks.

## Towards Finding Longer Proofs

Authors: ['Anonymous']

Keywords: ['automated reasoning', 'reinforcement learning', 'reasoning by analogy']

We present a reinforcement learning (RL) based guidance system for automated theorem proving geared towards Finding Longer Proofs (FLoP). FLoP is a step towards learning to reason by analogy, reducing the dependence on large scale search in automated theorem provers. We use several simple, structured datasets with very long proofs to show that FLoP can successfully generalise a single training proof to a large class of related problems, implementing a simple form of analogical reasoning. On these benchmarks, FLoP is competitive with strong theorem provers despite using very limited search.

## Trust, but verify: model-based exploration in sparse reward environments

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'model-based', 'exploration', 'on-line planning', 'imperfect environment model']

We propose $\textit{trust-but-verify}$ (TBV) mechanism, a new method which uses model uncertainty estimates to guide  exploration. The mechanism augments graph search planning algorithms by the capacity to deal with learned model's imperfections. We identify certain type of frequent model errors, which we dub $\textit{false loops}$, and which are particularly dangerous for graph search algorithms in discrete environments. These errors impose falsely pessimistic expectations and thus hinder exploration. We confirm this experimentally and show that TBV can effectively alleviate them. TBV combined with MCTS or Best First Search forms an effective model-based reinforcement learning solution, which is able to robustly solve sparse reward problems. 

## XLVIN: eXecuted Latent Value Iteration Nets

Authors: ['Anonymous']

Keywords: ['value iteration', 'graph neural networks', 'reinforcement learning']

Value Iteration Networks (VINs) have emerged as a popular method to perform implicit planning within deep reinforcement learning, enabling performance improvements on tasks requiring long-range reasoning and understanding of environment dynamics. This came with several limitations, however: the model is not explicitly incentivised to perform meaningful planning computations, the underlying state space is assumed to be discrete, and the Markov decision process (MDP) is assumed fixed and known. We propose eXecuted Latent Value Iteration Networks (XLVINs), which combine recent developments across contrastive self-supervised learning, graph representation learning and neural algorithmic reasoning to alleviate all of the above limitations, successfully deploying VIN-style models on generic environments. XLVINs match the performance of VIN-like models when the underlying MDP is discrete, fixed and known, and provide significant improvements to model-free baselines across three general MDP setups.

## PAC-Bayesian Randomized Value Function with Informative Prior

Authors: ['Anonymous']

Keywords: ['Reinforcement learning']

Randomized value function has been shown as an effective exploration strategy for reinforcement learning (RL), which samples from a learned estimation of the distribution over the randomized Q-value function and then selects the optimal action. However, value function methods are known to suffer from value estimation error. Overfitting of value function is one of the main reasons to estimation error. To address this, in this paper, we propose a Bayesian linear regression with informative prior (IP-BLR) operator to leverage the data-dependent prior in the learning process of randomized value function, which can leverage the statistics of training results from previous iterations. We theoretically derive a generalization error bound for the proposed IP-BLR operation at each learning iteration based on PAC-Bayesian theory, showing a trade-off between the distribution obtained by IP-BLR and the informative prior. Since the optimal posterior that minimizes this generalization error bound is intractable, we alternatively develop an adaptive noise parameter update algorithm to balance this trade-off. The performance of the proposed IP-BLR deep Q-network (DQN) with adaptive noise parameter update is validated through some classical control tasks. It demonstrates that compared to existing methods using non-informative prior, the proposed IP-BLR DQN can achieve higher accumulated rewards in fewer interactions with the environment, due to the capabilities of more accurate value function approximation and better generalization.

## Non-Markovian Predictive Coding For Planning In Latent Space

Authors: ['Anonymous']

Keywords: ['representation learning', 'reinforcement learning', 'information theory']

High-dimensional observations are a major challenge in the application of model-based reinforcement learning (MBRL) to real-world environments. In order to handle high-dimensional sensory inputs, existing MBRL approaches use representation learning to map high-dimensional observations into a lower-dimensional latent space that is more amenable to dynamics estimation and planning. Crucially, the task-relevance and predictability of the learned representations play critical roles in the success of planning in latent space. In this work, we present Non-Markovian Predictive Coding (NMPC), an information-theoretic approach for planning from high-dimensional observations with two key properties: 1) it formulates a mutual information objective that prioritizes the encoding of task-relevant components of the environment; and 2) it employs a recurrent neural network capable of modeling non-Markovian latent dynamics. To demonstrate NMPC’s ability to prioritize task-relevant information, we evaluate our new model on a challenging modification of standard DMControl tasks where the DMControl background is replaced with natural videos, containing complex but irrelevant information to the planning task. Our experiments show that NMPC is superior to existing methods in the challenging complex-background setting while remaining competitive with current state-of-the-art MBRL models in the standard setting.

## Factored Action Spaces in Deep Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement Learning', 'Large action spaces', 'Parameterized action spaces', 'Multi-Agent', 'Continuous Control']

Very large action spaces constitute a critical challenge for deep Reinforcement Learning (RL) algorithms. An existing approach consists in splitting the action space into smaller components and choosing either independently or sequentially actions in each dimension. This approach led to astonishing results for the StarCraft and Dota 2 games, however it remains underexploited and understudied. In this paper, we name this approach Factored Actions Reinforcement Learning (FARL) and study both its theoretical impact and practical use. Notably, we provide a theoretical analysis of FARL on the Proximal Policy Optimization (PPO) and Soft Actor Critic (SAC) algorithms and evaluate these agents in different classes of problems. We show that FARL is a very versatile and efficient approach to combinatorial and continuous control problems.

## Model-based micro-data reinforcement learning: what are the crucial model properties and which model to choose?

Authors: ['Anonymous']

Keywords: ['model-based reinforcement learning', 'generative models', 'mixture density nets', 'dynamic systems', 'heteroscedasticity']

We contribute to model-based micro-data reinforcement learning (MBRL) by rigorously comparing popular generative models using a fixed (random shooting) control agent. We find that on an environment that requires multimodal posterior predictives, mixture density nets outperform all other models by a large margin. When multimodality is not required, our surprising finding is that we do not need probabilistic posterior predictives: deterministic models may perform optimally but only if they are trained with a probabilistic goal, allowing heteroscedasticity at training time. Our hypothesis is that heteroscedasticity somehow alleviates long-term error accumulation which often hinders the performance of MBRL. At the methodological side, we design metrics and an experimental protocol which can be used to evaluate the various models, predicting their asymptotic performance when using them on the control problem. Using this framework, we improve the state-of-the-art sample complexity of MBRL on Acrobot by two to four folds, using an aggressive training schedule which is outside of the hyperparameter interval usually considered.

## Combining Imitation and Reinforcement Learning with Free Energy Principle

Authors: ['Anonymous']

Keywords: ['Imitation', 'Reinforcement Learning', 'Free Energy Principle']

Imitation Learning (IL) and Reinforcement Learning (RL) from high dimensional sensory inputs are often introduced as separate problems, but a more realistic problem setting is how to merge the techniques so that the agent can reduce exploration costs by partially imitating experts at the same time it maximizes its return. Even when the experts are suboptimal (e.g. Experts learned halfway with other RL methods or human-crafted experts), it is expected that the agent outperforms the suboptimal experts’ performance. In this paper, we propose to address the issue by using and theoretically extending Free Energy Principle, a unified brain theory that explains perception, action and model learning in a Bayesian probabilistic way. Our results show that our approach achieves at least equivalent or better performance than standard IL or RL in visual control tasks with sparse rewards.

## Ordering-Based Causal Discovery with Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Causal Discovery', 'Reinforcement Learning', 'Ordering Search']

It is a long-standing question to discover causal relations among a set of variables in many empirical sciences. Recently, Reinforcement Learning (RL) has achieved promising results in causal discovery. However, searching the space of directed graphs directly and enforcing acyclicity by implicit penalties tend to be inefficient and restrict the method to the small problems. In this work, we alternatively consider searching an ordering by RL from the variable ordering space that is much smaller than that of directed graphs, which also helps avoid dealing with acyclicity. Specifically, we formulate the ordering search problem as a Markov decision process, and then use different reward designs to optimize the ordering generating model. A generated ordering is then processed using variable selection methods to obtain the final directed acyclic graph. In contrast to other causal discovery methods, our method can also utilize a pretrained model to accelerate training. We conduct experiments on both synthetic and real-world datasets, and show that the proposed method outperforms other baselines on important metrics even on large graph tasks.

## Explicit Pareto Front Optimization for Constrained Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['constrained reinforcement learning', 'multi-objective reinforcement learning', 'continuous control', 'deep reinforcement learning']

Many real-world problems require that reinforcement learning (RL) agents learn policies that not only maximize a scalar reward, but do so while meeting constraints, such as remaining below an energy consumption threshold. Typical approaches for solving constrained RL problems rely on Lagrangian relaxation, but these suffer from several limitations. We draw a connection between multi-objective RL and constrained RL, based on the key insight that the constraint-satisfying optimal policy must be Pareto optimal. This leads to a novel, multi-objective perspective for constrained RL. We propose a framework that uses a multi-objective RL algorithm to find a Pareto front of policies that trades off between the reward and constraint(s), and simultaneously searches along this front for constraint-satisfying policies. We show that in practice, an instantiation of our framework outperforms existing approaches on several challenging continuous control domains, both in terms of solution quality and sample efficiency, and enables flexibility in recovering a portion of the Pareto front rather than a single constraint-satisfying policy.

## Robust Constrained Reinforcement Learning for Continuous Control with Model Misspecification

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'constraints', 'robustness']

Many real-world physical control systems are required to satisfy constraints upon deployment. Furthermore, real-world systems are often subject to effects such as non-stationarity, wear-and-tear, uncalibrated sensors and so on. Such effects effectively perturb the system dynamics and can cause a policy trained successfully in one domain to perform poorly when deployed to a perturbed version of the same domain. This can affect a policy's ability to maximize future rewards as well as the extent to which it satisfies constraints. We refer to this as constrained model misspecification. We present an algorithm with theoretical guarantees that mitigates this form of misspecification, and showcase its performance in multiple Mujoco tasks from the Real World Reinforcement Learning (RWRL) suite.

## Transient Non-stationarity and Generalisation in Deep Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Generalization']

Non-stationarity can arise in Reinforcement Learning (RL) even in stationary environments. For example, most RL algorithms collect new data throughout training, using a non-stationary behaviour policy. Due to the transience of this non-stationarity, it is often not explicitly addressed in deep RL and a single neural network is continually updated. However, we find evidence that neural networks exhibit a memory effect, where these transient non-stationarities can permanently impact the latent representation and adversely affect generalisation performance. Consequently, to improve generalisation of deep RL agents, we propose Iterated Relearning (ITER). ITER augments standard RL training by repeated knowledge transfer of the current policy into a freshly initialised network, which thereby experiences less non-stationarity during training. Experimentally, we show that ITER improves performance on the challenging generalisation benchmarks ProcGen and Multiroom.

## Universal Value Density Estimation for Imitation Learning and Goal-Conditioned Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Imitation Learning', 'Reinforcement Learning', 'Universal Value Functions']

This work considers two distinct settings: imitation learning and goal-conditioned reinforcement learning. In either case, effective solutions require the agent to reliably reach a specified state (a goal), or set of states (a demonstration). Drawing a connection between probabilistic long-term dynamics and the desired value function, this work introduces an approach that utilizes recent advances in density estimation to effectively learn to reach a given state. We develop a unified view on the two settings and show that the approach can be applied to both. In goal-conditioned reinforcement learning, we show it to circumvent the problem of sparse rewards while addressing hindsight bias in stochastic domains. In imitation learning, we show that the approach can learn from extremely sparse amounts of expert data and achieves state-of-the-art results on a common benchmark.

## Coverage as a Principle for Discovering Transferable Behavior in Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['deep reinforcement learning', 'transfer learning', 'unsupervised learning', 'exploration']

Designing agents that acquire knowledge autonomously and use it to solve new tasks efficiently is an important challenge in reinforcement learning. Unsupervised learning provides a useful paradigm for autonomous acquisition of task-agnostic knowledge. In supervised settings, representations discovered through unsupervised pre-training offer important benefits when transferred to downstream tasks. Given the nature of the reinforcement learning problem, we explore how to transfer knowledge through behavior instead of representations. The behavior of pre-trained policies may be used for solving the task at hand (exploitation), as well as for collecting useful data to solve the problem (exploration). We argue that pre-training policies to maximize coverage will result in behavior that is useful for both strategies. When using these policies for both exploitation and exploration, our agents discover solutions that lead to larger returns. The largest gains are generally observed in domains requiring structured exploration, including settings where the behavior of the pre-trained policies is misaligned with the downstream task.

## Balancing Constraints and Rewards with Meta-Gradient D4PG

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'meta-gradients', 'constraints']

Deploying Reinforcement Learning (RL) agents to solve real-world applications often requires satisfying complex system constraints. Often the constraint thresholds are incorrectly set due to the complex nature of a system or the inability to verify the thresholds offline (e.g, no simulator or reasonable offline evaluation procedure exists). This results in solutions where a task cannot be solved without violating the constraints. However, in many real-world cases, constraint violations are undesirable yet they are not catastrophic, motivating the need for soft-constrained RL approaches. We present two soft-constrained RL approaches that utilize meta-gradients to find a good trade-off between expected return and minimizing constraint violations. We demonstrate the effectiveness of these approaches by showing that they consistently outperform the baselines across four different Mujoco domains.

## Decentralized Deterministic Multi-Agent Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['multiagent reinforcement learning', 'MARL', 'decentralized actor-critic algorithm']

Recent work in multi-agent reinforcement learning (MARL) by [Zhang, ICML12018] provided the first decentralized actor-critic algorithm to offer convergence guarantees. In that work, policies are stochastic and are defined on finite action spaces. We extend those results to develop a provably-convergent decentralized actor-critic algorithm for learning deterministic policies on continuous action spaces. Deterministic policies are important in many real-world settings. To handle the lack of exploration inherent in deterministic policies we provide results for the off-policy setting as well as the on-policy setting. We provide the main ingredients needed for this problem: the expression of a local deterministic policy gradient, a decentralized deterministic actor-critic algorithm, and convergence guarantees when the value functions are approximated linearly. This work enables decentralized MARL in high-dimensional action spaces and paves the way for more widespread application of MARL.

## Solving NP-Hard Problems on Graphs with Extended AlphaGo Zero

Authors: ['Anonymous']

Keywords: ['Graph neural network', 'Combinatorial optimization', 'Reinforcement learning']

There have been increasing challenges to solve combinatorial optimization problems by machine learning. 
Khalil et al. (NeurIPS 2017) proposed an end-to-end reinforcement learning framework, which automatically learns graph embeddings to construct solutions to a wide range of problems.
However, it sometimes performs poorly on graphs having different characteristics than training graphs.
To improve its generalization ability to various graphs, we propose a novel learning strategy based on AlphaGo Zero, a Go engine that achieved a superhuman level without the domain knowledge of the game.
We redesign AlphaGo Zero for combinatorial optimization problems, taking into account several differences from two-player games.
In experiments on five NP-hard problems such as {\sc MinimumVertexCover} and {\sc MaxCut}, our method, with only a policy network, shows better generalization than the previous method to various instances that are not used for training, including random graphs, synthetic graphs, and real-world graphs.
Furthermore, our method is significantly enhanced by a test-time Monte Carlo Tree Search which makes full use of the policy network and value network.
We also compare recently-developed graph neural network (GNN) models, with an interesting insight into a suitable choice of GNN models for each task.

## Q-Value Weighted Regression: Reinforcement Learning with Limited Data

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'rl', 'offline rl', 'continuous control', 'atari', 'sample efficiency']

Sample efficiency and performance in the offline setting have emerged as among the main
challenges of deep reinforcement learning. We introduce Q-Value Weighted Regression (QWR),
a simple RL algorithm that excels in these aspects.

QWR is an extension of Advantage Weighted Regression (AWR), an off-policy actor-critic algorithm
that performs very well on continuous control tasks, also in the offline setting, but struggles
on tasks with discrete actions and in sample efficiency. We perform a theoretical analysis
of AWR that explains its shortcomings and use the insights to motivate QWR theoretically.

We show experimentally that QWR matches state-of-the-art algorithms both on tasks with
continuous and discrete actions. We study the main hyperparameters of QWR
and find that it is stable in a wide range of their choices and on different tasks.
In particular, QWR yields results on par with SAC on the MuJoCo suite and - with
the same set of hyperparameters -- yields results on par with a highly tuned Rainbow
implementation on a set of Atari games. We also verify that QWR performs well in the
offline RL setting, making it a compelling choice for reinforcement learning in domains
with limited data.

## Learning a Transferable Scheduling Policy for Various Vehicle Routing Problems based on Graph-centric Representation Learning

Authors: ['Anonymous']

Keywords: ['Vehicle Routing Problem', 'Multiple Traveling Salesmen Problem', 'Capacitated Vehicle Routing Problem', 'Reinforcement Learning', 'Graph Neural Network']

Reinforcement learning has been used to learn to solve various routing problems. however, most of the algorithm is restricted to finding an optimal routing strategy for only a single vehicle. In addition, the trained policy under a specific target routing problem is not able to solve different types of routing problems with different objectives and constraints. This paper proposes an reinforcement learning approach to solve the min-max capacitated multi vehicle routing problem (mCVRP), the problem seeks to minimize the total completion time for multiple vehicles whose one-time traveling distance is constrained by their fuel levels to serve the geographically distributed customer nodes. The method represents the relationships among vehicles, customers, and fuel stations using relationship-specific graphs to consider their topological relationships and employ graph neural network (GNN) to extract the graph's embedding to be used to make a routing action. We train the proposed model using the random mCVRP instance with different numbers of vehicles, customers, and refueling stations. We then validate that the trained policy solve not only new mCVRP problems with different complexity (weak transferability but also different routing problems (CVRP, mTSP, TSP) with different objectives and constraints (storing transferability). 

## Plan-Based Asymptotically Equivalent Reward Shaping

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'reward shaping', 'plan-based reward shaping', 'robotics', 'robotic manipulation']

In high-dimensional state spaces, the usefulness of Reinforcement Learning (RL) is limited by the problem of exploration. This issue has been addressed using potential-based reward shaping (PB-RS) previously. In the present work, we introduce Asymptotically Equivalent Reward Shaping (ASEQ-RS). ASEQ-RS relaxes the strict optimality guarantees of PB-RS to a guarantee of asymptotic equivalence. Being less restrictive, ASEQ-RS allows for reward shaping functions that are even better suited for improving the sample efficiency of RL algorithms. In particular, we consider settings in which the agent has access to an approximate plan. Here, we use examples of simulated robotic manipulation tasks to demonstrate that plan-based ASEQ-RS can indeed significantly improve the sample efficiency of RL over plan-based PB-RS.

## Sample efficient Quality Diversity for neural continuous control

Authors: ['Anonymous']

Keywords: ['Deep Neuroevolution', 'Quality Diversity', 'Reinforcement Learning']

We propose a novel Deep Neuroevolution algorithm, QD-RL, that combines the strengths of off-policy reinforcement learning (RL) algorithms and Quality Diversity (QD) approaches to solve continuous control problems with neural controllers. The QD part contributes structural biases by decoupling the search for diversity from the search for high return, resulting in efficient management of the exploration-exploitation trade-off. The RL part contributes sample efficiency by relying on off-policy gradient-based updates of the agents. More precisely, we train a population of off-policy deep RL agents to simultaneously maximize diversity within the population and the return of each individual agent. QD-RL selects agents interchangeably from a Pareto front or from a Map-Elites grid, resulting in stable and efficient population updates. Our experiments on the AntMaze and AntTrap environments show that QD-RL can solve challenging exploration and control problems with deceptive rewards while being over 15 times more sample efficient than its evolutionary counterparts.

## Variational Intrinsic Control Revisited

Authors: ['Anonymous']

Keywords: ['Unsupervised reinforcement learning', 'Information theory']

In this paper, we revisit variational intrinsic control (VIC), an unsupervised reinforcement learning method for finding the largest set of intrinsic options available to an agent. In the original work by Gregor et al. (2016), two VIC algorithms were proposed: one that represents the options explicitly, and the other that does it implicitly. We show that the intrinsic reward used in the latter is subject to bias in stochastic environments, causing convergence to suboptimal solutions. To correct this behavior, we propose two methods respectively based on the transitional probability model and Gaussian Mixture Model. We substantiate our claims through rigorous mathematical derivations and experimental analyses. 

## Winning the L2RPN Challenge: Power Grid Management via Semi-Markov Afterstate Actor-Critic

Authors: ['Anonymous']

Keywords: ['power grid management', 'deep reinforcement learning', 'graph neural network']

Safe and reliable electricity transmission in power grids is crucial for modern society. It is thus quite natural that there has been a growing interest in the automatic management of power grids, exempliﬁed by the Learning to Run a Power Network Challenge (L2RPN), modeling the problem as a reinforcement learning (RL) task. However, it is highly challenging to manage a real-world scale power grid, mostly due to the massive scale of its state and action space. In this paper, we present an off-policy actor-critic approach that effectively tackles the unique challenges in power grid management by RL, adopting the hierarchical policy together with the afterstate representation. Our agent ranked ﬁrst in the latest challenge (L2RPN WCCI 2020), being able to avoid disastrous situations while maintaining the highest level of operational efﬁciency in every test scenarios. This paper provides a formal description of the algorithmic aspect of our approach, as well as further experimental studies on diverse power grids.

## Truly Deterministic Policy Optimization

Authors: ['Anonymous']

Keywords: ['Deterministic Policy Gradient', 'Deterministic Exploration', 'Reinforcement Learning']

In this paper, we present a policy gradient method that avoids exploratory noise injection and performs policy search over the deterministic landscape. By avoiding noise injection all sources of estimation variance can be eliminated in systems with deterministic dynamics (up to the initial state distribution). Since deterministic policy regularization is impossible using traditional non-metric measures such as the KL divergence, we derive a Wasserstein-based quadratic model for our purposes. We state conditions on the system model under which it is possible to establish a monotonic policy improvement guarantee, propose a surrogate function for policy gradient estimation, and show that it is possible to compute exact advantage estimates if both the state transition model and the policy are deterministic. Finally, we describe two novel robotic control environments---one with non-local rewards in the frequency domain and the other with a long horizon (8000 time-steps)---for which our policy gradient method (TDPO) significantly outperforms existing methods (PPO, TRPO, DDPG, and TD3).

## Learning not to learn: Nature versus nurture in silico

Authors: ['Anonymous']

Keywords: ['Meta-Learning', 'Reinforcement Learning']

Animals are equipped with a rich innate repertoire of sensory, behavioral and motor skills, which allows them to interact with the world immediately after birth. At the same time, many behaviors are highly adaptive and can be tailored to specific environments by means of learning. In this work, we use mathematical analysis and the framework of meta-learning (or 'learning to learn') to answer when it is beneficial to learn such an adaptive strategy and when to hard-code a heuristic behavior. We find that the interplay of ecological uncertainty, task complexity and the agents' lifetime has crucial effects on the meta-learned amortized Bayesian inference performed by an agent. There exist two regimes: One in which meta-learning yields a learning algorithm that implements task-dependent information-integration and a second regime in which meta-learning imprints a heuristic or 'hard-coded' behavior. Further analysis reveals that non-adaptive behaviors are not only optimal for aspects of the environment that are stable across individuals, but also in situations where an adaptation to the environment would in fact be highly beneficial, but could not be done quickly enough to be exploited within the remaining lifetime. Hard-coded behaviors should hence not only be those that always work, but also those that are too complex to be learned within a reasonable time frame.

## Genetic Soft Updates for Policy Evolution in Deep Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement Learning', 'Machine Learning for Robotics']

The combination of Evolutionary Strategies (ES) and Deep Reinforcement Learning (DRL) has been recently proposed to merge the benefits of both solutions. Existing mixed approaches, however, have been successfully applied only to actor-critic methods and present significant overhead. We address these issues by introducing a novel mixed framework that exploits a periodical genetic evaluation to soft update the weights of a DRL agent. The resulting approach is applicable with any DRL method and, in a worst-case scenario, it does not exhibit detrimental behaviours. Experiments in robotic applications and continuous control benchmarks demonstrate the versatility of our approach that significantly outperforms prior DRL, ES, and mixed approaches. Finally, we employ formal verification to confirm the policy improvement, mitigating the inefficient exploration and hyper-parameter sensitivity of DRL.

## Simple Augmentation Goes a Long Way: ADRL for DNN Quantization

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Quantization', 'mixed precision', 'augmented deep reinforcement learning', 'DNN']

Mixed precision quantization improves DNN performance by assigning different layers with different bit-width values. Searching for the optimal bit-width for each layer, however, remains a challenge. Deep Reinforcement Learning (DRL) shows some recent promise. It however suffers instability due to function approximation errors, causing large variances in the early training stages, slow convergence, and suboptimal policies in the mixed-precision quantization problem. This paper proposes augmented DRL (ADRL) as a way to alleviate these issues. This new strategy augments the neural networks in DRL with a complementary scheme to boost the performance of learning. The paper examines the effectiveness of ADRL both analytically and empirically, showing that it can produce more accurate quantized models than the state of the art DRL-based quantization while improving the learning speed by 4.5-64 times. 

## Monte-Carlo Planning and Learning with Language Action Value Estimates

Authors: ['Anonymous']

Keywords: ['natural language processing', 'Monte-Carlo tree search', 'reinforcement learning', 'interactive fiction']

Interactive Fiction (IF) games provide a useful testbed for language-based reinforcement learning agents, posing significant challenges of natural language understanding, commonsense reasoning, and non-myopic planning in the combinatorial search space. Agents based on standard planning algorithms struggle to play IF games due to the massive search space of language actions. Thus, language-grounded planning is a key ability of such agents, since inferring the consequence of language action based on semantic understanding can drastically improve search. In this paper, we introduce Monte-Carlo planning with Language Action Value Estimates (MC-LAVE) that combines a Monte-Carlo tree search with language-driven exploration. MC-LAVE invests more search effort into semantically promising language actions using locally optimistic language value estimates, yielding a significant reduction in the effective search space of language actions. We then present a reinforcement learning approach via MC-LAVE, which alternates between MC-LAVE planning and supervised learning of the self-generated language actions. In the experiments, we demonstrate that our method significantly outperforms the state-of-the-art on the extensive set of IF games.

## Addressing Distribution Shift in Online Reinforcement Learning with Offline Datasets

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'offline reinforcement learning', 'control', 'distribution shift']

Recent progress in offline reinforcement learning (RL) has made it possible to train strong RL agents from previously-collected, static datasets. However, depending on the quality of the trained agents and the application being considered, it is often desirable to improve such offline RL agents with further online interaction. As it turns out, fine-tuning offline RL agents is a non-trivial challenge, due to distribution shift – the agent encounters out-of-distribution samples during online interaction, which may cause bootstrapping error in Q-learning and instability during fine-tuning. In order to address the issue, we present a simple yet effective framework, which incorporates a balanced replay scheme and an ensemble distillation scheme. First, we propose to keep separate offline and online replay buffers, and carefully balance the number of samples from each buffer during updates. By utilizing samples from a wider distribution, i.e., both online and offline samples, we stabilize the Q-learning. Next, we present an ensemble distillation scheme, where we train an ensemble of independent actor-critic agents, then distill the policies into a single policy. In turn, we improve the policy using the Q-ensemble during fine-tuning, which allows the policy updates to be more robust to error in each individual Q-function. We demonstrate the superiority of our method on MuJoCo datasets from the recently proposed D4RL benchmark suite.


## Unsupervised Active Pre-Training for Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Unsupervised Learning', 'Entropy Maximization', 'Contrastive Learning']

We introduce a new unsupervised pre-training method for reinforcement learning called $\textbf{APT}$, which stands for $\textbf{A}\text{ctive}\textbf{P}\text{re-}\textbf{T}\text{raining}$. APT learns a representation and a policy initialization by actively searching for novel states in reward-free environments. We use the contrastive learning framework for learning the representation from collected transitions. The key novel idea is to collect data during pre-training by maximizing a particle based entropy computed in the learned latent representation space. By doing particle based entropy maximization, we alleviate the need for challenging density modeling and are thus able to scale our approach to image observations. APT successfully learns meaningful representations as well as policy initializations without using any reward. APT is conceptually simple to implement, scalable, and empirically powerful. We empirically evaluate APT on the Atari game suite and DMControl suite by exposing task-specific reward to agent after a long unsupervised pre-training phase. On DMControl suite, APT beats all baselines in terms of asymptotic performance and data efficiency and dramatically improves performance on tasks that are extremely difficult for training from scratch. On Atari games, APT achieves human-level performance on $12$ games and obtains highly competitive performance compared to canonical RL algorithms.

## Bounded Myopic Adversaries for Deep Reinforcement Learning Agents

Authors: ['Anonymous']

Keywords: ['deep reinforcement learning', 'adversarial']

Adversarial attacks against deep neural networks have been widely studied. Adversarial examples for deep reinforcement learning (DeepRL) have significant security implications, due to the deployment of these algorithms in many application domains. In this work we formalize an optimal myopic adversary for deep reinforcement learning agents. Our adversary attempts to find a bounded perturbation of the state which minimizes the value of the action taken by the agent. We show with experiments in various games in the Atari environment that our attack formulation achieves significantly larger impact as compared to the current  state-of-the-art. Furthermore, this enables us to lower the bounds by several orders of magnitude on the perturbation needed to efficiently achieve significant impacts on DeepRL agents.

## Contrastive Explanations for Reinforcement Learning via Embedded Self Predictions

Authors: ['Anonymous']

Keywords: ['Explainable AI', 'Deep Reinforcement Learning']

We investigate a deep reinforcement learning (RL) architecture that supports explaining why a learned agent prefers one action over another. The key idea is to learn action-values that are directly represented via human-understandable properties of expected futures. This is realized via the embedded self-prediction (ESP) model, which learns said properties in terms of human provided features. Action preferences can then be explained by contrasting the future properties predicted for each action. To address cases where there are a large number of features, we develop a novel method for computing minimal sufficient explanations from an ESP. Our case studies in three domains, including a complex strategy game, show that ESP models can be effectively learned and support insightful explanations. 

## Explore with Dynamic Map: Graph Structured Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement Learning', 'Graph Structured Reinforcement Learning', 'Exploration']

In reinforcement learning, a map with states and transitions built based on historical trajectories is often helpful in exploration and exploitation. Even so, learning and planning on such a map within a sparse environment remains a challenge. As a step towards this goal, we propose Graph Structured Reinforcement Learning (GSRL), which utilizes historical trajectories to slowly adjust exploration directions and learn related experiences while rapidly updating the value function estimation. GSRL constructs a dynamic graph on top of state transitions in the replay buffer based on historical trajectories, and develops an attention strategy on the map to select an appropriate goal direction, which decomposes the task of reaching a distant goal state into a sequence of easier tasks. We also leverage graph structure to sample related trajectories for efficient value learning. Results demonstrate that GSRL can outperform the state-of-the-art algorithms in terms of sample efficiency on benchmarks with sparse reward functions. 

## RECONNAISSANCE FOR REINFORCEMENT LEARNING WITH SAFETY CONSTRAINTS

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Safety constraints', 'Constrained Markov Decision Process']

Practical reinforcement learning problems are often formulated as constrained Markov decision process (CMDP) problems, in which the agent has to maximize the expected return while satisfying a set of prescribed safety constraints. In this study, we consider a situation in which the agent has access to the generative model which provides us with a next state sample for any given state-action pair, and propose a model to solve a CMDP problem by decomposing the CMDP into a pair of MDPs; \textit{reconnaissance} MDP (R-MDP) and \textit{planning} MDP (P-MDP). In R-MDP, we train threat function, the Q-function analogue of danger that can determine whether a given state-action pair is safe or not. In P-MDP, we train a reward-seeking policy while using a fixed threat function to determine the safeness of each action. With the help of generative model, we can efficiently train the threat function by preferentially sampling rare dangerous events. Once the threat function for a baseline policy is computed, we can solve other CMDP problems with different reward and different danger-constraint without the need to re-train the model. We also present an efficient approximation method for the threat function that can greatly reduce the difficulty of solving R-MDP. We will demonstrate the efficacy of our method over classical approaches in benchmark dataset and complex collision-free navigation tasks.

## QTRAN++: Improved Value Transformation for Cooperative Multi-Agent Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['multi-agent reinforcement learning']

QTRAN is a multi-agent reinforcement learning (MARL) algorithm capable of learning the largest class of joint-action value functions up to date. However, despite its strong theoretical guarantee, it has shown poor empirical performance in complex environments, such as Starcraft Multi-Agent Challenge (SMAC). In this paper, we identify the performance bottleneck of QTRAN and propose a substantially improved version, coined QTRAN++. Our gains come from (i) stabilizing the training objective of QTRAN, (ii) removing the strict role separation between the action-value estimators of QTRAN, and (iii) introducing a multi-head mixing network for value transformation. Through extensive evaluation, we confirm that our diagnosis is correct, and QTRAN++ successfully bridges the gap between empirical performance and theoretical guarantee. In particular, QTRAN++ newly achieves state-of-the-art performance in the SMAC environment. The code will be released.

## Offline Meta-Reinforcement Learning with Advantage Weighting

Authors: ['Anonymous']

Keywords: ['offline', 'meta-reinforcement learning', 'meta-learning', 'reinforcement learning', 'maml']

This paper introduces the offline meta-reinforcement learning (offline meta-RL) problem setting and proposes an algorithm that performs well in this setting. Offline meta-RL is analogous to the widely successful supervised learning strategy of pre-training a model on a large batch of fixed, pre-collected data (possibly from various tasks) and fine-tuning the model to a new task with relatively little data. That is, in offline meta-RL, we meta-train on fixed, pre-collected data from several tasks and adapt to a new task with a very small amount (less than 5 trajectories) of data from the new task. By nature of being offline, algorithms for offline meta-RL can utilize the largest possible pool of training data available and eliminate potentially unsafe or costly data collection during meta-training. This setting inherits the challenges of offline RL, but it differs significantly because offline RL does not generally consider a) transfer to new tasks or b) limited data from the test task, both of which we face in offline meta-RL. Targeting the offline meta-RL setting, we propose Meta-Actor Critic with Advantage Weighting (MACAW). MACAW is an optimization-based meta-learning algorithm that uses simple, supervised regression objectives for both the inner and outer loop of meta-training. On offline variants of common meta-RL benchmarks, we empirically find that this approach enables fully offline meta-reinforcement learning and achieves notable gains over prior methods.

## PODS: Policy Optimization via Differentiable Simulation

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Decision and Control', 'Planning', 'Robotics.']

Current reinforcement learning (RL) methods use simulation models as simple black-box oracles. In this paper, with the goal of improving the performance exhibited by RL algorithms, we explore a systematic way of leveraging the additional information provided by an emerging class of differentiable simulators. Building on concepts established by Deterministic Policy Gradients (DPG) methods, the neural network policies learned with our approach represent deterministic actions. In a departure from standard methodologies, however, learning these policy does not hinge on approximations of the value function that must be learned concurrently in an actor-critic fashion. Instead, we exploit differentiable simulators to directly compute the analytic gradient of a policy's value function with respect to the actions it outputs. This, in turn, allows us to efficiently perform locally optimal policy improvement iterations. Compared against other state-of-the-art RL methods, we show that with minimal hyper-parameter tuning our approach consistently leads to better asymptotic behavior across a set of payload manipulation tasks that demand high precision.

## Regioned Episodic Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement Learning', 'Episodic Memory', 'Sample Efficiency']

Goal-oriented reinforcement learning algorithms are often good at exploration, not exploitation, while episodic algorithms excel at exploitation, not exploration. As a result, neither of these approaches alone can lead to a sample efficient algorithm in complex environments with high dimensional state space and delayed rewards. Motivated by these observations and shortcomings, in this paper, we introduce Regioned Episodic Reinforcement Learning (RERL) that combines the episodic and goal-oriented learning strengths and leads to a more sample efficient and ef- fective algorithm. RERL achieves this by decomposing the space into several sub-space regions and constructing regions that lead to more effective exploration and high values trajectories. Extensive experiments on various benchmark tasks show that RERL outperforms existing methods in terms of sample efficiency and final rewards.

## Learning to Solve Multi-Robot Task Allocation with a Covariant-Attention based Neural Architecture

Authors: ['Anonymous']

Keywords: ['Graph neural network', 'Attention mechanism', 'Reinforcement learning', 'Multi-robotic task allocation']

This paper presents a novel graph (reinforcement) learning method to solve an important class of multi-robot task allocation (MRTA) problems that involve tasks with deadlines, and robots with ferry range and payload constraints (thus requiring multiple tours per robot). While drawing motivation from recent graph learning methods that learn to solve combinatorial optimization problems of the mTSP/VRP type, this paper seeks to provide better convergence and tractable scalability (with an increasing number of tasks) in learning to solve MRTA. The proposed neural architecture, called Covariant Attention-based Model or CAM, includes two main components: 1) an encoder: a covariant node-based embedding model to represent each task as a learnable feature vector, and 2) a decoder: an attention-based model to facilitate a sequential output. In order to learn the feature vectors, a policy-gradient method based on REINFORCE is used. The new learning architecture is applied to a flood response problem (representative of the general class of Multi-task Robots, and Single-robot Tasks or MR-ST problems), where multiple unmanned aerial vehicles (UAVs) supply survival kits to victims spread out over an area. To perform a comparative analysis, the well-known attention-based approach (that is designed to solve mTSP/VRP problems) is extended and applied to the stated MRTA class of problems. A comparison is performed in terms of a cost function that considers both task completion rate and cumulative traveled distance. The results show that our primary method is superior not only in terms of the cost function (over various training and unseen test scenarios), but also provide significantly faster convergence and yields learnt policies that can be executed within 6 milliseconds per robot, thereby allowing real-time application.

## Sample-Efficient Automated Deep Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['AutoRL', 'Deep Reinforcement Learning', 'Hyperparameter Optimization', 'Neuroevolution']

Despite significant progress in challenging problems across various domains, applying state-of-the-art deep reinforcement learning (RL) algorithms remains challenging due to their sensitivity to the choice of hyperparameters. This sensitivity can partly be attributed to the non-stationarity of the RL problem, potentially requiring different hyperparameter settings at various stages of the learning process. Additionally, in the RL setting, hyperparameter optimization (HPO) requires a large number of environment interactions, hindering the transfer of the successes in RL to real-world applications. In this work, we tackle the issues of sample-efficient and dynamic HPO in RL. We propose a population-based automated RL (AutoRL) framework to meta-optimize arbitrary off-policy RL algorithms. In this framework, we optimize the hyperparameters and also the neural architecture while simultaneously training the agent. By sharing the collected experience across the population, we substantially increase the sample efficiency of the meta-optimization. We demonstrate the capabilities of our sample-efficient AutoRL approach in a case study with the popular TD3 algorithm in the MuJoCo benchmark suite, where we reduce the number of environment interactions needed for meta-optimization by up to an order of magnitude compared to population-based training.

## Linear Representation Meta-Reinforcement Learning for Instant Adaptation

Authors: ['Anonymous']

Keywords: ['meta reinforcement learning', 'out-of-distribution', 'reinforcement learning']

This paper introduces Fast Linearized Adaptive Policy (FLAP), a new meta-reinforcement learning (meta-RL) method that is able to extrapolate well to out-of-distribution tasks without the need to reuse data from training, and adapt almost instantaneously with the need of only a few samples during testing. FLAP builds upon the idea of learning a shared linear representation of the policy so that when adapting to a new task, it suffices to predict a set of linear weights. A separate adapter network is trained simultaneously with the policy such that during adaptation, we can directly use the adapter network to predict these linear weights instead of updating a meta-policy via gradient descent such as in prior Meta-RL algorithms like  MAML to obtain the new policy. The application of the separate feed-forward network not only speeds up the adaptation run-time significantly, but also generalizes extremely well to very different tasks that prior Meta-RL methods fail to generalize to. Experiments on standard continuous-control meta-RL benchmarks show FLAP presenting significantly stronger performance on out-of-distribution tasks with up to double the average return and up to 8X faster adaptation run-time speeds when compared to prior methods.

## Adaptive Procedural Task Generation for Hard-Exploration Problems

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'curriculum learning', 'task generation']

We introduce Adaptive Procedural Task Generation (APT-Gen), an approach to progressively generate a sequence of tasks as curricula to facilitate reinforcement learning in hard-exploration problems. At the heart of our approach, a task generator learns to create tasks from a parameterized task space via a black-box procedural generation module. To enable curriculum learning in the absence of a direct indicator of learning progress, we propose to train the task generator by balancing the agent's performance in the generated tasks and the similarity to the target tasks. Through adversarial training, the task similarity is adaptively estimated by a task discriminator defined on the agent's experiences, allowing the generated tasks to approximate target tasks of unknown parameterization or outside of the predefined task space. Our experiments on grid world and robotic manipulation task domains show that APT-Gen achieves substantially better performance than various existing baselines by generating suitable tasks of rich variations.

## Meta-Reinforcement Learning Robust to Distributional Shift via Model Identification and Experience Relabeling

Authors: ['Anonymous']

Keywords: ['Meta-Reinforcement Learning', 'Meta Learning', 'Reinforcement Learning']

Reinforcement learning algorithms can acquire policies for complex tasks autonomously. However, the number of samples required to learn a diverse set of skills can be prohibitively large. While meta-reinforcement learning methods have enabled agents to leverage prior experience to adapt quickly to new tasks, their performance depends crucially on how close the new task is to the previously experienced tasks. Current approaches are either not able to extrapolate well, or can do so at the expense of requiring extremely large amounts of data for on-policy meta-training. In this work, we present model identification and experience relabeling (MIER), a meta-reinforcement learning algorithm that is both efficient and extrapolates well when faced with out-of-distribution tasks at test time. Our method is based on a simple insight: we recognize that dynamics models can be adapted efficiently and consistently with off-policy data, more easily than policies and value functions. These dynamics models can then be used to continue training policies and value functions for out-of-distribution tasks without using meta-reinforcement learning at all, by generating synthetic experience for the new task.

## Reinforcement Learning with Bayesian Classifiers: Efficient Skill Learning from Outcome Examples

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Goal Reaching', 'Bayesian Classification', 'Reward Inference']

Exploration in reinforcement learning is, in general, a challenging problem. In this work, we study a more tractable class of reinforcement learning problems defined by data that provides examples of successful outcome states. In this case, the reward function can be obtained automatically by training a classifier to classify states as successful or not. We argue that, with appropriate representation and regularization, such a classifier can guide a reinforcement learning algorithm to an effective solution. However, as we will show, this requires the classifier to make uncertainty-aware predictions that are very difficult with standard deep networks. To address this, we propose a novel mechanism for obtaining calibrated uncertainty based on an amortized technique for computing the normalized maximum likelihood distribution. We show that the resulting algorithm has a number of intriguing connections to both count-based exploration methods and prior algorithms for learning reward functions from data, while being able to guide algorithms towards the specified goal more effectively. We show how using amortized normalized maximum likelihood for reward inference is able to provide effective reward guidance for solving a number of challenging navigation and robotic manipulation tasks which prove difficult for other algorithms.

## Solving Compositional Reinforcement Learning Problems via Task Reduction

Authors: ['Anonymous']

Keywords: ['compositional task', 'sparse reward', 'reinforcement learning', 'task reduction', 'imitation learning']

We propose a novel learning paradigm, Self-Imitation via Reduction (SIR), for solving compositional reinforcement learning problems. SIR is based on two core ideas: task reduction and self-imitation. Task reduction tackles a hard-to-solve task by actively reducing it to an easier task whose solution is known by the RL agent. Once the original hard task is successfully solved by task reduction, the agent naturally obtains a self-generated solution trajectory to imitate. By continuously collecting and imitating such demonstrations, the agent is able to progressively expand the solved subspace in the entire task space. Experiment results show that SIR can significantly accelerate and improve learning on a variety of challenging sparse-reward continuous-control problems with compositional structures.

## Decoupling Exploration and Exploitation for Meta-Reinforcement Learning without Sacrifices

Authors: ['Anonymous']

Keywords: ['meta-reinforcement learning', 'reinforcement learning', 'exploration']

The goal of meta-reinforcement learning (meta-RL) is to build agents that can quickly learn new tasks by leveraging prior experience on related tasks. Learning a new task often requires both exploring to gather task-relevant information and exploiting this information to solve the task. In principle, optimal exploration and exploitation can be learned end-to-end by simply maximizing task performance. However, such meta-RL approaches struggle with local optima due to a chicken-and-egg problem: learning to explore requires good exploitation to gauge the exploration's utility, but learning to exploit requires information gathered via exploration. Optimizing separate objectives for exploration and exploitation can avoid this problem, but prior meta-RL exploration objectives yield suboptimal policies that gather information irrelevant to the task. We alleviate both concerns by constructing an exploitation objective that automatically identifies task-relevant information and an exploration objective to recover only this information. This avoids local optima in end-to-end training, without sacrificing optimal exploration. Empirically, DREAM substantially outperforms existing approaches on complex meta-RL problems, such as sparse-reward 3D visual navigation.

## Hard Attention Control By Mutual Information Maximization

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'attention', 'partial observability', 'mutual information', 'information theory']

Biological agents have adopted the principle of attention to limit the rate of incoming information from the environment. One question that arises is if an artificial agent has access to only a limited view of its surroundings, how can it control its attention to effectively solve tasks? We propose an approach for learning how to control a hard attention window by maximizing the mutual information between the environment state and the attention location at each step. The agent employs an internal world model to make predictions about its state and focuses attention towards where the predictions may be wrong. Attention is trained jointly with a dynamic memory architecture that stores partial observations and keeps track of the unobserved state. We demonstrate that our approach is effective in predicting the full state from a sequence of partial observations. We also show that the agent's internal representation of the surroundings, a live mental map, can be used for control in two partially observable reinforcement learning tasks. Videos of the trained agent can be found at ~\url{https://sites.google.com/view/hard-attention-control}.

## Robust Imitation via Decision-Time Planning

Authors: ['Anonymous']

Keywords: ['imitation learning', 'reinforcement learning', 'inverse reinforcement learning']

The goal of imitation learning is to mimic expert behavior from demonstrations, without access to an explicit reward signal. A popular class of approaches infers the (unknown) reward function via inverse reinforcement learning (IRL) followed by maximizing this reward function via reinforcement learning (RL). The policies learned via these approaches are however very brittle in practice and deteriorate quickly even with small test-time perturbations due to compounding errors. We propose Imitation with Planning at Test-time (IMPLANT), a new algorithm for imitation learning that utilizes decision-time planning to correct for compounding errors of any base imitation policy. In contrast to existing approaches, we retain both the imitation policy and the rewards model at decision-time, thereby benefiting from the learning signal of the two components. Empirically, we demonstrate that IMPLANT significantly outperforms benchmark imitation learning approaches on standard control environments and excels at zero-shot generalization when subject to challenging perturbations in test-time dynamics.

## Discovering Diverse Multi-Agent Strategic Behavior via Reward Randomization

Authors: ['Anonymous']

Keywords: ['strategic behavior', 'multi-agent reinforcement learning', 'reward randomization', 'diverse strategies']

We propose a simple, general and effective technique, Reward Randomization for discovering diverse strategic policies in complex multi-agent games. Combining reward randomization and policy gradient, we derive a new algorithm, Reward-Randomized Policy Gradient (RPG). RPG is able to discover a set of multiple distinctive human-interpretable strategies in challenging temporal trust dilemmas, including grid-world games and a real-world game Agar.io, where multiple equilibria exist but standard multi-agent policy gradient algorithms always converge to a fixed one with a sub-optimal payoff for every player even using state-of-the-art exploration techniques. Furthermore, with the set of diverse strategies from RPG, we can (1) achieve higher payoffs by fine-tuning the best policy from the set; and (2) obtain an adaptive agent by using this set of strategies as its training opponents. 

## Play to Grade: Grading Interactive Coding Games as Classifying Markov Decision Process

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement Learning', 'Education', 'Automated Grading', 'Program Testing']

Contemporary coding education often present students with the task of developing programs that have user interaction and complex dynamic systems, such as mouse based games. While pedagogically compelling, grading such student programs requires dynamic user inputs, therefore they are difficult to grade by unit tests. In this paper we formalize the challenge of grading interactive programs as a task of classifying Markov Decision Processes (MDPs). Each student's program fully specifies an MDP where the agent needs to operate and decide, under reasonable generalization, if the dynamics and reward model of the input MDP conforms to a set of latent MDPs. We demonstrate that by experiencing a handful of latent MDPs millions of times, we can use the agent to sample trajectories from the input MDP and use a classifier to determine membership. Our method drastically reduces the amount of data needed to train an automatic grading system for interactive code assignments and present a challenge to state-of-the-art reinforcement learning generalization methods. Together with Code.org, we curated a dataset of 700k student submissions, one of the largest dataset of anonymized student submissions to a single assignment. This Code.org assignment had no previous solution for automatically providing correctness feedback to students and as such this contribution could lead to meaningful improvement in educational experience.

## Blending MPC & Value Function Approximation for Efficient Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'model-predictive control']

Model-Predictive Control (MPC) is a powerful tool for controlling complex, real-world systems that uses a model to make predictions about future behavior. For each state encountered, MPC solves an online optimization problem to choose a control action that will minimize future cost. This is a surprisingly effective strategy, but real-time performance requirements warrant the use of simple models. If the model is not sufficiently accurate, then the resulting controller can be biased, limiting performance. We present a framework for improving on MPC with model-free reinforcement learning (RL). The key insight is to view MPC as constructing a series of local Q-function approximations. We show that by using a parameter $\lambda$, similar to the trace decay parameter in TD($\lambda$), we can systematically trade-off learned value estimates against the local Q-function approximations. We present a theoretical analysis that shows how error from inaccurate models in MPC and value function estimation in RL can be balanced. We further propose an algorithm that changes $\lambda$ over time to reduce the dependence on MPC as our estimates of the value function improve, and test the efficacy our approach on challenging high-dimensional manipulation tasks with biased models in simulation. We demonstrate that our approach can obtain performance comparable with MPC with access to true dynamics even under severe model bias and is more sample efficient as compared to model-free RL.

## A Policy Gradient Algorithm for Learning to Learn in Multiagent Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Multiagent reinforcement learning', 'Meta-learning', 'Non-stationarity']

A fundamental challenge in multiagent reinforcement learning is to learn beneficial behaviors in a shared environment with other agents that are also simultaneously learning. In particular, each agent perceives the environment as effectively non-stationary due to the changing policies of other agents. Moreover, each agent is itself constantly learning, leading to natural nonstationarity in the distribution of experiences encountered. In this paper, we propose a novel meta-multiagent policy gradient theorem that directly accommodates for the non-stationary policy dynamics inherent to these multiagent settings. This is achieved by modeling our gradient updates to directly consider both an agent’s own non-stationary policy dynamics and the non-stationary policy dynamics of other agents interacting with it in the environment. We find that our theoretically grounded approach provides a general solution to the multiagent learning problem, which inherently combines key aspects of previous state of the art approaches on this topic. We test our method on several multiagent benchmarks and demonstrate a more efficient ability to adapt to new agents as they learn than previous related approaches across the spectrum of mixed incentive, competitive, and cooperative environments.

## AWAC: Accelerating Online Reinforcement Learning with Offline Datasets

Authors: ['Anonymous']

Keywords: ['reinforcement learning']

Reinforcement learning provides an appealing formalism for learning control policies from experience. However, the classic active formulation of reinforcement learning necessitates a lengthy active exploration process for each behavior, making it difficult to apply in real-world settings. If we can instead allow reinforcement learning to effectively use previously collected data to aid the online learning process, where the data could be expert demonstrations or more generally any prior experience, we could make reinforcement learning a substantially more practical tool. While a number of recent methods have sought to learn offline from previously collected data, it remains exceptionally difficult to train a policy with offline data and improve it further with online reinforcement learning. In this paper we systematically analyze why this problem is so challenging, and propose an algorithm that combines sample-efficient dynamic programming with maximum likelihood policy updates, providing a simple and effective framework that is able to leverage large amounts of offline data and then quickly perform online fine-tuning of reinforcement learning policies. We show that our method enables rapid learning of skills with a combination of prior demonstration data and online experience across a suite of difficult dexterous manipulation and benchmark tasks.

## Symmetry-Augmented Representation for Time Series

Authors: ['Anonymous']

Keywords: ['Time Series', 'Symmetry', 'Homology', 'Augmentation', 'Machine Learning', 'Reinforcement Learning']

We examine the hypothesis that the concept of symmetry augmentation is fundamentally linked to learning. Our focus in this study is on the augmentation of symmetry embedded in 1-dimensional time series (1D-TS). Motivated by the duality between 1D-TS and networks, we augment the symmetry by converting 1D-TS into three 2-dimensional representations: temporal correlation (GAF), transition dynamics (MTF), and recurrent events (RP). This conversion does not require a priori knowledge of the types of symmetries hidden in the 1D-TS. We then exploit the equivariance property of CNNs to learn the hidden symmetries in the augmented 2-dimensional data. We show that such conversion only increases the amount of symmetry, which may lead to more efficient learning. Specifically, we prove that a direct sum based augmentation will never decrease the amount of symmetry. We also attempt to measure the amount of symmetry in the original 1D-TS and augmented representations using the notion of persistent homology, which reveals symmetry increase after augmentation quantitatively. We present empirical studies to confirm our findings using two cases: reinforcement learning for financial portfolio management and classification with the CBF data set. Both cases demonstrate significant improvements in learning efficiency.

## Imitation with Neural Density Models

Authors: ['Anonymous']

Keywords: ['Imitation Learning', 'Reinforcement Learning', 'Density Estimation', 'Density Model', 'Maximum Entropy RL', 'Mujoco']

We propose a new framework for Imitation Learning (IL) via density estimation of the expert's occupancy measure followed by Maximum Occupancy Entropy Reinforcement Learning (RL) using the density as a reward. Our approach maximizes a non-adversarial model-free RL objective that provably lower bounds reverse Kullback–Leibler divergence between occupancy measures of the expert and imitator. We present a practical IL algorithm, Neural Density Imitation (NDI), which obtains state-of-the-art demonstration efficiency on benchmark control tasks. 

## Improving Learning to Branch via Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Mixed Integer Programming', 'Branching and Bound', 'Strong Branching', 'Reinforcement Learning', 'Evolution Strategy', 'Novelty Search']

Branch-and-Bound~(B\&B) is a general and widely used algorithm paradigm for solving Mixed Integer Programming~(MIP). 
Recently there is a surge of interest in designing learning-based branching policies as a fast approximation of strong branching, a human-designed heuristic. In this work, we argue strong branching is not a good expert to imitate for its poor decision quality when turning off its side effects in solving linear programming. To obtain more effective and non-myopic policies than a local heuristic, we formulate the branching process in MIP as reinforcement learning~(RL) and design a policy characterization for the B\&B process to improve our agent by novelty search evolutionary strategy. Across a range of NP-hard problems, our trained RL agent significantly outperforms expert-designed branching rules and the state-of-the-art learning-based branching methods in terms of both speed and effectiveness. Our results suggest that with carefully designed policy networks and learning algorithms, reinforcement learning has the potential to advance algorithms for solving MIPs.

## Offline Policy Optimization with Variance Regularization

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'offline batch RL', 'off-policy', 'policy optimization', 'variance regularization']

Learning policies from fixed offline datasets is a key challenge to scale up reinforcement learning (RL) algorithms towards practical applications. This is often because off-policy RL algorithms suffer from distributional shift, due to mismatch between dataset and the target policy, leading to high variance and over-estimation of value functions. In this work, we propose variance regularization for offline RL algorithms, using stationary distribution corrections. We show that by using Fenchel duality, we can avoid double sampling issues for computing the gradient of the variance regularizer. The proposed algorithm for offline variance regularization can be used to augment  any existing offline policy optimization algorithms. We  show that the regularizer leads to a lower bound to the offline policy optimization objective, which can help avoid over-estimation errors, and explains the benefits of our approach  across a range of continuous control domains when compared to existing algorithms. 

## Causal Inference Q-Network: Toward Resilient Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement Learning', 'Causal Inference', 'Robust Reinforcement Learning', 'Adversarial Robustness']

Deep reinforcement learning (DRL) has demonstrated impressive performance in various gaming simulators and real-world applications. In practice, however, a DRL agent may receive faulty observation by abrupt interferences such as black-out, frozen-screen, and adversarial perturbation. How to design a resilient DRL algorithm against these rare but mission-critical and safety-crucial scenarios is an important yet challenging task. In this paper, we consider a resilient DRL framework with observational interferences. Under this framework, we discuss the importance of the causal relation and propose a causal inference based DRL algorithm called causal inference Q-network (CIQ). We evaluate the performance of CIQ in several benchmark DRL environments with different types of interferences. Our experimental results show that the proposed CIQ method could achieve higher performance and more resilience against observational interferences.

## Impact-driven Exploration with Contrastive Unsupervised Representations

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'exploration', 'curiosity', 'episodic memory']

Procedurally-generated sparse reward environments pose significant challenges for many RL algorithms. The recently proposed impact-driven exploration method (RIDE) by Raileanu & Rocktäschel (2020), which rewards actions that lead to large changes (measured by $\ell_2$-distance) in the observation embedding, achieves state-of-the-art performance on such procedurally-generated MiniGrid tasks. Yet, the definition of "impact" in RIDE is not conceptually clear because its learned embedding space is not inherently equipped with any similarity measure, let alone $\ell_2$-distance. We resolve this issue in RIDE via contrastive learning. That is, we train the embedding with respect to cosine similarity, where we define two observations to be similar if the agent can reach one observation from the other within a few steps, and define impact in terms of this similarity measure. Experimental results show that our method performs similarly to RIDE on the MiniGrid benchmarks while learning a conceptually clear embedding space equipped with the cosine similarity measure. Our modification of RIDE also provides a new perspective which connects RIDE and episodic curiosity (Savinov et al., 2019), a different exploration method which rewards the agent for visiting states that are unfamiliar to the agent's episodic memory. By incorporating episodic memory into our method, we outperform RIDE on the MiniGrid benchmarks.

## Safe Reinforcement Learning with Natural Language Constraints

Authors: ['Anonymous']

Keywords: ['Safe reinforcement learning', 'Language grounding']

In this paper, we tackle the problem of learning control policies for tasks when provided with constraints in natural language.  In contrast to instruction following, language here is used not to specify goals, but rather to describe situations that an agent must avoid during its exploration of the environment. Specifying constraints in natural language also differs from the predominant paradigm in safe reinforcement learning, where safety criteria are enforced by hand-defined cost functions. While natural language allows for easy and flexible specification of safety constraints and budget limitations, its ambiguous nature presents a challenge when mapping these specifications into representations that can be used by techniques for safe reinforcement learning. To address this, we develop a model that contains two components: (1) a constraint interpreter to encode natural language constraints into vector representations capturing spatial and temporal information on forbidden states, and (2) a policy network that uses these representations to output a policy with minimal constraint violations. Our model is end-to-end differentiable and we train it using a recently proposed algorithm for constrained policy optimization. To empirically demonstrate the effectiveness of our approach, we create a new benchmark task for autonomous navigation with crowd-sourced free-form text specifying three different types of constraints. Our method outperforms several baselines by achieving 6-7 times higher returns and 76% fewer constraint violations on average. Dataset and code to reproduce our experiments are available at https://sites.google.com/view/polco-hazard-world/.

## Reset-Free Lifelong Learning with Skill-Space Planning

Authors: ['Anonymous']

Keywords: ['reset-free', 'lifelong', 'reinforcement learning']

The objective of \textit{lifelong} reinforcement learning (RL) is to optimize agents which can continuously adapt and interact in changing environments. However, current RL approaches fail drastically when environments are non-stationary and interactions are non-episodic. We propose \textit{Lifelong Skill Planning} (LiSP), an algorithmic framework for lifelong RL based on planning in an abstract space of higher-order skills. We learn the skills in an unsupervised manner using intrinsic rewards and plan over the learned skills using a learned dynamics model. Moreover, our framework permits skill discovery even from offline data, thereby reducing the need for excessive real-world interactions. We demonstrate empirically that LiSP successfully enables long-horizon planning and learns agents that can avoid catastrophic failures even in challenging non-stationary and non-episodic environments derived from gridworld and MuJoCo benchmarks.

## Robust Multi-Agent Reinforcement Learning Driven by Correlated Equilibrium

Authors: ['Anonymous']

Keywords: ['Robust', 'Multi-agent', 'Reinforcement Learning', 'Correlated Equilibrium']

In this paper we deal with robust cooperative multi-agent reinforcement learning (CMARL). While CMARL has many potential applications, only a trained policy that is robust enough can be confidently deployed in real world. Existing works on robust MARL mainly apply vanilla adversarial training in centralized training and decentralized execution paradigm. We, however, find that if a CMARL environment contains an adversarial agent, the performance of decentralized equilibrium might perform significantly poor for achieving such adversarial robustness. To tackle this issue, we suggest that when execution the non-adversarial agents must jointly make the decision to improve the robustness, therefore solving correlated equilibrium instead. We theoretically demonstrate the superiority of correlated equilibrium over the decentralized one in adversarial MARL settings. Therefore, to achieve robust CMARL, we introduce novel strategies to encourage agents to learn correlated equilibrium while maximally preserving the convenience of the decentralized execution. The global variables with mutual information are proposed to help agents learn robust policies with MARL algorithms. The experimental results show that our method can dramatically boost performance on the SMAC environments.

## Graph Convolutional Value Decomposition in Multi-Agent Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['multi-agent deep reinforcement learning', 'graph neural networks', 'value function factorization', 'attention mechanisms']

We propose a novel framework for value function factorization in multi-agent deep reinforcement learning using graph neural networks (GNNs). In particular, we consider the team of agents as the set of nodes of a complete directed graph, whose edge weights are governed by an attention mechanism. Building upon this underlying graph, we introduce a mixing GNN module, which is responsible for two tasks: i) factorizing the team state-action value function into individual per-agent observation-action value functions, and ii) explicit credit assignment to each agent in terms of fractions of the global team reward. Our approach, which we call GraphMIX, follows the centralized training and decentralized execution paradigm, enabling the agents to make their decisions independently once training is completed. Experimental results on the StarCraft II multi-agent challenge (SMAC) environment demonstrate the superiority of our proposed approach as compared to the state-of-the-art.

## Planning from Pixels using Inverse Dynamics Models

Authors: ['Anonymous']

Keywords: ['model based reinforcement learning', 'deep reinforcement learning', 'multi-task learning', 'deep learning', 'goal-conditioned reinforcement learning']

Learning dynamics models in high-dimensional observation spaces can be challenging for model-based RL agents. We propose a novel way to learn models in a latent space by learning to predict sequences of future actions conditioned on task completion. These models track task-relevant environment dynamics over a distribution of tasks, while simultaneously serving as an effective heuristic for planning with sparse rewards. We evaluate our method on challenging visual goal completion tasks and show a substantial increase in performance compared to prior model-free approaches.

## A Coach-Player Framework for Dynamic Team Composition

Authors: ['Anonymous']

Keywords: ['Multiagent reinforcement learning']

In real-world multi-agent teams, agents with different capabilities may join or leave "on the fly" without altering the team's overarching goals. Coordinating teams with such dynamic composition remains a challenging problem: the optimal team strategy may vary with its composition. Inspired by real-world team sports, we propose a coach-player framework to tackle this problem. We assume that the players only have a partial view of the environment, while the coach has a complete view. The coach coordinates the players by distributing individual strategies. Specifically, we 1) propose an attention mechanism for both the players and the coach; 2) incorporate a variational objective to regularize  learning; and 3) design an adaptive communication method to let the coach decide when to communicate with different players. Our attention mechanism on the players and the coach allows for a varying number of heterogeneous agents, and can thus tackle the dynamic team composition.  We validate our methods on resource collection tasks in multi-agent particle environment. We demonstrate zero-shot generalization to new team compositions with varying numbers of heterogeneous  agents. The performance of our method is comparable or even better than the  setting where all players have a full view of the environment, but no coach. Moreover, we see that the performance stays nearly the same even when the coach communicates as little as 13% of the time using our adaptive communication strategy.  These results demonstrate   the significance of a coach to coordinate players in dynamic teams.

## Prior Preference Learning From Experts: Designing A Reward with Active Inference

Authors: ['Anonymous']

Keywords: ['Active Inference', 'Free Energy Principle', 'Reinforcement Learning', 'Reward Design']

Active inference may be defined as Bayesian modeling of a brain with a biologically plausible model of the agent. Its primary idea relies on the free energy principle and the prior preference of the agent. An agent will choose an action that leads to its prior preference for a future observation. In this paper, we claim that active inference can be interpreted using reinforcement learning (RL) algorithms and find a theoretical connection between them. We extend the concept of expected free energy (EFE), which is a core quantity in active inference, and claim that EFE can be treated as a negative value function. Motivated by the concept of prior preference and a theoretical connection, we propose a simple but novel method for learning a prior preference from experts. This illustrates that the problem with RL can be approached with a new perspective of active inference. Experimental results of prior preference learning show the possibility of active inference with EFE-based rewards and its application to an inverse RL problem.

## Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients

Authors: ['Anonymous']

Keywords: ['symbolic regression', 'reinforcement learning', 'automated machine learning']

Discovering the underlying mathematical expressions describing a dataset is a core challenge for artificial intelligence. This is the problem of symbolic regression. Despite recent advances in training neural networks to solve complex tasks, deep learning approaches to symbolic regression are underexplored. We propose a framework that leverages deep learning for symbolic regression via a simple idea: use a large model to search the space of small models. Specifically, we use a recurrent neural network to emit a distribution over tractable mathematical expressions and employ a novel risk-seeking policy gradient to train the network to generate better-fitting expressions. Our algorithm outperforms several baseline methods (including Eureqa, the gold standard for symbolic regression) in its ability to exactly recover symbolic expressions on a series of benchmark problems, both with and without added noise. More broadly, our contributions include a framework that can be applied to optimize hierarchical, variable-length objects under a black-box performance metric, with the ability to incorporate constraints in situ, and a risk-seeking policy gradient formulation that optimizes for best-case performance instead of expected performance.

## Bridging the Gap: Providing Post-Hoc Symbolic Explanations for Sequential Decision-Making Problems with Inscrutable Representations

Authors: ['Anonymous']

Keywords: ['Explanations', 'Concept based explanations', 'Learning symbolic representations', 'Sequential Decision Making', 'Planning', 'Reinforcement learning.']

As increasingly complex AI systems are introduced into our daily lives, it becomes important for such systems to be capable of explaining the rationale for their decisions and allowing users to contest these decisions. A significant hurdle to allowing for such explanatory dialogue could be the vocabulary mismatch between the user and the AI system. This paper introduces methods for providing contrastive explanations in terms of user-specified concepts for sequential decision-making settings where the system's model of the task may be best represented as a blackbox simulator. We do this by building partial symbolic models of a local approximation of the task that can be leveraged to answer the user queries. We empirically test these methods on a popular Atari game (Montezuma's Revenge) and modified versions of Sokoban (a well-known planning benchmark) and report the results of user studies to evaluate whether people find explanations generated in this form useful.

## A Strong On-Policy Competitor To PPO

Authors: ['Anonymous']

Keywords: ['proximal policy optimization', 'deep reinforcement learning']

As a recognized variant and improvement for Trust Region Policy Optimization (TRPO), proximal policy optimization (PPO) has been widely used with several advantages: efficient data utilization, easy implementation and good parallelism. In this paper, a first-order gradient on-policy learning algorithm called Policy Optimization with Penalized Point Probability Distance (POP3D), which is a lower bound to the square of total variance divergence is proposed as another powerful variant. The penalty item has dual effects, prohibiting policy updates from overshooting and encouraging more explorations.  Carefully controlled experiments on both discrete and continuous benchmarks verify our approach is highly competitive to PPO.

## On Effective Parallelization of Monte Carlo Tree Search

Authors: ['Anonymous']

Keywords: ['parallel Monte Carlo Tree Search (MCTS)', 'Upper Confidence bound for Trees (UCT)', 'Reinforcement Learning (RL)']

Despite its groundbreaking success in Go and computer games, Monte Carlo Tree Search (MCTS) is computationally expensive as it requires a substantial number of rollouts to construct the search tree, which calls for effective parallelization. However, how to design effective parallel MCTS algorithms has not been systematically studied and remains poorly understood. In this paper, we seek to lay its first theoretical foundation, by examining the potential performance loss caused by parallelization when achieving a desired speedup. In particular, we discover the necessary conditions of achieving a desirable parallelization performance, and highlight two of their practical benefits. First, by examining whether existing parallel MCTS algorithms satisfy these conditions, we identify key design principles that should be inherited by future algorithms, for example tracking the unobserved samples (used in WU-UCT (liu et al., 2020). We theoretically establish this essential design facilitates $\mathcal{O} ( \ln n + M / \sqrt{\ln n} )$ cumulative regret when the maximum tree depth is 2, where $n$ is the number of rollouts and $M$ is the number of workers. A regret of this form is highly desirable, as compared to $\mathcal{O} ( \ln n )$ regret incurred by a sequential counterpart, its excess part approaches zero as $n$ increases. Second, and more importantly, we demonstrate how the proposed necessary conditions can be adopted to design more effective parallel MCTS algorithms. To illustrate this, we propose a new parallel MCTS algorithm, called BU-UCT, by following our theoretical guidelines. The newly proposed algorithm, albeit preliminary, out-performs four competitive baselines on 11 out of 15 Atari games. We hope our theoretical results could inspire future work of more effective parallel MCTS.

## Learn Goal-Conditioned Policy with Intrinsic Motivation for Deep Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['unsupervised reinforcement learning', 'goal-conditioned policy', 'intrinsic reward']

It is of significance for an agent to learn a widely applicable and general-purpose policy that can achieve diverse goals including images and text descriptions. Considering such perceptually-specific goals, the frontier of deep reinforcement learning research is to learn a goal-conditioned policy without hand-crafted rewards. To learn this kind of policy, recent works usually take as the reward the non-parametric distance to a given goal in an explicit embedding space. From a different viewpoint, we propose a novel unsupervised learning approach named goal-conditioned policy with intrinsic motivation (GPIM), which jointly learns both an abstract-level policy and a goal-conditioned policy. The abstract-level policy is conditioned on a latent variable to provide intrinsic reward and discovers diverse states that are further rendered into perceptually-specific goals for the goal-conditioned policy, and both of the policies share the identical reward function. Experiments on various robotic tasks demonstrate the effectiveness and efficiency of our proposed GPIM method which substantially outperforms prior techniques. 

## Mobile Construction Benchmark

Authors: ['Anonymous']

Keywords: ['localization', 'dynamic environment', 'deep reinforcement learning', 'benchmark']

We need intelligent robots to perform mobile construction, the process of moving in an environment and modifying its geometry according to a design plan. Without GPS or similar techniques, carefully engineered and learning-based methods face challenges to exactly achieve the plan due to the difficulty of accurately localizing the robot while strategically evolving the environment, because common tasks (manipulation/navigation) address at most one of the two coupled aspects. To seek a generic solution, we simplify mobile construction in 1/2/3D grid worlds to benchmark the performance of existing deep RL methods on this partially observable MDP. Our results show that the coupling makes this problem very challenging for model-free RL, and emphasize the need for novel task-specific solutions.

## Interpretable Meta-Reinforcement Learning with Actor-Critic Method

Authors: ['Anonymous']

Keywords: ['meta-reinforcement learning', 'actor-critic', 'deep learning', 'interpretable']

Meta-reinforcement learning (meta-RL) algorithms have successfully trained agent systems to perform well on different tasks within only few updates. However, in gradient-based meta-RL algorithms, the Q-function at adaptation step is mainly estimated by the return of few trajectories, which can lead to high variance in Q-value and biased meta-gradient estimation, and the adaptation uses a large number of batched trajectories. To address these challenges, we propose a new meta-RL algorithm that can reduce the variance and bias of the meta-gradient estimation and perform few-shot task data sampling, which makes the meta-policy more interpretable. We reformulate the meta-RL objective, and introduce contextual Q-function as a meta-policy critic during task adaptation step and learn the Q-function under a soft actor-critic (SAC) framework. The experimental results on 2D navigation task and meta-RL benchmarks show that our approach can learn an more interpretable meta-policy to explore unknown environment and the performance are comparable to previous gradient-based algorithms.

## Fine-Tuning Offline Reinforcement Learning with Model-Based Policy Optimization

Authors: ['Anonymous']

Keywords: ['Offline Reinforcement Learning', 'Model-Based Reinforcement Learning', 'Off-policy Reinforcement Learning', 'uncertainty estimation']

In offline reinforcement learning (RL), we attempt to learn a control policy from a fixed dataset of environment interactions. This setting has the potential benefit of allowing us to learn effective policies without needing to collect additional interactive data, which can be expensive or dangerous in real-world systems. However, traditional off-policy RL methods tend to perform poorly in this setting due to the distributional shift between the fixed data set and the learned policy. In particular, they tend to extrapolate optimistically and overestimate the action-values outside of the dataset distribution. Recently, two major avenues have been explored to address this issue. First, behavior-regularized methods that penalize actions that deviate from the demonstrated action distribution. Second, uncertainty-aware model-based (MB) methods that discourage state-actions where the dynamics are uncertain. In this work, we propose an algorithmic framework that consists of two stages. In the first stage, we train a policy using behavior-regularized model-free RL on the offline dataset. Then, we can optionally enter a second stage where we fine-tune the policy using our novel Model-Based Behavior-Regularized Policy Optimization (MB2PO) algorithm. We demonstrate that for certain tasks and dataset distributions our conservative model-based fine tuning can greatly increase performance and allow the agent to generalize and outperform the demonstrated behavior. We evaluate our method on a variety of the Gym-MuJoCo tasks in the D4RL benchmark and demonstrate that our method is competitive and in some cases superior to the state of the art for most of the evaluated tasks.

## Measuring Visual Generalization in Continuous Control from Pixels

Authors: ['Anonymous']

Keywords: ['Continuous Control', 'Reinforcement Learning']

Self-supervised learning and data augmentation have significantly reduced the performance gap between state and image-based reinforcement learning agents in continuous control tasks. However, it is still unclear whether current techniques can face the variety of visual conditions required by real-world environments. We propose a challenging benchmark that tests agents' visual generalization by adding graphical variety to existing continuous control domains. Our empirical analysis shows that current methods struggle to generalize across a diverse set of visual changes, and we examine the specific factors of variation that make these tasks difficult. We find that data augmentation techniques outperform self-supervised learning approaches, and that more significant image transformations provide better visual generalization.

## TOMA: Topological Map Abstraction for Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Planning', 'Reinforcement Learning', 'Representation Learning']

Animals are able to discover the topological map (graph) of surrounding environment, which will be used for navigation. Inspired by this biological phenomenon, researchers have recently proposed to learn a graph representation for Markov decision process (MDP) and use such graphs for planning in reinforcement learning (RL). However, existing learning-based graph generation methods suffer from many drawbacks. One drawback is that existing methods do not learn an abstraction for graphs, which results in high memory and computation cost. This drawback also makes generated graph 
non-robust, which degrades the planning performance. Another drawback is that existing methods cannot be used for facilitating exploration which is important in RL. In this paper, we propose a new method, called topological map abstraction (TOMA), for graph generation. TOMA can learn an abstract graph representation for MDP, which costs much less memory and computation cost than existing methods. Furthermore, TOMA can be used for facilitating exploration. In particular, we propose planning to explore, in which TOMA is used to accelerate exploration by guiding the agent towards unexplored states. A novel experience replay module called vertex memory is also proposed to improve exploration performance. Experimental results show that TOMA can outperform existing methods to achieve the state-of-the-art performance.

## The act of remembering: A study in partially observable reinforcement learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Partial Observability', 'Memory Representations', 'External Memories', 'POMDPs.']

Reinforcement Learning (RL) agents typically learn memoryless policies---policies that only consider the last observation when selecting actions. Learning memoryless policies is efficient and optimal in fully observable environments. However, some form of memory is necessary when RL agents are faced with partial observability. In this paper, we study a lightweight approach to tackle partial observability in RL. We provide the agent with an external memory and additional actions to control what, if anything, is written to the memory. At every step, the current memory state is part of the agent’s observation, and the agent selects a tuple of actions: one action that modifies the environment and another that modifies the memory. When the external memory is sufficiently expressive, optimal memoryless policies yield globally optimal solutions. Unfortunately, previous attempts to use external memory in the form of binary memory have produced poor results in practice. Here, we investigate alternative forms of memory in support of learning effective memoryless policies. Our novel forms of memory outperform binary and LSTM-based memory in well-established partially observable domains.

## Evolving Reinforcement Learning Algorithms

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'evolutionary algorithms', 'meta-learning', 'genetic programming']

We propose a method for meta-learning reinforcement learning algorithms by searching over the space of computational graphs which compute the loss function for a value-based model-free RL agent to optimize. The learned algorithm is domain-agnostic and can generalize to new environments not seen during training. Our method can both learn from scratch and bootstrap off known existing algorithms, enabling interpretable modifications which obtain superior performance. We highlight two learned algorithms which obtain good generalization performance over a set of classical control tasks and gridworld type tasks. Our analysis of the learned algorithm behavior shows resemblance to recently proposed RL algorithms that have been designed manually.

## Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms

Authors: ['Anonymous']

Keywords: ['Multi-agent Reinforcement Learning', 'Benchmarking']

We benchmark commonly used multi-agent deep reinforcement learning (MARL) algorithms on a variety of cooperative multi-agent games. While there has been significant innovation in MARL algorithms, algorithms tend to be tested and tuned on a single domain and their average performance across multiple domains is less characterized. Furthermore, since the hyperparameters of the algorithms are carefully tuned to the task of interest, it is unclear whether hyperparameters can easily be found that allow the algorithm to be repurposed for other cooperative tasks with different reward structure and environment dynamics. To investigate the consistency of the performance of MARL algorithms, we build an open-source library of multi-agent algorithms including DDPG/TD3/SAC with centralized Q functions, PPO with centralized value functions, and QMix and test them across a range of tasks that vary in coordination difficulty and agent number. The domains include the particle-world environments, starcraft micromanagement challenges, the Hanabi challenge, and the hide-and-seek environments. Finally, we investigate the ease of hyper-parameter tuning for each of the algorithms by tuning hyper-parameters in one environment per domain and re-using them in the other environments within the domain.

## ReaPER: Improving Sample Efficiency in Model-Based Latent Imagination

Authors: ['Anonymous']

Keywords: ['model-based reinforcement learning', 'visual control', 'sample efficiency']

Deep Reinforcement Learning (DRL) can distill behavioural policies from sensory input that solve complex tasks, however, the policies tend to be task-specific and sample inefficient, requiring a large number of interactions with the environment that may be costly or impractical for many real world applications. Model-based DRL (MBRL) can allow learned behaviours and dynamics from one task to be translated to a new task in a related environment, but still suffer from low sample efficiency. In this work we introduce ReaPER, an algorithm that addresses the sample efficiency challenge in model-based DRL, we illustrate the power of the proposed solution on the DeepMind Control benchmark. Our improvements are driven by sparse , self-supervised, contrastive model representations and efficient use of past experience. We empirically analyze each novel  component of ReaPER and analyze how they contribute to sample efficiency. We also illustrate how other standard alternatives fail to improve upon previous methods. Code will be made available.

## EMaQ: Expected-Max Q-Learning Operator for Simple Yet Effective Offline and Online RL

Authors: ['Anonymous']

Keywords: ['Offline Reinforcement Learning', 'Off-Policy Reinforcement Learning']

Off-policy reinforcement learning (RL) holds the promise of sample-efficient learning of decision-making policies by leveraging past experience. However, in the offline RL setting -- where a fixed collection of interactions are provided and no further interactions are allowed -- it has been shown that standard off-policy RL methods can significantly underperform. Recently proposed methods often aim to address this shortcoming by constraining learned policies to remain close to the given dataset of interactions. In this work, we closely investigate an important simplification of BCQ~\citep{fujimoto2018off} -- a prior approach for offline RL -- which removes a heuristic design choice and naturally restrict extracted policies to remain \emph{exactly} within the support of a given behavior policy. Importantly, in contrast to their original theoretical considerations, we derive this simplified algorithm through the introduction of a novel backup operator, Expected-Max Q-Learning (EMaQ), which is more closely related to the resulting practical algorithm. Specifically, in addition to the distribution support, EMaQ explicitly considers the number of samples and the proposal distribution, allowing us to derive new sub-optimality bounds which can serve as a novel measure of complexity for offline RL problems. In the offline RL setting -- the main focus of this work -- EMaQ matches and outperforms prior state-of-the-art in the D4RL benchmarks~\citep{fu2020d4rl}. In the online RL setting, we demonstrate that EMaQ is competitive with Soft Actor Critic (SAC). The key contributions of our empirical findings are demonstrating the importance of careful generative model design for estimating behavior policies, and an intuitive notion of complexity for offline RL problems. With its simple interpretation and fewer moving parts, such as no explicit function approximator representing the policy, EMaQ serves as a strong yet easy to implement baseline for future work.

## Robust Offline Reinforcement Learning from Low-Quality Data

Authors: ['Anonymous']

Keywords: ['Offline reinforcement learning']

In practice, due to the need for online interaction with the environment, deploying reinforcement learning (RL) agents in safety-critical scenarios is challenging. Offline RL offers the promise of directly leveraging large, previously collected datasets to acquire effective policies without further interaction. Although a number of offline RL algorithms have been proposed, their performance is generally limited by the quality of dataset. To address this problem, we propose an Adaptive Policy constrainT (AdaPT) method, which allows effective exploration on out-of-distribution actions by imposing an adaptive constraint on the learned policy. We theoretically show that AdaPT produces a tight upper bound on the distributional deviation between the learned policy and the behavior policy, and this upper bound is the minimum requirement to guarantee policy improvement at each iteration. Formally, we present a practical AdaPT augmented Actor-Critic (AdaPT-AC) algorithm, which is able to learn a generalizable policy even from the dataset that contains a large amount of random data and induces a bad behavior policy. The empirical results on a range of continuous control benchmark tasks demonstrate that AdaPT-AC substantially outperforms several popular algorithms in terms of both final performance and robustness on four datasets of different qualities.

## Reinforcement Learning for Sparse-Reward Object-Interaction Tasks in First-person Simulated 3D Environments

Authors: ['Anonymous']

Keywords: ['object-centric', 'representation learning', 'reinforcement learning', 'sparse reward']

First-person object-interaction tasks in high-fidelity, 3D, simulated environments such as the AI2Thor virtual home-environment pose significant sample-efficiency challenges for reinforcement learning (RL) agents learning from sparse task rewards. To alleviate these challenges, prior work has provided extensive supervision via a combination of reward-shaping, ground-truth object-information, and expert demonstrations. In this work, we show that one can learn object-interaction tasks from scratch without supervision by learning an attentive object-model as an auxiliary task during task learning with an object-centric relational RL agent. Our key insight is that learning an object-model that incorporates object-relationships into forward prediction provides a dense learning signal for unsupervised representation learning of both objects and their relationships. This, in turn, enables faster policy learning for an object-centric relational RL agent. We demonstrate our agent by introducing a set of challenging object-interaction tasks in the AI2Thor environment where learning with our attentive object-model is key to strong performance. Specifically, by comparing our agent and relational RL agents with alternative auxiliary tasks with a relational RL agent equipped with ground-truth object-information, we find that learning with our object-model best closes the performance gap in terms of both learning speed and maximum success rate. Additionally, we find that incorporating object-relationships into an object-model's forward predictions is key to learning representations that capture object-category and object-state.

## Leveraging the Variance of Return Sequences for Exploration Policy

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Deep Reinforcement Learning', 'Exploration', 'Temporal Difference Error', 'Variance']

This paper introduces a novel method for constructing an upper bound for exploration policy using either the weighted variance of return sequences or the weighted temporal difference (TD) error.  We demonstrate that the variance of the return sequence for a specific state-action pair is an important information source that can be leveraged to guide exploration in reinforcement learning.  The intuition is that fluctuation in the return sequence indicates greater uncertainty in the near future returns.  This  divergence occurs because of the cyclic nature of value-based reinforcement learning; the evolving value function begets policy improvements which in turn modify the value function.  Although both variance and TD errors capture different aspects of this uncertainty, our analysis shows that both can be valuable to guide exploration.  We propose a two-stream network architecture to estimate weighted variance/TD errors within DQN agents for our exploration method and show that it outperforms the baseline on a wide range of Atari games.

## Multi-agent Policy Optimization with Approximatively Synchronous Advantage Estimation

Authors: ['Anonymous']

Keywords: ['multi-agent reinforcement learning', 'policy optimization', 'advantage estimation', 'credit assignment']

Cooperative multi-agent tasks require agents to deduce their own contributions with shared global rewards, known as the challenge of credit assignment. General methods for policy based multi-agent reinforcement learning to solve the challenge introduce differentiate value functions or advantage functions for individual agents. In multi-agent system, polices of different agents need to be evaluated jointly. In order to update polices synchronously, such value functions or advantage functions also need synchronous evaluation. However, in current methods, value functions or advantage functions use counter-factual joint actions which are evaluated asynchronously, thus suffer from natural estimation bias. In this work, we propose the approximatively synchronous advantage estimation. We first derive the marginal advantage function, an expansion from single-agent advantage function to multi-agent system. Further more, we introduce a policy approximation for synchronous advantage estimation, and break down the multi-agent policy optimization problem into multiple sub-problems of single-agent policy optimization. Our method is compared with baseline algorithms on StarCraft multi-agent challenges, and shows the best performance on most of the tasks.

## Re-examining Routing Networks for Multi-task Learning

Authors: ['Anonymous']

Keywords: ['Routing Networks', 'Multi-task Learning', 'Reinforcement Learning']

We re-examine Routing Networks, an approach to multi-task learning that uses reinforcement learning to decide parameter sharing with the goal of maximizing knowledge transfer between related tasks while avoiding task interference. These benefits come with the cost of solving a more difficult optimization problem. We argue that the success of this model depends on a few key assumptions and, when they are not satisfied, the difficulty of learning a good route can outweigh the benefits of the approach. In these cases, a simple unlearned routing strategy, which we propose, achieves the best results.

## Joint Perception and Control as Inference with an Object-based Implementation

Authors: ['Anonymous']

Keywords: ['model-based reinforcement learning', 'perception modeling', 'object-based reinforcement learning']

Existing model-based reinforcement learning methods often study perception modeling and decision making separately. We introduce joint Perception and Control as Inference (PCI), a general framework to combine perception and control for partially observable environments through Bayesian inference. Based on the fact that object-level inductive biases are critical in human perceptual learning and reasoning, we propose Object-based Perception Control (OPC), an instantiation of PCI which manages to facilitate control using automatic discovered object-based representations. We develop an unsupervised end-to-end solution and analyze the convergence of the perception model update. Experiments in a high-dimensional pixel environment demonstrate the learning effectiveness of our object-based perception control approach. Specifically, we show that OPC achieves good perceptual grouping quality and outperforms several strong baselines in accumulated rewards.

## Parrot: Data-Driven Behavioral Priors for Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'imitation learning']

Reinforcement learning provides a general framework for flexible decision making and control, but requires extensive data collection for each new task that an agent needs to learn. In other machine learning fields, such as natural language processing or computer vision, pre-training on large, previously collected datasets to bootstrap learning for new tasks has emerged as a powerful paradigm to reduce data requirements when learning a new task. In this paper, we ask the following question: how can we enable similarly useful pre-training for RL agents? We propose a method for pre-training behavioral priors that can capture complex input-output relationships observed in successful trials from a wide range of previously seen tasks, and we show how this learned prior can be used for rapidly learning new tasks without impeding the RL agent's ability to try out novel behaviors. We demonstrate the effectiveness of our approach in challenging robotic manipulation domains involving image observations and sparse reward functions, where our method outperforms prior works by a substantial margin. 

## Benchmarks for Deep Off-Policy Evaluation

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'off-policy evaluation', 'benchmarks']

Off-policy evaluation (OPE) holds the promise of being able to leverage large, offline datasets for both obtaining and selecting complex policies for decision making. The ability to perform evaluation offline is particularly important in many real-world domains, such as healthcare, recommender systems, or robotics, where online data collection is an expensive and potentially dangerous process. Being able to accurately evaluate and select high-performing policies without requiring online interaction could yield significant benefits in safety, time, and cost for these applications. While many OPE methods have been proposed in recent years, comparing results between works is difficult because there is currently a lack of a comprehensive and unified benchmark. Moreover, it is difficult to measure how far algorithms have progressed, due to the lack of challenging evaluation tasks. In order to address this gap, we propose a new benchmark for off-policy evaluation which includes tasks on a range of challenging, high-dimensional control problems, with wide selections of datasets and policies for performing policy selection. The goal of of our benchmark is to provide a standardized measure of progress that is motivated from a set of principles designed to challenge and test the limits of existing OPE methods. We perform a comprehensive evaluation of state-of-the-art algorithms, and we will provide open-source access to all data and code to foster future research in this area.

## Success-Rate Targeted Reinforcement Learning by Disorientation Penalty

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'undiscounted return', 'success rate']

Current reinforcement learning generally uses discounted return as its learning objective. However, real-world tasks may often demand a high success rate, which can be quite different from optimizing rewards. In this paper, we explicitly formulate the success rate as an undiscounted form of return with {0, 1}-binary reward function. Unfortunately, applying traditional Bellman updates to value function learning can be problematic for learning undiscounted return, and thus not suitable for optimizing success rate. From our theoretical analysis, we discover that values across different states tend to converge to the same value, resulting in the agent wandering around those states without making any actual progress. This further leads to reduced learning efficiency and inability to complete a task in time. To combat the aforementioned issue, we propose a new method, which introduces Loop Penalty (LP) into value function learning, to penalize disoriented cycling behaviors in the agent's decision-making. We demonstrate the effectiveness of our proposed LP on three environments, including grid-world cliff-walking, Doom first-person navigation and robot arm control, and compare our method with Q-learning, Monte-Carlo and Proximal Policy Optimization (PPO). Empirically, LP improves the convergence of training and achieves a higher success rate.

## Decorrelated Double Q-learning

Authors: ['Anonymous']

Keywords: ['q-learning', 'control variates', 'reinforcement learning']

Q-learning with value function approximation may have the poor performance because of overestimation bias and imprecise estimate. Specifically, overestimation bias is from the maximum operator over noise estimate, which is exaggerated using the estimate of a subsequent state. Inspired by the recent advance of deep reinforcement learning and Double Q-learning, we introduce the decorrelated double Q-learning (D2Q). Specifically, we introduce the decorrelated regularization item to reduce the correlation between value function approximators, which can lead to less biased estimation and low variance. The experimental results on a suite of MuJoCo continuous control tasks demonstrate that our decorrelated double Q-learning can effectively improve the performance.

## Return-Based Contrastive Representation Learning for Reinforcement  Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'auxiliary task', 'representation learning', 'contrastive learning']

Recently, various auxiliary tasks have been proposed to accelerate representation learning and improve sample efficiency in deep reinforcement learning (RL). Existing auxiliary tasks do not take the characteristics of RL problems into consideration and are unsupervised. By leveraging returns, the most important feedback signals in RL, we propose a novel auxiliary task that forces the learnt representations to discriminate state-action pairs with different returns. Our auxiliary loss is theoretically justified to learn representations that capture the structure of a new form of state-action abstraction, under which state-action pairs with similar return distributions are aggregated together. Empirically, our algorithm outperforms strong baselines on complex tasks in DeepMind Control suite and Atari games, and achieves even better performance when combined with existing auxiliary tasks.

## Active Feature Acquisition with Generative Surrogate Models

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Active Feature Acquisition', 'Feature Selection']

Many real-world situations allow for the acquisition of additional relevant information when making an assessment with limited or uncertain data. However, traditional ML approaches either require all features to be acquired beforehand or regard part of them as missing data that cannot be acquired. In this work, we propose models that perform active feature acquisition (AFA) to improve the prediction assessments at evaluation time. We formulate the AFA problem as a Markov decision process (MDP) and resolve it using reinforcement learning (RL). The AFA problem yields sparse rewards and contains a high-dimensional complicated action space. Thus, we propose learning a generative surrogate model that captures the complicated dependencies among input features to assess potential information gain from acquisitions. We also leverage the generative surrogate model to provide intermediate rewards and auxiliary information to the agent. Furthermore, we extend AFA in a task we coin active instance recognition (AIR) for the unsupervised case where the target variables are the unobserved features themselves and the goal is to collect information for a particular instance in a cost-efficient way. Empirical results demonstrate that our approach achieves considerably better performance than previous state of the art methods on both supervised and unsupervised tasks.

## Provable Rich Observation Reinforcement Learning with Combinatorial Latent States

Authors: ['Anonymous']

Keywords: ['Reinforcement learning theory', 'Rich observation', 'Noise-contrastive learning', 'State abstraction', 'Factored MDP']

We propose a novel setting for reinforcement learning that combines two common real-world difficulties: presence of observations (such as camera images) and factored states (such as location of objects). In our setting, the agent receives observations generated stochastically from a "latent" factored state. These observations are "rich enough" to enable decoding of the latent state and remove partial observability concerns. Since the latent state is combinatorial, the size of state space is exponential in the number of latent factors. We create a learning algorithm FactoRL (Fact-o-Rel) for this setting,  which uses noise-contrastive learning to identify latent structures in emission processes and discover a factorized state space. We derive polynomial sample complexity guarantees for FactoRL which polynomially depend upon the number factors, and very weakly depend on the size of the observation space.  We also provide a guarantee of polynomial time complexity when given access to an efficient planning algorithm.

## Regularized Inverse Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['inverse reinforcement learning', 'reward learning', 'regularized markov decision processes', 'reinforcement learning']

Inverse Reinforcement Learning (IRL) aims to facilitate a learner's ability to imitate expert behavior by acquiring reward functions that explain the expert's decisions. Regularized IRL applies convex regularizers to the learner's policy in order to avoid degenerate solutions, i.e., whereby any constant reward rationalizes the expert behavior. We propose analytical solutions, and practical methods to obtain them, for regularized IRL. Current methods are restricted to the maximum-entropy IRL framework, limiting them to Shannon-entropy regularizers, as well as proposing functional-form solutions that are generally intractable. We present theoretical backing for our proposed IRL method's applicability for both discrete and continuous controls, empirically validating our performance on a variety of tasks.  

## Safety Verification of Model Based Reinforcement Learning Controllers

Authors: ['Anonymous']

Keywords: ['Reachable set', 'state constraints', 'safety verification', 'model-based reinforcement learning']

Model-based reinforcement learning (RL) has emerged as a promising tool for developing controllers for real world systems (e.g., robotics, autonomous driving, etc.).  However, real systems often have constraints imposed on their state space which must be satisfied to ensure the safety of the system and its environment. Developing a verification tool for RL algorithms is challenging because the non-linear structure of neural networks impedes analytical verification of such models or controllers.  To this end, we present a novel safety verification framework for model-based RL controllers using reachable set analysis.  The proposed framework can efficiently handle models and controllers which are represented using neural networks. Additionally, if a controller fails to satisfy the safety constraints in general, the proposed framework can also be used to identify the subset of initial states from which the controller can be safely executed.

## Automatic Data Augmentation for Generalization in Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'generalization', 'data augmentation']

Deep reinforcement learning (RL) agents often fail to generalize beyond their training environments. To alleviate this problem, recent work has proposed the use of data augmentation. However, different tasks tend to benefit from different types of augmentations and selecting the right one typically requires expert knowledge. In this paper, we introduce three approaches for automatically finding an effective augmentation for any RL task. These are combined with two novel regularization terms for the policy and value function, required to make the use of data augmentation theoretically sound for actor-critic algorithms. We evaluate our method on the Procgen benchmark which consists of 16 procedurally generated environments and show that it improves test performance by 40% relative to standard RL algorithms. Our approach also outperforms methods specifically designed to improve generalization in RL, thus setting a new state-of-the-art on Procgen. In addition, our agent learns policies and representations which are more robust to changes in the environment that are irrelevant for solving the task, such as the background. 

## What are the Statistical Limits of Batch RL with Linear Function Approximation?

Authors: ['Anonymous']

Keywords: ['batch reinforcement learning', 'function approximation', 'lower bound', 'representation']

Function approximation methods coupled with batch reinforcement learning (or off-policy reinforcement learning) are providing an increasingly important framework to help alleviate the excessive sample complexity burden in modern reinforcement learning problems. However, the extent to which function approximation, when coupled with off-policy data, can be effective is not well understood, where the literature largely consists of \emph{sufficient} conditions.
This work focuses on the basic question: what are \emph{necessary} representational and distributional conditions that permit provable sample-efficient off-policy RL? Perhaps surprisingly, our main result shows even if 1) we have \emph{realizability} in that the true value function of our target policy has a linear representation in a given set of features and 2) our off-policy data has good \emph{coverage} over all these features (in a precisely defined and strong sense), any algorithm information-theoretically still requires an exponential number of off-policy samples to non-trivially estimate the value of the target policy. Our results highlight that sample-efficient, batch RL is not guaranteed unless significantly stronger conditions, such as the distribution shift is sufficiently mild (which we precisely characterize) or representation conditions that are far stronger than realizability, are met.    

## Formal Language Constrained Markov Decision Processes

Authors: ['Anonymous']

Keywords: ['safe reinforcement learning', 'formal languages', 'constrained Markov decision process', 'safety gym', 'safety']

In order to satisfy safety conditions, an agent may be constrained from acting freely. A safe controller can be designed a priori if an environment is well understood, but not when learning is employed. In particular, reinforcement learned (RL) controllers require exploration, which can be hazardous in safety critical situations. We study the benefits of giving structure to the constraints of a constrained Markov decision process by specifying them in formal languages as a step towards using safety methods from software engineering and controller synthesis. We instantiate these constraints as finite automata to efficiently recognise constraint violations. Constraint states are then used to augment the underlying MDP state and to learn a dense cost function, easing the problem of quickly learning joint MDP/constraint dynamics. We empirically evaluate the effect of these methods on training a variety of RL algorithms over several constraints specified in Safety Gym, MuJoCo, and Atari environments.

## Behavioral Cloning from Noisy Demonstrations

Authors: ['Anonymous']

Keywords: ['Imitation Learning', 'Inverse Reinforcement Learning', 'Noisy Demonstrations']

We consider the problem of learning an optimal expert behavior policy given noisy demonstrations that contain observations from both optimal and non-optimal expert behaviors. Popular imitation learning algorithms, such as generative adversarial imitation learning, assume that (clear) demonstrations are given from optimal expert policies but the non-optimal ones, and thus often fail to imitate the optimal expert behaviors given the noisy demonstrations. Prior works that address the problem require (1) learning policies through environment interactions in the same fashion as reinforcement learning, and (2) annotating each demonstration with confidence scores or rankings. However, such environment interactions and annotations in real-world settings take impractically long training time and a significant human effort. In this paper, we propose an imitation learning algorithm to address the problem without any environment interactions and annotations associated with the non-optimal demonstrations. The proposed algorithm learns ensemble policies with a generalized behavioral cloning (BC) objective function where we exploit another policy already learned by BC. Experimental results show that the proposed algorithm can learn behavior policies that are much closer to the optimal policies than ones learned by BC.

## Reinforcement Learning with Random Delays

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Deep Reinforcement Learning']

Action and observation delays commonly occur in many Reinforcement Learning applications, such as remote control scenarios. We study the anatomy of randomly delayed environments, and show that partially resampling trajectory fragments in hindsight allows for off-policy multi-step value estimation. We apply this principle to derive Delay-Correcting Actor-Critic (DCAC), an algorithm based on Soft Actor-Critic with significantly better performance in environments with delays. This is shown theoretically and also demonstrated practically on a delay-augmented version of the MuJoCo continuous control benchmark.

## On Proximal Policy Optimization's Heavy-Tailed Gradients 

Authors: ['Anonymous']

Keywords: ['Heavy-tailed Gradients', 'Proximal Policy Optimization', 'Robust Estimation', 'Deep Reinforcement Learning']

Modern policy gradient algorithms, notably Proximal Policy Optimization (PPO), rely on an arsenal of heuristics, including loss clipping and gradient clipping, to ensure successful learning. These heuristics are reminiscent of techniques from robust statistics, commonly used for estimation in outlier-rich ("heavy-tailed") regimes. In this paper, we present a detailed empirical study to characterize the heavy-tailed nature of the gradients of the PPO surrogate reward function. 
We demonstrate pronounced heavy-tailedness of the gradients, specifically for the actor network,  which increases as the current policy diverges from the behavioral one (i.e., as the agent goes further off policy).  Further examination implicates the likelihood ratios and advantages in the surrogate reward as the main sources to the observed heavy-tailedness. Subsequently, we study the effects of the standard PPO clipping heuristics, demonstrating how these tricks primarily serve to offset heavy-tailedness in gradients. Motivated by these connections, we propose incorporating GMOM (a high-dimensional robust estimator) into PPO as a substitute for three clipping tricks, achieving performance close to PPO (with all heuristics enabled) on a battery of MuJoCo continuous control tasks. 

## Learning to Reason in Large Theories without Imitation

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'thoerem proving', 'exploration', 'mathematics']

In this paper, we demonstrate how to do automated higher-order logic theorem proving in the presence of a large knowledge base of potential premises without learning from human proofs. We augment the exploration of premises based on a simple tf-idf (term frequency-inverse document frequency) based lookup in a deep reinforcement learning scenario. Our experiments show that our theorem prover trained with this exploration mechanism but no human proofs, dubbed DeepHOL Zero, outperforms provers that are trained only on human proofs. It approaches the performance of a prover trained by a combination of imitation and reinforcement learning. We perform multiple experiments to understand the importance of the underlying assumptions that make our exploration approach work, thus explaining our design choices.

## Offline policy selection under Uncertainty

Authors: ['Anonymous']

Keywords: ['Off-policy selection', 'reinforcement learning', 'Bayesian inference']

The presence of uncertainty in policy evaluation  significantly complicates the process of policy ranking and selection in real-world settings. We formally consider offline policy selection as learning preferences over a set of policy prospects given a fixed experience dataset. While one can select or rank policies based on point estimates of their policy values or high-confidence intervals, access to the full distribution over one's belief of the policy value enables more flexible selection algorithms under a wider range of downstream evaluation metrics. We propose BayesDICE for estimating this belief distribution in terms of posteriors of distribution correction ratios derived from stochastic constraints (as opposed to explicit likelihood, which is not available). Empirically, BayesDICE is highly competitive to existing state-of-the-art approaches in confidence interval estimation. More importantly, we show how the belief distribution estimated by BayesDICE may be used to rank policies with respect to any arbitrary downstream policy selection metric, and we empirically demonstrate that this selection procedure significantly outperforms existing approaches, such as ranking policies according to mean or high-confidence lower bound value estimates.

## Molecule Optimization by Explainable Evolution

Authors: ['Anonymous']

Keywords: ['Molecule Design', 'Explainable Model', 'Evolutionary Algorithm', 'Reinforcement Learning', 'Graph Generative Model']

Optimizing molecules for desired properties is a fundamental yet challenging task in chemistry, material science and drug discovery. In this paper, we develop a novel algorithm for optimizing molecule properties via an Expectation Maximization~(EM)-like explainable evolutionary process. Our algorithm is designed to mimic human experts in the process of searching for desirable molecules and alternate between two stages: the first stage on explainable local search which identifies rationales, \ie{}, critical subgraph patterns accounting for desired molecular properties, and the second stage on molecule completion which explores the larger space of molecules containing good rationales. We test our method against various baselines on a real-world multi-property optimization task where each method is given the same number of queries to the property oracle. We show that our evolution-by-explanation algorithm is 79\% better than the best baseline in terms of a generic metric combining aspects such as success rate, novelty and diversity. Human expert evaluation on optimized molecules shows that 60\% of top molecules obtained from our methods are deemed as successful ones. 

## Learning to Plan Optimistically: Uncertainty-Guided Deep Exploration via Latent Model Ensembles

Authors: ['Anonymous']

Keywords: ['Model-Based Reinforcement Learning', 'Deep Exploration', 'Continuous Visual Control', 'UCB', 'Latent Space', 'Ensembling']

Learning complex behaviors through interaction requires coordinated long-term planning. Random exploration and novelty search lack task-centric guidance and waste effort on non-informative interactions. Instead, decision making should target samples with the potential to optimize performance far into the future, while only reducing uncertainty where conducive to this objective. This paper presents latent optimistic value exploration (LOVE), a strategy that enables deep exploration through optimism in the face of uncertain long-term rewards. We combine finite-horizon rollouts from a latent model with value function estimates to predict infinite-horizon returns and recover associated uncertainty through ensembling. Policy training then proceeds on an upper confidence bound (UCB) objective to identify and select the interactions most promising to improve long-term performance. We apply LOVE to continuous visual control tasks and demonstrate improved sample complexity on a selection of benchmarking tasks. 

## A Primal Approach to Constrained Policy Optimization: Global Optimality and Finite-Time Analysis

Authors: ['Anonymous']

Keywords: ['safe reinforcement learning', 'constrained markov decision process', 'policy optimization']

Safe reinforcement learning (SRL) problems are typically modeled as constrained Markov Decision Process (CMDP), in which an agent explores the environment to maximize the expected total reward and meanwhile avoids violating certain constraints on a number of expected total costs. In general, such SRL problems have nonconvex objective functions subject to multiple nonconvex constraints, and hence are very challenging to solve, particularly to provide a globally optimal policy. Many popular SRL algorithms adopt a primal-dual structure which utilizes the updating of dual variables for satisfying the constraints. In contrast, we propose a primal approach, called constraint-rectified policy optimization (CRPO), which updates the policy alternatingly between objective improvement and constraint satisfaction. CRPO provides a primal-type algorithmic framework to solve SRL problems, where each policy update can take any variant of policy optimization step. To demonstrate the theoretical performance of CRPO, we adopt natural policy gradient (NPG) for each policy update step and show that CRPO achieves an $\mathcal{O}(1/\sqrt{T})$ convergence rate to the global optimal policy in the constrained policy set and an $\mathcal{O}(1/\sqrt{T})$ error bound on constraint satisfaction. This is the first finite-time analysis of SRL algorithms with global optimality guarantee. Our empirical results demonstrate that CRPO can outperform the existing primal-dual baseline algorithms significantly.

## Learning Latent Landmarks for Generalizable Planning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Planning']

Planning - the ability to analyze the structure of a problem in the large and decompose it into interrelated subproblems - is a hallmark of human intelligence. While deep reinforcement learning (RL) has shown great promise for solving relatively straightforward control tasks, it remains an open problem how to best incorporate planning into existing deep RL paradigms to handle increasingly complex environments. This problem seems difficult because planning requires an agent to reason over temporally extended periods. In principle, however, planning can take advantage of the higher-level structure of a complex problem - freeing the lower-level policy to focus on learning simple behaviors. In this work, we leverage the graph structure inherent to MDPs to address the problem of planning in RL. Rather than learning a graph over a collection of visited states, we learn latent landmarks that are scattered - in terms of reachability - across the goal space to provide state and temporal abstraction. On a variety of high-dimensional continuous control tasks, we demonstrate that our method outperforms prior work, and is oftentimes the only method capable of leveraging both the robustness of model-free RL and generalization of graph-search algorithms. We believe our work is an important step towards scalable planning in the reinforcement learning setting.

## C-Learning: Horizon-Aware Cumulative Accessibility Estimation

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'goal reaching', 'Q-learning']

Multi-goal reaching is an important problem in reinforcement learning needed to achieve algorithmic generalization. Existing methods suffer from the fundamental limitation of learning a single way to reach each goal from each state. This does not allow for any trade-off between reaching the goal quickly and reaching it reliably. We introduce the concept of cumulative accessibility functions to address this limitation. We show that these functions obey a recurrence relation which enables learning from offline interactions, and that they allow us to balance the speed / reliability trade-off at test time. In addition, these functions are monotonic in the planning horizon which decreases the complexity of the learning task. We propose a specially tailored replay buffer that allows efficient learning, validate our approach in a variety of experiments, and demonstrate that it allows the speed / reliability trade-off while outperforming state-of-the-art goal reaching algorithms in planning tasks. Additional visualizations can be found at https://sites.google.com/view/learning-cae/

## Grounding Language to Entities for Generalization in Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'language grounding']

In this paper, we consider the problem of leveraging textual descriptions to improve generalization of control policies to new scenarios. Unlike prior work in this space, we do not assume access to any form of prior knowledge connecting text and state observations, and learn both symbol grounding and control policy simultaneously. This is challenging due to a lack of concrete supervision, and incorrect groundings can result in worse performance than policies that do not use the text at all. 
We develop a new model, EMMA (Entity Mapper with Multi-modal Attention) which uses a multi-modal entity-conditioned attention module that allows for selective focus over relevant sentences in the manual for each entity in the environment. EMMA is end-to-end differentiable and can learn a latent grounding of entities and dynamics from text to observations using environment rewards as the only source of supervision.
To empirically test our model, we design a new framework of 1320 games and collect text manuals with free-form natural language via crowd-sourcing. We demonstrate that EMMA achieves successful zero-shot generalization to unseen games with new dynamics, obtaining significantly higher rewards compared to multiple baselines. The grounding acquired by EMMA is also robust to noisy descriptions and linguistic variation.

## D4RL: Datasets for Deep Data-Driven Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'deep learning', 'benchmarks']

The offline reinforcement learning (RL) problem, also known as batch RL, refers to the setting where a policy must be learned from a static dataset, without additional online data collection. This setting is compelling as it potentially allows RL methods to take advantage of large, pre-collected datasets, much like how the rise of large datasets has fueled results in supervised learning in recent years. However, existing online RL benchmarks are not tailored towards the offline setting, making progress in offline RL difficult to measure. In this work, we introduce benchmarks specifically designed for the offline setting, guided by key properties of datasets relevant to real-world applications of offline RL. Examples of such properties include: datasets generated via hand-designed controllers and human demonstrators, multi-objective datasets where an agent can perform different tasks in the same environment, and datasets consisting of a mixtures of policies. To facilitate research, we release our benchmark tasks and datasets with a comprehensive evaluation of existing algorithms and an evaluation protocol together with an open-source codebase. We hope that our benchmark will focus research effort on methods that drive improvements not just on simulated tasks, but ultimately on the kinds of real-world problems where offline RL will have the largest impact. 

## Non-asymptotic Confidence Intervals of Off-policy Evaluation:  Primal and Dual Bounds 

Authors: ['Anonymous']

Keywords: ['Non-asymptotic Confidence Intervals', 'Off Policy Evaluation', 'Reinforcement Learnings']

Off-policy evaluation remains a central challenge in applying reinforcement learning to real-world observational data. Constructing confidence intervals of off-policy estimators can be even more challenging. However, in high-stakes areas such as medical treatment, it is essential to provide rigorous confidence bounds, not just a point estimation, for safety concerns. In this work, we propose a novel approach to constructing non-asymptotic confidence intervals of off-policy evaluation. Our algorithm is highly flexible and can be applied to infinite-horizon reinforcement learning problems with behavior-agnostic data, consisting of dependent transition pairs collected in an arbitrary way. The key idea of our method is taking an optimization-based approach to confidence bounds through a new primal-dual lens, and leveraging a new tight concentration inequality applicable to dependent data. We further present empirical results that clearly demonstrate the advantages of our approach over existing methods. 

## Measuring and mitigating interference in reinforcement learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Representation Learning']

Catastrophic interference is common in many network-based learning systems, and many proposals exist for mitigating it. But, before we overcome interference we must understand it better. In this work, we first provide a definition and novel measure of interference for value-based control methods such as Fitted Q Iteration and DQN. We systematically evaluate our measure of interference, showing that it correlates with forgetting, across a variety of network architectures. Our new interference measure allows us to ask novel scientific questions about commonly used deep learning architectures and develop new learning algorithms. In particular we show that updates on the last layer result in significantly higher interference than updates internal to the network. Lastly, we introduce a novel online-aware representation learning algorithm to minimize interference, and we empirically demonstrate that it improves stability and has lower interference.

## Shortest-Path Constrained Reinforcement Learning for Sparse Reward Tasks

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'exploration', 'sample efficient reinforcement learning', 'sparse rewards']

We propose the k-Shortest-Path (k-SP) constraint: a novel constraint on the agent’s trajectory that improves the sample-efficiency in sparse-reward MDPs. We show that any optimal policy necessarily satisfies the k-SP constraint. Notably, the k-SP constraint prevents the policy from exploring state-action pairs along the non-k-SP trajectories (e.g., going back and forth).  However, in practice, excluding state-action pairs may hinder convergence of many RL algorithms. To overcome this, we propose a novel cost function that penalizes the policy violating SP constraint, instead of completely excluding it.  Our numerical experiment in a tabular RL setting demonstrate that the SP constraint can significantly reduce the trajectory space of policy. As a result, our constraint enables more sample efficient learning by suppressing redundant exploration and exploitation. Our empirical experiment results on MiniGrid and DeepMind Lab show that the proposed method significantly improves proximal policy optimization (PPO) and outperforms existing novelty-seeking exploration methods including count-based exploration, indicating that it improves the sample efficiency by preventing the agent from taking redundant actions.

## BeBold: Exploration Beyond the Boundary of Explored Regions

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'exploration']

Efficient exploration under sparse rewards remains a key challenge in deep reinforcement learning. To guide exploration, previous work makes extensive use of intrinsic reward (IR). There are many heuristics for IR, including visitation counts, curiosity, and state-difference. In this paper, we analyze the pros and cons of each method and propose the regulated difference of inverse visitation counts as a simple but effective criterion for IR. The criterion helps the agent explore Beyond the Boundary of explored regions and mitigates common issues in count-based methods, such as short-sightedness and detachment. The resulting method, BeBold, solves the 12 most challenging procedurally-generated tasks in MiniGridwith just 120M environment steps, without any curriculum learning. In comparison, previous SoTA only solves 50%of the tasks. BeBold also achieves SoTAon multiple tasks in NetHack, a popular rogue-like game that contains more challenging procedurally-generated environments.

## D3C: Reducing the Price of Anarchy in Multi-Agent Learning

Authors: ['Anonymous']

Keywords: ['multiagent', 'social dilemma', 'reinforcement learning']

Even in simple multi-agent systems, fixed incentives can lead to outcomes that are poor for the group and each individual agent. We propose a method, D3C, for online adjustment of agent incentives that reduces the loss incurred at a Nash equilibrium. Agents adjust their incentives by learning to mix their incentive with that of other agents, until a compromise is reached in a distributed fashion. We show that D3C improves outcomes for each agent and the group as a whole in several social dilemmas including a traffic network with Braess’s paradox, a prisoner’s dilemma, and several reinforcement learning domains.

## Beyond Prioritized Replay: Sampling States in Model-Based RL via Simulated Priorities

Authors: ['Anonymous']

Keywords: ['Experience replay', 'prioritized sampling', 'model-based reinforcement learning', 'Dyna architecture']

Model-based reinforcement learning (MBRL) can significantly improve sample efficiency, particularly when carefully choosing the states from which to sample hypothetical transitions. Such prioritization has been empirically shown to be useful for both experience replay (ER) and Dyna-style planning. However, there is as yet little theoretical understanding in RL about such prioritization strategies, and why they help. In this work, we revisit prioritized ER and, in an ideal setting, show equivalence to minimizing cubic loss, providing theoretical insight into why it improves upon uniform sampling. This theoretical equivalence highlights two limitations of current prioritized experience replay methods: insufficient coverage of the sample space and outdated priorities of training samples. This motivates our model-based approach, which does not suffer from these limitations. Our key idea is to actively search for high priority states using gradient ascent. Under certain conditions, we prove that the hypothetical experiences generated from these states are sampled proportionally to approximately true priorities. We also characterize the distance between the sampling distribution of our method and the true prioritized sampling distribution. Our experiments on both benchmark and application-oriented domains show that our approach achieves superior performance over baselines. 

## Playing Nondeterministic Games through Planning with a Learned Model

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'alphazero', 'muzero', 'mcts', 'planning', 'search']

The MuZero algorithm is known for achieving high-level performance on traditional zero-sum two-player games of perfect information such as chess, Go, and shogi, as well as visual, non-zero sum, single-player environments such as the Atari suite. Despite lacking a perfect simulator and employing a learned model of environmental dynamics,  MuZero produces game-playing agents comparable to its predecessor AlphaZero. However, the current implementation of MuZero is restricted only to deterministic environments. This paper presents Nondeterministic MuZero (NDMZ), an extension of MuZero for nondeterministic, two-player, zero-sum games of perfect information. Borrowing from Nondeterministic Monte Carlo Tree Search and the theory of extensive-form games, NDMZ formalizes chance as a player in the game and incorporates it into the MuZero network architecture and tree search. Experiments show that NDMZ is capable of learning effective strategies and an accurate model of the game.

## Weighted Bellman Backups for Improved Signal-to-Noise in Q-Updates

Authors: ['Anonymous']

Keywords: ['Deep reinforcement learning', 'ensemble learning', 'Q-learning']

Off-policy deep reinforcement learning (RL) has been successful in a range of challenging domains. However, standard off-policy RL algorithms can suffer from low signal and even instability in Q-learning because target values are derived from current Q-estimates, which are often noisy. To mitigate the issue, we propose ensemble-based weighted Bellman backups, which re-weight target Q-values based on uncertainty estimates from a Q-ensemble. We empirically observe that the proposed method stabilizes and improves learning on both continuous and discrete control benchmarks. We also specifically investigate the signal-to-noise aspect by studying environments with noisy rewards, and find that weighted Bellman backups significantly outperform standard Bellman backups. Furthermore, since our weighted Bellman backups rely on maintaining an ensemble, we investigate how weighted Bellman backups interact with other benefits previously derived from ensembles: (a) Bootstrap; (b) UCB Exploration. We show that these different ideas are largely orthogonal and can be fruitfully integrated, together further improving the performance of existing off-policy RL algorithms, such as Soft Actor-Critic and Rainbow DQN, for both continuous and discrete control tasks on both low-dimensional and high-dimensional environments.

## Leaky Tiling Activations: A Simple Approach to Learning Sparse Representations Online

Authors: ['Anonymous']

Keywords: ['Reinforcement learning', 'sparse representation', 'leaky tiling activation functions']

Recent work has shown that sparse representations---where only a small percentage of units are active---can significantly reduce interference. Those works, however, relied on relatively complex regularization or meta-learning approaches, that have only been used offline in a pre-training phase. We design an activation function that naturally produces sparse representations, and so is more amenable to online training. The idea relies on the simple approach of binning, but overcomes the two key limitations of binning: zero gradients for the flat regions almost everywhere,  and lost precision---reduced discrimination---due to coarse aggregation. We introduce a Leaky Tiling Activation (LTA) that provides non-negligible gradients and produces overlap between bins that improves discrimination. We first show that LTA is robust under covariate shift in a synthetic online supervised problem, where we can vary the level of correlation and drift. Then we move to deep reinforcement learning setting and investigate both value-based and policy gradient algorithms that use neural networks with LTAs, in classic discrete control and Mujoco continuous control environments. We show that algorithms equipped with LTAs are able to learn a stable policy faster without needing target networks on most domains. 

## Revisiting Parameter Sharing in Multi-Agent Deep Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Multi-agent Reinforcement Learning']

"``Nonstationarity" is a fundamental problem in cooperative multi-agent reinforcement learning (MARL). It results from every agent's policy changing during learning, while being part of the environment from the perspective of other agents. This causes information to inherently oscillate between agents during learning, greatly slowing convergence. We use the MAILP model of information transfer during multi-agent learning to show that increasing centralization during learning arbitrarily mitigates the slowing of convergence due to nonstationarity. The most centralized case of learning is parameter sharing, an uncommonly used MARL method, specific to environments with homogeneous agents. It bootstraps single-agent reinforcement learning (RL) methods and learns an identical policy for each agent. We experimentally replicate our theoretical result of increased learning centralization leading to better performance. We further apply parameter sharing to 8 more modern single-agent deep RL methods for the first time, achieving up to 44 times more average reward in 16% as many episodes compared to previous parameter sharing experiments. We finally give a formal proof of a set of methods that allow parameter sharing to serve in environments with heterogeneous agents.

## Self-Activating Neural Ensembles for Continual Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['continual reinforcement learning', 'lifelong learning', 'deep reinforcement learning']

The ability for an agent to continuously learn new skills without catastrophically forgetting existing knowledge is of critical importance for the development of generally intelligent agents. Most methods devised to address this problem depend heavily on well-defined task boundaries which simplify the problem considerably. Our task-agnostic method, Self-Activating Neural Ensembles (SANE), uses a hierarchical modular architecture designed to avoid catastrophic forgetting without making any such assumptions. At each timestep a path through the SANE tree is activated; during training only activated nodes are updated, ensuring that unused nodes do not undergo catastrophic forgetting. Additionally, new nodes are created as needed, allowing the system to leverage and retain old skills while growing and learning new ones. We demonstrate our approach on MNIST and a set of grid world environments, demonstrating that SANE does not undergo catastrophic forgetting where existing methods do.

## Aspect-based Sentiment Classification via Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Sentiment classification', 'reinforcement learning']

Aspect-based sentiment classification aims to predict sentimental polarities of one or multiple aspects in texts. As texts always contain a large proportion of task-irrelevant words, accurate alignment between aspects and their sentimental descriptions is the most crucial and challenging step. State-of-the-art approaches are mainly based on word-level attention learned from recurrent neural network variants (e.g., LSTM) or graph neural networks. From another view, these methods essentially weight and aggregate all possible alignments. However, this mechanism heavily relies on large-scale supervision training: without enough labels, it could easily overfit with difficulty in generalization. To address this challenge, we propose SentRL, a reinforcement learning-based framework for aspect-based sentiment classification. In this framework, input texts are transformed into their dependency graphs. Then, an agent is deployed to walk on the graphs, explores paths from target aspect nodes to their potential sentimental regions, and differentiates the effectiveness of different paths. By limiting the agent's exploration budget, our method encourages the agent to skip task-irrelevant information and focus on the most effective paths for alignment purpose. Our method considerably reduces the impact of task-irrelevant words and improves generalization performance. Compared with competitive baseline methods, our approach achieves the highest performance on public benchmark datasets with up to $3.7\%$ improvement. 

## Regularization Matters in Policy Optimization - An Empirical Study on Continuous Control

Authors: ['Anonymous']

Keywords: ['Policy Optimization', 'Regularization', 'Continuous Control', 'Deep Reinforcement Learning']

Deep Reinforcement Learning (Deep RL) has been receiving increasingly more attention  thanks to its encouraging performance on a variety of control tasks. Yet, conventional regularization techniques in training neural networks (e.g., $L_2$ regularization, dropout) have been largely ignored in RL methods, possibly because agents are typically trained and evaluated in the same environment, and because the deep RL community focuses more on high-level algorithm designs. In this work, we present the first comprehensive study of regularization techniques with multiple policy optimization algorithms on continuous control tasks. Interestingly, we find conventional regularization techniques on the policy networks can often bring large improvement, especially on harder tasks. Our findings are shown to be robust against training hyperparameter variations. We also compare these techniques with the more widely used entropy regularization. In addition, we study regularizing different components and find that only regularizing the policy network is typically the best. We further analyze why regularization may help generalization in RL from four perspectives - sample complexity, reward distribution, weight norm, and noise robustness. We hope our study provides guidance for future practices in regularizing policy optimization algorithms. Our code is available at https://github.com/anonymouscode114/iclr2021_rlreg .

## PettingZoo: Gym for Multi-Agent Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Multi-agent Reinforcement Learning']

OpenAI's Gym library  contains a large, diverse set of environments that are useful benchmarks in reinforcement learning, under a single elegant Python API (with tools to develop new compliant environments) . The introduction of this library has proven a watershed moment for the reinforcement learning community, because it created an accessible set of benchmark environments that everyone could use (including wrapper important existing libraries), and because a standardized API let RL learning methods and environments from anywhere be trivially exchanged. This paper similarly introduces PettingZoo, a library of diverse set of multi-agent environments under a single elegant Python API, with tools to easily make new compliant environments.

## Neuro-algorithmic Policies for Discrete Planning

Authors: ['Anonymous']

Keywords: ['planning', 'reinforcement learning', 'combinatorial optimization', 'control', 'imitation learning']

Although model-based and model-free approaches to learning control of systems have achieved impressive results on standard benchmarks, most have been shown to be lacking in their generalization capabilities. These methods usually require sampling an exhaustive amount of data from different environment configurations.

We introduce a neuro-algorithmic policy architecture with the ability to plan consisting of a model working in unison with a shortest path solver to predict trajectories with low way-costs. These policies can be trained end-to-end by blackbox differentiation. We show that this type of hybrid architectures generalize well to unseen environment configurations.

https://sites.google.com/view/neuro-algorithmic

## Exploring Zero-Shot Emergent Communication in Embodied Multi-Agent Populations

Authors: ['Anonymous']

Keywords: ['emergent communication', 'multi-agent communication', 'multi-agent reinforcement learning']

Effective communication is an important skill for enabling information exchange and cooperation in multi-agent settings. Indeed, emergent communication is now a vibrant field of research, with common settings involving discrete cheap-talk channels. One limitation of this setting is that it does not allow for the emergent protocols to generalize beyond the training partners. 
 Furthermore, so far emergent communication has primarily focused on the use of symbolic channels. In this work, we extend this line of work to a new modality, by studying agents that learn to communicate via actuating their joints in a 3D environment. We show that under realistic assumptions, a non-uniform distribution of intents and a common-knowledge energy cost, these agents can find protocols that generalize to novel partners. We also explore and analyze specific difficulties associated with finding these solutions in practice. Finally, we propose and evaluate initial training improvements to address these challenges, involving both specific training curricula and providing the latent feature that can be coordinated on during training.

## Near-Optimal Regret Bounds for Model-Free RL in Non-Stationary Episodic MDPs

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'non-stationary environment', 'model-free approach', 'regret analysis']

We consider model-free reinforcement learning (RL) in non-stationary Markov decision processes (MDPs). Both the reward functions and the state transition distributions are allowed to vary over time, either gradually or abruptly, as long as their cumulative variation magnitude does not exceed certain budgets. We propose an algorithm, named Restarted Q-Learning with Upper Confidence Bounds (RestartQ-UCB), for this setting, which adopts a simple restarting strategy and an extra optimism term. Our algorithm outperforms the state-of-the-art (model-based) solution in terms of dynamic regret. Specifically, RestartQ-UCB with Freedman-type bonus terms achieves a dynamic regret of $\widetilde{O}(S^{\frac{1}{3}} A^{\frac{1}{3}} \Delta^{\frac{1}{3}} H T^{\frac{2}{3}})$, where $S$ and $A$ are the numbers of states and actions, respectively, $\Delta>0$ is the variation budget, $H$ is the number of steps per episode, and $T$ is the total number of steps. We further show that our algorithm is near-optimal by establishing an information-theoretical lower bound of $\Omega(S^{\frac{1}{3}} A^{\frac{1}{3}} \Delta^{\frac{1}{3}} H^{\frac{2}{3}} T^{\frac{2}{3}})$, which to the best of our knowledge is the first impossibility result in non-stationary RL in general.

## Communication in Multi-Agent Reinforcement Learning: Intention Sharing

Authors: ['Anonymous']

Keywords: ['Multi-agent reinforcement learning', 'communication', 'intention', 'attention']

Communication is one of the core components for learning coordinated behavior in multi-agent systems.
In this paper, we propose a new communication scheme named  Intention Sharing (IS) for multi-agent reinforcement learning in order to enhance the coordination among agents. In the proposed IS scheme, each agent generates an imagined trajectory by modeling the environment dynamics and other agents' actions. The imagined trajectory is the simulated future trajectory of each agent based on the learned model of the environment dynamics and other agents and represents each agent's future action plan. Each agent compresses this imagined trajectory capturing its future action plan to generate its intention message for communication by applying an attention mechanism to learn the relative importance of the components in the imagined trajectory based on the received message from other agents. Numeral results show that the proposed IS scheme outperforms other communication schemes in multi-agent reinforcement learning.

## A Maximum Mutual Information Framework for Multi-Agent Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Multi-agent reinforcement learning', 'coordination', 'mutual information']

In this paper, we propose a maximum mutual information (MMI) framework for multi-agent reinforcement learning (MARL) to enable multiple agents to learn coordinated behaviors by regularizing the accumulated return with the mutual information between actions. By introducing a latent variable to induce nonzero mutual information between actions and applying a variational bound, we derive a tractable lower bound on the considered MMI-regularized objective function. Applying policy iteration to maximize the derived lower bound, we propose a practical algorithm named variational maximum mutual information multi-agent actor-critic (VM3-AC), which follows centralized learning with decentralized execution (CTDE). We evaluated VM3-AC for several games requiring coordination, and numerical results show that VM3-AC outperforms MADDPG and other MARL algorithms in multi-agent tasks requiring coordination.

## Randomized Entity-wise Factorization for Multi-Agent Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['MARL', 'multi-agent reinforcement learning', 'value function factorization', 'attention']

Real world multi-agent tasks often involve varying types and quantities of agents and non-agent entities; however, agents within these tasks rarely need to consider all others at all times in order to act effectively. Factored value function approaches have historically leveraged such independences to improve learning efficiency, but these approaches typically rely on domain knowledge to select fixed subsets of state features to include in each factor. We propose to utilize value function factoring with random subsets of entities in each factor as an auxiliary objective in order to disentangle value predictions from irrelevant entities. This factoring approach is instantiated through a simple attention mechanism masking procedure. We hypothesize that such an approach helps agents learn more effectively in multi-agent settings by discovering common trajectories across episodes within sub-groups of agents/entities. Our approach, Randomized Entity-wise Factorization for Imagined Learning (REFIL), outperforms all strong baselines by a significant margin in challenging StarCraft micromanagement tasks.

## Dropout's Dream Land: Generalization from Learned Simulators to Reality

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning']

A World Model is a generative model used to simulate an environment. World Models have proven capable of learning spatial and temporal representations of Reinforcement Learning environments. In some cases, a World Model offers an agent the opportunity to learn entirely inside of its own dream environment. In this work we explore improving the generalization capabilities from dream environments to reality (Dream2Real). We present a general approach to improve a controller's ability to transfer from a neural network dream environment to reality at little additional cost. These improvements are gained by drawing on inspiration from domain randomization, where the basic idea is to randomize as much of a simulator as possible without fundamentally changing the task at hand. Generally, domain randomization assumes access to a pre-built simulator with configurable parameters but oftentimes this is not available. By training the World Model using dropout, the dream environment is capable of creating a nearly infinite number of \textit{different} dream environments. Our experimental results show that Dropout's Dream Land is an effective technique to bridge the reality gap between dream environments and reality. Furthermore, we additionally perform an extensive set of ablation studies. 

## Attention-driven Robotic Manipulation

Authors: ['Anonymous']

Keywords: ['Robotics', 'Robot Manipulation', 'Reinforcement Learning']

Despite the success of reinforcement learning methods, they have yet to have their breakthrough moment when applied to a broad range of robotic manipulation tasks. This is partly due to the fact that reinforcement learning algorithms are notoriously difficult and time consuming to train, which is exacerbated when training from images rather than full-state inputs. As humans perform manipulation tasks, our eyes closely monitor every step of the process with our gaze focusing sequentially on the objects being manipulated. With this in mind, we present our Attention-driven Robotic Manipulation (ARM) algorithm, which is a general manipulation algorithm that can be applied to a range of real-world sparse-rewarded tasks without any prior task knowledge. ARM splits the complex task of manipulation into a 3 stage pipeline: (1) a Q-attention agent extracts interesting pixel locations from RGB and point cloud inputs, (2) a next-best pose agent that accepts crops from the Q-attention agent and outputs poses, and (3) a control agent that takes the goal pose and outputs joint actions. We show that current state-of-the-art reinforcement learning algorithms catastrophically fail on a range of RLBench tasks, whilst ARM is successful within a few hours.

## Learning to Dynamically Select Between Reward Shaping Signals

Authors: ['Anonymous']

Keywords: ['selection', 'automatic', 'reward', 'shaping', 'reinforcement learning']

Reinforcement learning (RL) algorithms often have the limitation of sample complexity. Previous research has shown that the reliance on large amounts of experience can be mitigated through the presence of additional feedback. Automatic reward shaping is one approach to solving this problem, using automatic identification and modulation of shaping reward signals that are more informative about how agents should behave in any given scenario to learn and adapt faster. However, automatic reward shaping is still very challenging. To better study it, we break it down into two separate sub-problems: learning shaping reward signals in an application and learning how the signals can be adaptively used to provide a single reward feedback in the RL learning process. This paper focuses on the latter sub-problem. Unlike existing research, which tries to learn one shaping reward function from shaping signals, the proposed method learns to dynamically select the right reward signal to apply at each state, which is considerably more flexible. We further show that using an online strategy that seeks to match the learned shaping feedback with optimal value differences can lead to effective reward shaping and accelerated learning. The proposed ideas are verified through experiments in a variety of environments using different shaping reward paradigms.

## Joint State-Action Embedding for Efficient Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'embedding', 'representation learning', 'state-action embedding']

While reinforcement learning has achieved considerable successes in recent years, state-of-the-art models are often still limited by the size of state and action spaces. Model-free reinforcement learning approaches use some form of state representations and the latest work has explored embedding techniques for actions, both with the aim of achieving better generalization and applicability. However, these approaches consider only states or actions, ignoring the interaction between them when generating embedded representations. In this work, we propose a new approach for jointly embedding states and actions that combines aspects of model-free and model-based reinforcement learning, which can be applied in both discrete and continuous domains. Specifically, we use a model of the environment to obtain embeddings for states and actions and present a generic architecture that uses these to learn a policy. In this way, the embedded representations obtained via our approach enable better generalization over both states and actions by capturing similarities in the embedding spaces. Evaluations of our approach on several gaming and recommender system environments show it significantly outperforms state-of-the-art models in discrete domains with large state/action space, thus confirming the efficacy of joint embedding and its overall superior performance.

## A Robust Fuel Optimization Strategy For Hybrid Electric Vehicles: A Deep Reinforcement Learning Based Continuous Time Design Approach

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement Learning', 'Optimal Control', 'Fuel Management System', 'Hybrid Electric vehicles', 'H∞ Performance Index']

This paper deals with the fuel optimization problem for hybrid electric vehicles in deep reinforcement learning framework. Firstly, considering the hybrid electric vehicle as an uncertain non-linear system with
unknown dynamics in continuous time frame, we solve an open-loop deterministic trajectory optimization problem without explicitly considering
the system model. This is followed by the design of a deep reinforcement
learning based optimal control law for the non-linear system (i.e., hybrid
electric vehicles) such that the actual states and the control policy remain
close to the optimal trajectory and optimal policy even in the presence of
external disturbances, modeling errors, uncertainties and noise. The low
value of the H-infinity ($H_{\infty})$ performance index (i.e., the ratio of the disturbance to the control energy) of the RL based optimization technique in
comparison with the traditional methods addresses the robustness issue.
The control strategy will also autonomously learn the optimal policy and
adapt itself to the different conditions which is in sharp contrast to the
conventional techniques that mostly depend on a set of pre-defined rules
and provide sub-optimal solutions to the fuel management problem. The
controller thus designed is compared with the traditional fuel optimization strategies for hybrid electric vehicles to illustate the efficacy of the
proposed method.

## Autoregressive Dynamics Models for Offline Policy Evaluation and Optimization

Authors: ['Anonymous']

Keywords: ['Off-policy policy evaluation', 'autoregressive models', 'offline reinforcement learning', 'policy optimization']

Standard dynamics models for continuous control make use of feedforward computation and assume that different dimensions of the next state and reward are conditionally independent given the current state and action. This modeling choice may be driven by the fact that fully observable physics-based simulation environments entail deterministic transition dynamics. In this paper, we question this conditional independence assumption and propose a family of expressive autoregressive dynamics models that generate different dimensions of the next state and reward sequentially conditioned on previous dimensions. We demonstrate that autoregressive dynamics models indeed outperform standard feedforward models in log-likelihood on heldout trajectories. Furthermore, we compare different model-based and model-free off-policy evaluation (OPE) methods on RL Unplugged, a suite of offline MuJoCo datasets, and find that autoregressive dynamics models consistently outperform all baselines, achieving a new state-of-the-art. Finally, we show that autoregressive dynamics models are useful for offline policy optimization by serving as a way to enrich the replay buffer through data augmentation and improving performance using model-based planning.




## How to Avoid Being Eaten by a Grue: Structured Exploration Strategies for Textual Worlds

Authors: ['Anonymous']

Keywords: ['text-based games', 'reinforcement learning', 'exploration', 'intrinsic motivation', 'knowledge graphs', 'question answering', 'natural language processing']

Text-based games are long puzzles or quests, characterized by a sequence of sparse and potentially deceptive rewards. They provide an ideal platform to develop agents that perceive and act upon the world using a combinatorially sized natural language state-action space. Standard Reinforcement Learning agents are poorly equipped to effectively explore such spaces and often struggle to overcome bottlenecks---states that agents are unable to pass through simply because they do not see the right action sequence enough times to be sufficiently reinforced. We introduce Q*BERT, an agent that learns to build a knowledge graph of the world by answering questions, which leads to greater sample efficiency. To overcome bottlenecks, we further introduce MC!Q*BERT an agent that uses an knowledge-graph-based intrinsic motivation to detect bottlenecks and a novel exploration strategy to efficiently learn a chain of policy modules to overcome them. We present an ablation study and results demonstrating how our method outperforms the current state-of-the-art on nine text games, including the popular game, Zork, where, for the first time, a learning agent gets past the bottleneck where the player is eaten by a Grue.

## How to Motivate Your Dragon: Teaching Goal-Driven Agents to Speak and Act in Fantasy Worlds

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'text-based games', 'goal-driven dialogue', 'natural language processing']

We seek to create agents that both act and communicate with other agents in pursuit of a goal. Towards this end, we extend LIGHT (Urbanek et al. 2019)---a large-scale crowd-sourced fantasy text-game---with a dataset of quests. These contain natural language motivations paired with in-game goals and human demonstrations;  completing a quest might require dialogue or actions (or both). We introduce a reinforcement learning system that (1) incorporates large-scale language modeling-based and commonsense reasoning-based pre-training to imbue the agent with relevant priors; and (2) leverages a factorized action space of action commands and dialogue, balancing between the two. We conduct zero-shot evaluations using held-out human expert demonstrations, showing that our agents are able to act consistently and talk naturally with respect to their motivations.

## Action Guidance: Getting the Best of Sparse Rewards and Shaped Rewards for Real-time Strategy Games

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'real-time strategy games', 'sparse rewards', 'shaped rewards', 'policy gradient', 'sample-efficiency']

Training agents using Reinforcement Learning in games with sparse rewards is a challenging problem, since large amounts of exploration are required to retrieve even the first reward. To tackle this problem, a common approach is to use reward shaping to help exploration. However, an important drawback of reward shaping is that agents sometimes learn to optimize the shaped reward instead of the true objective. In this paper, we present a novel technique that we call action guidance that successfully trains agents to eventually optimize the true objective in games with sparse rewards while maintaining most of the sample efficiency that comes with reward shaping. We evaluate our approach in a simplified real-time strategy (RTS) game simulator called $\mu$RTS. 

## Evaluating Agents Without Rewards

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'task-agnostic', 'agent evaluation', 'exploration', 'information gain', 'empowerment', 'curiosity']

Reinforcement learning has enabled agents to solve challenging control tasks from raw image inputs. However, manually crafting reward functions can be time consuming, expensive, and prone to human error. Competing objectives have been proposed for agents to learn without external supervision, such as artificial curiosity, information gain, and empowerment. Estimating these objectives can be challenging and it remains unclear how well they reflect task rewards or human behavior. We study these objectives across seven agents and three Atari games. Retrospectively computing the objectives from the agent's lifetime of experience simplifies accurate estimation. We find that all three objectives correlate more strongly with human behavior than with the task reward. Moreover, task reward with curiosity better explains human behavior than task reward alone.

## Action and Perception as Divergence Minimization

Authors: ['Anonymous']

Keywords: ['objective functions', 'reinforcement learning', 'information theory', 'probabilistic modeling', 'control as inference', 'exploration', 'intrinsic motivation', 'world models']

We introduce a unified objective for action and perception of intelligent agents. Extending representation learning and control, we minimize the joint divergence between the combined system of agent and environment and a target distribution. Intuitively, such agents use perception to align their beliefs with the world, and use actions to align the world with their beliefs. Minimizing the joint divergence to an expressive target maximizes the mutual information between the agent's representations and inputs, thus inferring representations that are informative of past inputs and exploring future inputs that are informative of the representations. This lets us explain intrinsic objectives, such as representation learning, information gain, empowerment, and skill discovery from minimal assumptions. Moreover, interpreting the target distribution as a latent variable model suggests powerful world models as a path toward highly adaptive agents that seek large niches in their environments, rendering task rewards optional. The framework provides a common language for comparing a wide range of objectives, advances the understanding of latent variables for decision making, and offers a recipe for designing novel objectives. We recommend deriving future agent objectives the joint divergence to facilitate comparison, to point out the agent's target distribution, and to identify the intrinsic objective terms needed to reach that distribution.

## Mastering Atari with Discrete World Models

Authors: ['Anonymous']

Keywords: ['Atari', 'world models', 'model-based reinforcement learning', 'reinforcement learning', 'planning', 'actor critic']

Intelligent agents need to generalize from past experience to achieve goals in complex environments. World models facilitate such generalization and allow learning behaviors from imagined outcomes to increase sample-efficiency. While learning world models from image inputs has recently become feasible for some tasks, modeling Atari games accurately enough to derive successful behaviors has remained an open challenge for many years. We introduce DreamerV2, a reinforcement learning agent that learns behaviors purely from predictions in the compact latent space of a powerful world model. The world model uses discrete representations and is trained separately from the policy. DreamerV2 constitutes the first agent that achieves human-level performance on the Atari benchmark of 55 tasks by learning behaviors inside a separately trained world model. With the same computational budget and wall-clock time, DreamerV2 reaches 200M frames and exceeds the final performance of the top single-GPU agents IQN and Rainbow.

## Approximating Pareto Frontier through Bayesian-optimization-directed Robust Multi-objective Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Multi–objective Optimization', 'Adversarial Machine Learning', 'Bayesian Optimization']

Many real-word decision or control problems involve multiple conflicting objectives and uncertainties, which requires learned policies are not only Pareto optimal but also robust. In this paper, we proposed a novel algorithm to approximate a representation for robust Pareto frontier through Bayesian-optimization-directed robust multi-objective reinforcement learning (BRMORL). Firstly, environmental uncertainty is modeled as an adversarial agent over the entire space of preferences by incorporating zero-sum game into multi-objective reinforcement learning (MORL). Secondly, a comprehensive metric based on hypervolume and information entropy is presented to evaluate convergence, diversity and evenness of the distribution for Pareto solutions. Thirdly, the agent’s learning process is regarded as a black-box, and the comprehensive metric we proposed is computed after each episode of training, then a Bayesian optimization (BO) algorithm is adopted to guide the agent to evolve towards improving the quality of the approximated Pareto frontier. Finally, we demonstrate the effectiveness of proposed approach on challenging high-dimensional multi-objective tasks across several environments, and show our scheme can produce robust policies under environmental uncertainty.

## Reinforcement Learning with Latent Flow

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'deep learning', 'machine learning', 'deep reinforcement learning']

Temporal information is  essential to learning effective policies with Reinforcement Learning (RL). However, current state-of-the-art RL algorithms either assume that such information is given as part of the state space or, when learning from pixels, use the simple heuristic of frame-stacking to implicitly capture temporal information present in the image observations. This heuristic is in contrast to the current paradigm in video classification architectures, which utilize explicit encodings of temporal information through methods such as optical flow and two-stream architectures to achieve state-of-the-art performance. Inspired by leading video classification architectures, we introduce the Flow of Latents for Reinforcement Learning (Flare), a network architecture for RL that explicitly encodes temporal information through latent vector differences. We show that Flare (i) recovers optimal performance in state-based RL without explicit access to the state velocity, solely with positional state information, (ii) achieves state-of-the-art performance on pixel-based continuous control tasks within the DeepMind control benchmark suite, and (iii) is the most sample efficient model-free pixel-based RL algorithm on challenging environments in the DeepMind control suite such as quadruped walk, hopper hop, finger turn hard, pendulum swing, and walker run, outperforming the prior model-free state-of-the-art by 1.9x and 1.5x on the 500k and 1M step benchmarks, respectively.

## Efficient Transformers in Reinforcement Learning using Actor-Learner Distillation

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement Learning', 'Memory', 'Transformers', 'Distillation']

Many real-world applications such as robotics provide hard constraints on power and compute that limit the viable model complexity of Reinforcement Learning (RL) agents. Similarly, in many distributed RL settings, acting is done on un-accelerated hardware such as CPUs, which likewise restricts model size to prevent intractable experiment run times. These "actor-latency" constrained settings present a major obstruction to the scaling up of model complexity that has recently been extremely successful in supervised learning. To be able to utilize large model capacity while still operating within the limits imposed by the system during acting, we develop an "Actor-Learner Distillation" (ALD) procedure that leverages a continual form of distillation that transfers learning progress from a large capacity learner model to a small capacity actor model. As a case study, we develop this procedure in the context of partially-observable environments, where transformer models have had large improvements over LSTMs recently, at the cost of significantly higher computational complexity. With transformer models as the learner and LSTMs as the actor, we demonstrate in several challenging memory environments that using Actor-Learner Distillation recovers the clear sample-efficiency gains of the transformer learner model while maintaining the fast inference and reduced total training time of the LSTM actor model.

## The Logical Options Framework

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'hierarchical methods', 'formal methods', 'formal logic']

Learning composable policies for environments with complex rules and tasks is a challenging problem. We introduce a hierarchical reinforcement learning framework called the Logical Options Framework (LOF) that learns policies that are satisfying, optimal, and composable. LOF efficiently learns policies that satisfy tasks by representing the task as an automaton and integrating it into learning and planning. We provide and prove conditions under which LOF will learn satisfying, optimal policies. And lastly, we show how LOF's learned policies can be composed to satisfy unseen tasks with no additional learning. We evaluate LOF on four tasks in discrete and continuous domains.

## Extracting Strong Policies for Robotics Tasks from zero-order trajectory optimizers

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'zero-order optimization', 'policy learning', 'model-based learning', 'robotics']

Solving high-dimensional, continuous robotic tasks is a challenging optimization problem. Model-based methods that rely on zero-order optimizers like the cross-entropy method (CEM) have so far shown strong performance and are considered state-of-the-art in the model-based reinforcement learning community. However, this success comes at the cost of high computational complexity, being therefore not suitable for real-time control. In this paper, we propose a technique to jointly optimize the trajectory and distill a policy, which is essential for fast execution in real robotic systems. Our method builds upon standard approaches, like guidance cost and dataset aggregation, and introduces a novel adaptive factor which prevents the optimizer from collapsing to the learner's behavior at the beginning of the training. The extracted policies reach unprecedented performance on challenging tasks as making a humanoid stand up and opening a door without reward shaping

## Status-Quo Policy Gradient in Multi-agent Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['multi-agent rl', 'reinforcement learning', 'social dilemma', 'policy gradient', 'game theory']

Individual rationality, which involves maximizing expected individual return, does not always lead to optimal individual or group outcomes in multi-agent problems. For instance, in social dilemma situations, Reinforcement Learning (RL) agents trained to maximize individual rewards converge to mutual defection that is individually and socially sub-optimal. In contrast, humans evolve individual and socially optimal strategies in such social dilemmas. Inspired by ideas from human psychology that attribute this behavior in humans to the status-quo bias, we present a status-quo loss (SQLoss) and the corresponding policy gradient algorithm that incorporates this bias in an RL agent. We demonstrate that agents trained with SQLoss evolve individually as well as socially optimal behavior in several social dilemma matrix games. To apply SQLoss to games where cooperation and defection are determined by a sequence of non-trivial actions, we present GameDistill, an algorithm that reduces a multi-step game with visual input to a matrix game. We empirically show how agents trained with SQLoss on a GameDistill reduced version of the Coin Game evolve optimal policies. 

## Efficient Wasserstein Natural Gradients for Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'optimization']

A novel optimization approach is proposed for application to policy gradient methods and evolution strategies for reinforcement learning (RL). The procedure uses a computationally efficient \emph{Wasserstein natural gradient} (WNG) descent that takes advantage of the geometry induced by a Wasserstein penalty to speed optimization. This method follows the recent theme in RL of including divergence penalties in the objective to establish trust regions. Experiments on challenging tasks demonstrate improvements in both computational cost and performance over advanced baselines. 


## Incremental Policy Gradients for Online Reinforcement Learning Control

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'policy gradient', 'incremental', 'online', 'eligibility traces']

Policy gradient methods are built on the policy gradient theorem, which involves a term representing the complete sum of rewards into the future: the return. Due to this, one usually either waits until the end of an episode before performing updates, or learns an estimate of this return--a so-called critic. Our emphasis is on the first approach in this work, detailing an incremental policy gradient update which neither waits until the end of the episode, nor relies on learning estimates of the return. We provide on-policy and off-policy variants of our algorithm, for both the discounted return and average reward settings. Theoretically, we draw a connection between the traces our methods use and the stationary distributions of the discounted and average reward settings. We conclude with an experimental evaluation of our methods on both simple-to-understand and complex domains.

## Learning to Explore with Pleasure

Authors: ['Anonymous']

Keywords: ['exploration', 'curiosity-driven reinforcement learning', 'Bayesian optimisation']

Exploration is a long-standing challenge in sequential decision problem in machine learning. This paper investigates the adoption of two theories of optimal stimulation level - "the pacer principle" and the Wundt curve - from psychology to improve the exploration challenges. We propose a method called exploration with pleasure (EP) which is formulated based on the notion of pleasure as defined in accordance with the above two theories. EP is able to identify the region of stimulations that will trigger pleasure to the learning agent during exploration and consequently improve on the learning process. The effectiveness of EP is studied in two machine learning settings: curiosity-driven reinforcement learning (RL) and Bayesian optimisation (BO). Experiments in purely curiosity-driven RL show that by using EP to generate intrinsic rewards, it can yield faster learning. Experiments in BO demonstrate that by using EP to specify the exploration parameters in two acquisition functions - Probability of Improvement and Expected Improvement - it can achieve faster convergence and better function values.


## Exploring Transferability of Perturbations in Deep Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['transferability', 'deep reinforcement learning', 'generalization', 'adversarial']

The use of Deep Neural Networks (DNNs) as function approximators has led to striking progress for reinforcement learning algorithms and applications. At the same time, deep reinforcement learning agents have inherited the vulnerability of DNNs to imperceptible adversarial perturbations to their inputs. Prior work on adversarial perturbations for deep reinforcement learning has generally relied on calculating an adversarial perturbation customized to each state visited by the agent. In this paper we propose a more realistic threat model in which the adversary computes the perturbation only once based on a single state. Furthermore, we show that to cause a deep reinforcement learning agent to fail it is enough to have only one adversarial offset vector in a black-box setting. We conduct experiments in various games from the Atari environment, and use our single-state adversaries to demonstrate the transferability of perturbations both between states of one MDP, and between entirely different MDPs. We believe our adversary framework reveals fundamental properties of the environments used in deep reinforcement learning training, and is a tangible step towards building robust and reliable deep reinforcement learning agents.

## Model-Based Reinforcement Learning via Latent-Space Collocation

Authors: ['Anonymous']

Keywords: ['visual model-based reinforcement learning', 'visual planning', 'long-horizon planning', 'collocation']

The ability to construct and execute long-term plans enables intelligent agents to solve complex multi-step tasks and prevents myopic behavior only seeking the short-term reward. Recent work has achieved significant progress on building agents that can predict and plan from raw visual observations. However, existing visual planning methods still require a densely shaped reward that provides the algorithm with a short-term signal that is always easy to optimize. These algorithms fail when the shaped reward is not available as they use simplistic planning methods such as sampling-based random shooting and are unable to plan for a distant goal. Instead, to achieve long-horizon visual control, we propose to use collocation-based planning, a powerful optimal control technique that plans forward a sequence of states while constraining the transitions to be physical. We propose a planning algorithm that adapts collocation to visual planning by leveraging probabilistic latent variable models. A model-based reinforcement learning agent equipped with our planning algorithm significantly outperforms prior model-based agents on challenging visual control tasks with sparse rewards and long-term goals. 

## Daylight: Assessing Generalization Skills of Deep Reinforcement Learning Agents

Authors: ['Anonymous']

Keywords: ['deep reinforcement learning', 'generalization']

Deep reinforcement learning algorithms have recently achieved significant success in learning high-performing policies from purely visual observations. The ability to perform end-to-end learning from raw high dimensional input alone has led to deep reinforcement learning algorithms being deployed in a variety of fields. Thus, understanding and improving the ability of deep reinforcement learning agents to generalize to unseen data distributions is of critical importance. Much recent work has focused on assessing the generalization of deep reinforcement learning agents by introducing specifically crafted adversarial perturbations to their inputs. In this paper, we propose another approach that we call daylight: a framework to assess the generalization skills of trained deep reinforcement learning agents. Rather than focusing on worst-case analysis of distribution shift, our approach is based on black-box perturbations that correspond to semantically meaningful changes to the environment or the agent's visual observation system ranging from brightness to compression artifacts. We demonstrate that even the smallest changes in the environment cause the performance of the agents to degrade significantly in various games from the Atari environment despite having orders of magnitude lower perceptual similarity distance compared to state-of-the-art adversarial attacks, . We show that our framework captures a diverse set of bands in the Fourier spectrum, giving a better overall understanding of the agent's generalization capabilities. We believe our work can be crucial towards building resilient and generalizable deep reinforcement learning agents.

## Learning Robust State Abstractions for Hidden-Parameter Block MDPs

Authors: ['Anonymous']

Keywords: ['multi-task reinforcement learning', 'bisimulation', 'hidden-parameter mdp', 'block mdp']

Many control tasks exhibit similar dynamics that can be modeled as having common latent structure. Hidden-Parameter Markov Decision Processes (HiP-MDPs) explicitly model this structure to improve sample efficiency in multi-task settings.
However, this setting makes strong assumptions on the observability of the state that limit its application in real-world scenarios with rich observation spaces.  In this work, we leverage ideas of common structure from the HiP-MDP setting, and extend it to enable robust state abstractions inspired by Block MDPs. We  derive instantiations of this new framework for  both multi-task reinforcement learning (MTRL) and  meta-reinforcement learning (Meta-RL) settings. Further, we provide transfer and generalization bounds based on task and state similarity, along with sample complexity bounds that depend on the aggregate number of samples across tasks, rather than the number of tasks, a significant improvement over prior work. To further demonstrate efficacy of the proposed method, we empirically compare and show improvement over multi-task and meta-reinforcement learning baselines.

## Learning Safe Multi-agent Control with Decentralized Neural Barrier Certificates

Authors: ['Anonymous']

Keywords: ['Multi-agent', 'safe', 'control barrier function', 'reinforcement learning']

We study the multi-agent safe control problem where agents should avoid collisions to static obstacles and collisions with each other while reaching their goals. Our core idea is to learn the multi-agent control policy jointly with  learning the control barrier functions as safety certificates. We propose a novel joint-learning framework that can be implemented in a decentralized fashion, with generalization guarantees for certain function classes. Such a decentralized framework can adapt to an arbitrarily large number of agents. Building upon this framework, we further improve the scalability by  incorporating neural network architectures  that are invariant to the quantity and permutation of neighboring agents. In addition, we propose a new spontaneous policy refinement method to further enforce the certificate condition during testing. We provide extensive experiments to demonstrate that our method significantly outperforms other leading multi-agent control approaches in terms of maintaining safety and completing original tasks. Our approach also shows exceptional generalization capability in that the control policy can be trained with 8 agents in one scenario, while being used on other scenarios with up to 1024 agents in complex multi-agent environments and dynamics.

## Density Constrained Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Constrained Reinforcement Learning', 'Density', 'Safe AI']

Constrained reinforcement learning (CRL) plays an important role in solving safety-critical and resource-limited tasks. However, existing methods typically rely on tuning reward or cost parameters to encode the constraints, which can be tedious and tend to not generalize well. Instead of building sophisticated cost functions for constraints, we present a pioneering study of imposing constraints directly on the state density function of the system. Density functions have clear physical meanings and can express a variety of constraints in a straightforward fashion. We prove the duality between the density function and Q function in CRL and use it to develop an effective primal-dual algorithm to solve density constrained reinforcement learning problems. We provide theoretical guarantees of the optimality of our approach and use a comprehensive set of case studies including standard benchmarks to show that our method outperforms other leading CRL methods in terms of achieving higher reward while respecting the constraints.

## Domain-Robust Visual Imitation Learning with Mutual Information Constraints

Authors: ['Anonymous']

Keywords: ['Imitation Learning', 'Reinforcement Learning', 'Observational Imitation', 'Third-Person Imitation', 'Mutual Information', 'Domain Adaption', 'Machine Learning']

Human beings are able to understand objectives and learn by simply observing others perform a task. Imitation learning methods aim to replicate such capabilities, however, they generally depend on access to a full set of optimal states and actions taken with the agent's actuators and from the agent's point of view. In this paper, we introduce a new algorithm - called Disentangling Generative Adversarial Imitation Learning (DisentanGAIL) - with the purpose of bypassing such constraints. Our algorithm enables autonomous agents to learn directly from high dimensional observations of an expert performing a task, by making use of adversarial learning with a latent representation inside the discriminator network. Such latent representation is regularized through mutual information constraints to incentivize learning only features that encode information about the completion levels of the task being demonstrated. This allows to obtain a shared feature space to successfully perform imitation while disregarding the differences between the expert's and the agent's domains. Empirically, our algorithm is able to efficiently imitate in a diverse range of control problems including balancing, manipulation and locomotive tasks, while being robust to various domain differences in terms of both environment appearance and agent embodiment.

## CDT: Cascading Decision Trees for Explainable Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Explainable Reinforcement Learning', 'Decision Tree', 'Matrix Factorization']

Deep Reinforcement Learning (DRL) has recently achieved significant advances in various domains. However, explaining the policy of RL agents still remains an open problem due to several factors, one being the complexity of explaining neural networks decisions. Recently, a group of works have used decision-tree-based models to learn explainable policies. Soft decision trees (SDTs) and discretized differentiable decision trees (DDTs) have demonstrated to achieve both good performance and share the benefit of having explainable policies. In this work, we further improve the results for tree-based explainable RL in both performance and explainability. Our proposal, Cascading Decision Trees (CDTs) apply representation learning on the decision path to allow richer expressivity. Empirical results show that in both situations, where CDTs are used as policy function approximators or as imitation learners to explain black-box policies, CDTs can achieve better performances with more succinct and explainable models than SDTs. As a second contribution our study reveals limitations of explaining black-box policies via imitation learning with tree-based explainable models, due to its inherent instability.

## Symmetry-Aware Actor-Critic for 3D Molecular Design

Authors: ['Anonymous']

Keywords: ['deep reinforcement learning', 'molecular design', 'covariant neural networks']

Automating molecular design using deep reinforcement learning (RL) has the potential to greatly accelerate the search for novel materials. Despite recent progress on leveraging graph representations to design molecules, such methods are fundamentally limited by the lack of three-dimensional (3D) information. In light of this, we propose a novel actor-critic architecture for 3D molecular design that can generate geometrically complex molecular structures unattainable with previous approaches. This is achieved by exploiting the symmetries of the design process through a rotationally covariant state-action representation based on a spherical harmonics series expansion. We demonstrate the benefits of our approach on several 3D molecular design tasks, where we find that building in such symmetries significantly improves generalization and the quality of generated molecules.

## Iterative Empirical Game Solving via Single Policy Best Response

Authors: ['Anonymous']

Keywords: ['Empirical Game Theory', 'Reinforcement Learning', 'Multiagent Learning']

Policy-Space Response Oracles (PSRO) is a general algorithmic framework for learning policies in multiagent systems by interleaving empirical game analysis with deep reinforcement learning (DRL).
At each iteration, DRL is invoked to train a best response to a mixture of opponent policies.
The repeated application of DRL poses an expensive computational burden as we look to apply this algorithm to more complex domains.
We introduce two variations of PSRO designed to reduce the amount of simulation required during DRL training.
Both algorithms modify how PSRO adds new policies to the empirical game, based on learned responses to a single opponent policy.
The first, Mixed-Oracles, transfers knowledge from previous iterations of DRL, requiring training only against the opponent's newest policy.
The second, Mixed-Opponents, constructs a pure-strategy opponent by mixing existing strategy's action-value estimates, instead of their policies.
Learning against a single policy mitigates conflicting experiences on behalf of a learner facing an unobserved distribution of opponents.
We empirically demonstrate that these algorithms substantially reduce the amount of simulation during training required by PSRO, while producing equivalent or better solutions to the game.

## The Skill-Action Architecture: Learning Abstract Action Embeddings for Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Hierarchical Reinforcement Learning', 'Reinforcement Learning']

The option framework, one of the most promising Hierarchical Reinforcement Learning (HRL) frameworks, is developed based on the Semi-Markov Decision Problem (SMDP) and employs a triple formulation of the option (i.e., an action policy, a termination probability, and an initiation set). These design choices, however, mean that the option framework: 1) has low sample efficiency, 2) cannot use more stable Markov Decision Problem (MDP) based learning algorithms, 3) represents abstract actions implicitly, and 4) is expensive to scale up. To overcome these problems, here we propose a simple yet effective MDP implementation of the option framework: the Skill-Action (SA) architecture. Derived from a novel discovery that the SMDP option framework has an MDP equivalence, SA hierarchically extracts skills (abstract actions) from primary actions and explicitly encodes these knowledge into skill context vectors (embedding vectors). Although SA is MDP formulated, skills can still be temporally extended by applying the attention mechanism to skill context vectors. Unlike the option framework, which requires $M$ action policies for $M$ skills, SA's action policy only needs one decoder to decode skill context vectors into primary actions. Under this formulation, SA can be optimized with any MDP based policy gradient algorithm. Moreover, it is sample efficient, cheap to scale up, and theoretically proven to have lower variance. Our empirical studies on challenging infinite horizon robot simulation games demonstrate that SA not only outperforms all baselines by a large margin, but also exhibits smaller variance, faster convergence, and good interpretability. A potential impact of SA is to pave the way for a large scale pre-training architecture in the reinforcement learning area.

## Correcting Momentum in Temporal Difference Learning

Authors: ['Anonymous']

Keywords: ['Momentum', 'Reinforcement Learning', 'Temporal Difference', 'Deep Reinforcement Learning']

A common optimization tool used in deep reinforcement learning is momentum, which consists in accumulating and discounting past gradients, reapplying them at each iteration. We argue that, unlike in supervised learning, momentum in Temporal Difference (TD) learning accumulates gradients that become doubly stale: not only does the gradient of the loss change due to parameter updates, the loss itself changes due to bootstrapping. We first show that this phenomenon exists, and then propose a first-order correction term to momentum. We show that this correction term improves sample efficiency in policy evaluation by correcting target value drift. An important insight of this work is that deep RL methods are not always best served by directly importing techniques from the supervised setting.

## Experience Replay with Likelihood-free Importance Weights

Authors: ['Anonymous']

Keywords: ['Experience Replay', 'Off-Policy Optimization', 'Deep Reinforcement Learning']

The use of past experiences to accelerate temporal difference (TD) learning of value functions, or experience replay, is a key component in deep reinforcement learning. In this work, we propose to reweight experiences based on their likelihood under the stationary distribution of the current policy, and justify this with a contraction argument over the Bellman evaluation operator. The resulting TD objective encourages small approximation errors on the value function over frequently encountered states. To balance bias and variance in practice, we use a likelihood-free density ratio estimator between on-policy and off-policy experiences, and use the ratios as the prioritization weights. We apply the proposed approach empirically on three competitive methods, Soft Actor Critic (SAC), Twin Delayed Deep Deterministic policy gradient (TD3) and Data-regularized Q (DrQ), over 11 tasks from OpenAI gym and DeepMind control suite. We achieve superior sample complexity on 35 out of 45 method-task combinations compared to the best baseline and similar sample complexity on the remaining 10. 

## Learning Deep Features in Instrumental Variable Regression

Authors: ['Anonymous']

Keywords: ['Causal Inference', 'Instrumental Variable Regression', 'Deep Learning', 'Reinforcement Learning']

Instrumental variable (IV) regression is a standard strategy for learning causal relationships between the confounded treatment and outcome variables by utilizing an instrumental variable, which is conditionally independent of the outcome given the treatment. In classical IV regression, learning proceeds in two stages: stage 1 performs linear regression from the instrument to the treatment; and stage 2 performs linear regression from the treatment to outcome, conditioned on the instrument. We propose a novel method, deep feature instrumental variable regression (DFIV), to address the case where relations between instruments, treatments, and outcomes may be nonlinear. In this case, deep neural nets are trained to define informative nonlinear features on the instrument and treatments, for both the stage 1 and stage 2 regression problems. We propose an alternating training regime for these features to ensure good end-to-end performance when composing stages 1 and 2, thus obtaining highly flexible feature maps in a computationally efficient manner. DFIV outperforms recent state-of-the-art methods across several challenging IV benchmarks, including settings involving high dimensional image data. DFIV also exhibits competitive performance in policy evaluation for reinforcement learning, which can be understood as an IV regression task.

## Self-Supervised Policy Adaptation during Deployment

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'robotics', 'self-supervised learning', 'generalization', 'sim2real']

In most real world scenarios, a policy trained by reinforcement learning in one environment needs to be deployed in another, potentially quite different environment. However, generalization across different environments is known to be hard. A natural solution would be to keep training after deployment in the new environment, but this cannot be done if the new environment offers no reward signal. Our work explores the use of self-supervision to allow the policy to continue training after deployment without using any rewards. While previous methods explicitly anticipate changes in the new environment, we assume no prior knowledge of those changes yet still obtain significant improvements. Empirical evaluations are performed on diverse simulation environments from DeepMind Control suite and ViZDoom, as well as real robotic manipulation tasks in  continuously changing environments, taking observations from an uncalibrated camera. Our method improves generalization in 28 out of 32 environments across various tasks and outperforms domain randomization on a majority of environments. Videos are available at https://iclr2021submission.github.io/ICLR2021_Anonymized_PAD/.

## Randomized Ensembled Double Q-Learning: Learning Fast Without a Model

Authors: ['Anonymous']

Keywords: ['Artificial Integlligence', 'Machine Learning', 'Deep Reinforcement Learning']

Using a high Update-To-Data (UTD) ratio, model-based methods have recently achieved much higher sample efficiency than previous model-free methods for continuous-action DRL benchmarks. In this paper, we introduce a simple model-free algorithm, Randomized Ensembled Double Q-Learning (REDQ), and show that its performance is just as good as, if not better than, a state-of-the-art model-based algorithm for the MuJoCo benchmark. Moreover, REDQ can achieve this performance using fewer parameters than the model-based method, and with less wall-clock run time. REDQ has three carefully integrated ingredients which allow it to achieve its high performance: (i) a UTD ratio $>>1$; (ii) an ensemble of Q functions; (iii) in-target minimization across a random subset of Q functions from the ensemble. Through carefully designed experiments, we provide a detailed analysis of REDQ and related model-free algorithms. To our knowledge, REDQ is the first successful model-free DRL algorithm for continuous-action spaces using a UTD ratio $>>1$. 

## PGPS : Coupling Policy Gradient with Population-based Search

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Population-based Search', 'Policy Gradient', 'Combining PG with PS']

Gradient-based policy search algorithms (such as PPO, SAC or TD3) in deep reinforcement learning (DRL) have shown successful results on a range of challenging control tasks. However, they often suffer from flat or deceptive gradient problems. As an alternative to policy gradient methods, population-based evolutionary approaches have been applied to DRL. While population-based search algorithms show more robust learning in a broader range of tasks, they are usually inefficient in the use of samples. Recently, reported are a few attempts (such as CEMRL) to combine gradient with a population in searching optimal policy. This kind of hybrid algorithm takes advantage of both camps. In this paper, we propose yet another hybrid algorithm, which more tightly couples policy gradient with the population-based search. More specifically, we use the Cross-Entropy Method (CEM) for population-based search and Twin Delayed Deep Deterministic Policy Gradient (TD3) for policy gradient. In the proposed algorithm called Coupling Policy Gradient with Population-based Search (PGPS), a single TD3 agent, which learns by a gradient from all experiences generated by population, leads a population by providing its critic function Q as a surrogate to select better performing next-generation population from candidates. On the other hand, if the TD3 agent falls behind the CEM population, then the TD3 agent is updated toward the elite member of the CEM population using loss function augmented with the distance between the TD3 and the CEM elite. Experiments in a MuJoCo environment show that PGPS is robust to deceptive gradient and also outperforms the state-of-the-art algorithms.


## Robust Memory Augmentation by Constrained Latent Imagination

Authors: ['Anonymous']

Keywords: ['Memory Augmentation', 'Model-based reinforcement learning', 'Latent imagination']

The latent dynamics model summarizes an agent’s high dimensional experiences in a compact way. While learning from imagined trajectories by the latent model is confirmed to has great potential to facilitate behavior learning, the lack of memory diversity limits generalization capability. Inspired by a neuroscience experiment of “forming artificial memories during sleep”, we propose a robust memory augmentation method with Constrained Latent ImaginatiON (CLION) under a novel actor-critic framework, which aims to speed up the learning of the optimal policy with virtual episodic. Various experiments on high-dimensional visual control tasks with arbitrary image uncertainty demonstrate that CLION outperforms existing approaches in terms of data-efficiency, robustness to uncertainty, and final performance.

## QPLEX: Duplex Dueling Multi-Agent Q-Learning

Authors: ['Anonymous']

Keywords: ['Multi-agent reinforcement learning', 'Value factorization', 'Dueling structure']

We explore value-based multi-agent reinforcement learning (MARL) in the popular paradigm of centralized training with decentralized execution (CTDE). CTDE has an important concept, Individual-Global-Max (IGM) principle, which requires the consistency between joint and local action selections to support efficient local decision-making. However, in order to achieve scalability, existing MARL methods either limit representation expressiveness of their value function classes or relax the IGM consistency, which may suffer from instability risk or lead to poor performance. This paper presents a novel MARL approach, called duPLEX dueling multi-agent Q-learning (QPLEX), which takes a duplex dueling network architecture to factorize the joint value function. This duplex dueling structure encodes the IGM principle into the neural network architecture and thus enables efficient value function learning. Theoretical analysis shows that QPLEX achieves a complete IGM function class. Empirical experiments on StarCraft II micromanagement tasks demonstrate that QPLEX significantly outperforms state-of-the-art baselines in both online and offline data collection settings, and also reveal that QPLEX achieves high sample efficiency and can benefit from offline datasets without additional online exploration.

## Meta-Learning of Compositional Task Distributions in Humans and Machines

Authors: ['Anonymous']

Keywords: ['meta-learning', 'human cognition', 'reinforcement learning', 'compositionality']

Modern machine learning systems struggle with sample efficiency and are usually trained with enormous amounts of data for each task. This is in sharp contrast with humans, who often learn with very little data. In recent years, meta-learning, in which one trains on a family of tasks (i.e. a task distribution), has emerged as an approach to improving the sample complexity of machine learning systems and to closing the gap between human and machine learning. However, in this paper, we argue that current meta-learning approaches still differ significantly from human learning. We argue that humans learn over tasks by constructing compositional generative models and using these to generalize, whereas current meta-learning methods are biased toward the use of simpler statistical patterns. To highlight this difference, we construct a new meta-reinforcement learning task with a compositional task distribution. We also introduce a novel approach to constructing a ``null task distribution'' with the same statistical complexity as the compositional distribution but without explicit compositionality. We train a standard meta-learning agent, a recurrent network trained with model-free reinforcement learning, and compare it with human performance across the two task distributions. We find that humans do better in the compositional task distribution whereas the agent does better in the non-compositional null task distribution -- despite comparable statistical complexity. This work highlights a particular difference between human learning and current meta-learning models, introduces a task that displays this difference, and paves the way for future work on human-like meta-learning. 

## What About Taking Policy as Input of Value Function: Policy-extended Value Function Approximator

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Value Function Approximation', 'Representation Learning']

The value function lies in the heart of Reinforcement Learning (RL), which defines the long-term evaluation of a policy in a given state. In this paper, we propose Policy-extended Value Function Approximator (PeVFA) which extends the conventional value to be not only a function of state but also an explicit policy representation. Such an extension enables PeVFA to preserve values of multiple policies in contrast to a conventional one with limited capacity for only one policy, inducing the new characteristic of \emph{value generalization among policies}. From both the theoretical and empirical lens, we study value generalization along the policy improvement path (called local generalization), from which we derive a new form of Generalized Policy Iteration with PeVFA to improve the conventional learning process. Besides, we propose a framework to learn the representation of an RL policy,  studying several different approaches to learn an effective policy representation from policy network parameters and state-action pairs through contrastive learning and action prediction. In our experiments, Proximal Policy Optimization (PPO) with PeVFA significantly outperforms its vanilla counterpart in MuJoCo continuous control tasks, demonstrating the effectiveness of value generalization offered by PeVFA and policy representation learning. 

## Data-Efficient Reinforcement Learning with Self-Predictive Representations

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Self-Supervised Learning', 'Representation Learning', 'Sample Efficiency']

While deep reinforcement learning excels at solving tasks where large amounts of data can be collected through virtually unlimited interaction with the environment, learning from limited interaction remains a key challenge. We posit that an agent can learn more efficiently if we augment reward maximization with self-supervised objectives based on structure in its visual input and sequential interaction with the environment. Our method, SPR, trains an agent to predict its own latent state representations multiple steps into the future. We compute target representations for future states using an encoder which is an exponential moving average of the agent's parameters and we make predictions using a learned transition model. On its own, this future prediction objective outperforms prior methods for sample-efficient deep RL from pixels. We further improve performance by adding data augmentation to the future prediction loss, which forces the agent's representations to be consistent across multiple views of an observation. Our full self-supervised objective, which combines future prediction and data augmentation, achieves a median human-normalized score of 0.415 on Atari in a setting limited to 100k steps of environment interaction, which represents a 55% relative improvement over the previous state-of-the-art. Notably, even in this limited data regime, SPR exceeds expert human scores on 7 out of 26 games.  We've made the code associated with this work available at https://anonymous.4open.science/r/b4b93ec6-6e5d-4f43-9b53-54bdf73bea95/.

## Learning to Observe with Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement learning', 'observation strategies', 'active data collection']

We consider a decision making problem where an autonomous agent decides on which actions to take based on the observations it collects from the environment. We are interested in revealing the information structure of the observation space illustrating which type of observations are the most important (such as position versus velocity) and the dependence of this on the state of agent (such as at the bottom versus top of a hill). We approach this problem by associating a cost with collecting observations which increases with the accuracy. We adopt a reinforcement learning (RL) framework where the RL agent learns to adjust the accuracy of the observations alongside learning to perform the original task. We consider both the scenario where the accuracy can be adjusted continuously and also the scenario where the agent has to choose between given preset levels, such as taking a sample perfectly or not taking a sample at all.   In contrast to the existing work that mostly focuses on sample efficiency during training, our focus is on the behaviour during the actual task. Our results illustrate that the RL agent can  learn to use the observation space efficiently and obtain satisfactory performance in the original task while collecting effectively smaller amount of data. By uncovering the relative usefulness of different types of observations and trade-offs within, these results also provide insights for further design of active data acquisition schemes. 

## Acting in Delayed Environments with Non-Stationary Markov Policies

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'delay']

The standard Markov Decision Process (MDP) formulation hinges on the assumption that an action is executed immediately after it was chosen. However, assuming it is often unrealistic and can lead to catastrophic failures in applications such as robotic manipulation, cloud computing, and finance. We introduce a framework for learning and planning in MDPs where the decision-maker commits actions that are executed with a delay of $m$ steps. The brute-force state augmentation baseline where the state is concatenated to the last $m$ committed actions suffers from an exponential complexity in $m$, as we show for policy iteration. We then prove that with execution delay, Markov policies in the original state-space are sufficient for attaining maximal reward, but need to be non-stationary. As for stationary Markov policies, we show they are sub-optimal in general. Consequently, we devise a non-stationary Q-learning style model-based algorithm that solves delayed execution tasks without resorting to state-augmentation. Experiments on tabular, physical, and Atari domains reveal that it converges quickly to high performance even for substantial delays, while standard approaches that either ignore the delay or rely on state-augmentation struggle or fail due to divergence. The code will be shared upon publication.

## Adapting to Reward Progressivity via Spectral Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Deep Reinforcement Learning']

In this paper we consider reinforcement learning tasks with progressive rewards; that is, tasks where the rewards tend to increase in magnitude over time. We hypothesise that this property may be problematic for value-based deep reinforcement learning agents, particularly if the agent must first succeed in relatively unrewarding regions of the task in order to reach more rewarding regions. To address this issue, we propose Spectral DQN, which decomposes the reward into frequencies such that the high frequencies only activate when large rewards are found. This allows us to balance the training loss, so that it gives more even weighting to regions of the task with very different reward scales. In two domains with extreme reward progressivity, where standard value-based methods struggle significantly, Spectral DQN is able to make much farther progress. Moreover, when evaluated on a set of six standard Atari games that do not overtly favour the approach, Spectral DQN's performance remains competitive, demonstrating that it is not overfit to its target problem. Interestingly, the latter experiments also suggest that Spectral DQN may have broader applicability to sparse reward tasks.

## Drop-Bottleneck: Learning Discrete Compressed Representation for Noise-Robust Exploration

Authors: ['Anonymous']

Keywords: ['Reinforcement learning', 'Information bottleneck']

We propose a novel information bottleneck (IB) method named Drop-Bottleneck, which discretely drops input features that are irrelevant to the target variable. Drop-Bottleneck not only enjoys a simple and tractable compression objective but also additionally provides a deterministic compressed representation of the input variable, which is useful for inference tasks that require consistent representation. Moreover, it can perform feature selection considering each feature dimension's relevance to the target task, which is unattainable by most neural network-based IB methods. We propose an exploration method based on Drop-Bottleneck for reinforcement learning tasks. In a multitude of noisy and reward sparse maze navigation tasks in VizDoom (Kempka et al., 2016) and DMLab (Beattie et al., 2016), our exploration method achieves state-of-the-art performance. As a new IB framework, we demonstrate that Drop-Bottleneck outperforms Variational Information Bottleneck (VIB) (Alemi et al., 2017) in multiple aspects including adversarial robustness and dimensionality reduction.

## Alpha-DAG: a reinforcement learning based algorithm to learn Directed Acyclic Graphs

Authors: ['Anonymous']

Keywords: ['Directed acyclic graph', 'reinforcement learning', 'Q Learning', 'Graph Auto-Encoder']

Directed acyclic graphs (DAGs) are widely used to model the casual relationships among random variables in many disciplines. One major class of algorithms for DAGs is  called `search-and-score', which attempts to maximize some goodness-of-fit measure and returns a DAG with the best score. However, most existing methods highly rely on their model assumptions and cannot be applied to the more general real-world problems. This paper proposes a  novel Reinforcement-Learning-based searching algorithm, Alpha-DAG, which gradually finds the optimal order to add edges by learning from the historical searching trajectories. At each decision window, the agent adds the edge with the largest scoring improvement to the current graph. The advantage of Alpha-DAG is supported by the numerical comparison against some state-of-the-art competitors in both synthetic and real examples.

## Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'deployment-efficiency', 'offline RL', 'Model-based RL']

Most reinforcement learning (RL) algorithms assume online access to the environment, in which one may readily interleave updates to the policy with experience collection using that policy. However, in many real-world applications such as health, education, dialogue agents, and robotics, the cost or potential risk of deploying a new data-collection policy is high, to the point that it can become prohibitive to update the data-collection policy more than a few times during learning. With this view, we propose a novel concept of deployment efficiency, measuring the number of distinct data-collection policies that are used during policy learning. We observe that naïvely applying existing model-free offline RL algorithms recursively does not lead to a practical deployment-efficient and sample-efficient algorithm. We propose a novel model-based algorithm, Behavior-Regularized Model-ENsemble (BREMEN), that not only performs better than or comparably as the state-of-the-art dynamic-programming-based and concurrently-proposed model-based offline approaches on existing benchmarks, but can also effectively optimize a policy offline using 10-20 times fewer data than prior works. Furthermore, the recursive application of BREMEN achieves impressive deployment efficiency while maintaining the same or better sample efficiency, learning successful policies from scratch on simulated robotic environments with only 5-10 deployments, compared to typical values of hundreds to millions in standard RL baselines.

## Unbiased learning with State-Conditioned Rewards in Adversarial Imitation Learning

Authors: ['Anonymous']

Keywords: ['Adversarial Learning', 'Imitation Learning', 'Inverse Reinforcement Learning', 'Reinforcement Learning', 'Transfer Learning']

Adversarial imitation learning has emerged as a general and scalable framework for automatic reward acquisition. However, we point out that previous methods commonly exploited occupancy-dependent reward learning formulation. Despite the theoretical justification, the occupancy measures tend to cause issues in practice because of high variance and low vulnerability to domain shifts. Another reported problem is termination biases induced by provided rewarding and regularization schemes around terminal states. In order to deal with these issues, this work presents a novel algorithm called causal adversarial inverse reinforcement learning. The framework employs a dual discriminator architecture for decoupling state densities from rewards formulation. We investigate the reward shaping theory to deal with the finite horizon problem and address the reward function's optimality. The formulation draws a strong connection between adversarial learning and energy-based reinforcement learning; thus, the architecture is capable of recovering a reward function that induces a multi-modal policy. In experiments, we demonstrate that our approach outperforms prior methods in challenging continuous control tasks, even under significant variation in the environments.

## Non-decreasing Quantile Function Network with Efficient Exploration for Distributional Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Non-decreasing Quantile Function', 'Distributional Reinforcement Learning', 'Distributional Prediction Error', 'Exploration']

Although distributional reinforcement learning (DRL) has been widely examined in the past few years, two important problems still remain unsolved. One is how to ensure the monotonicity of the quantile estimates, the other is how to design an efficient exploration strategy to fully utilize the distribution information. In this paper, we propose a non-decreasing quantile function network (NDQFN) to address the crossing issue of the learned quantile curve and incorporate it with an efficient exploration method called distributional prediction error (DPE) for DRL. We demonstrate the necessity and efficiency of our method from both the theory and model implementation perspectives. Experiments on Atari 2600 Games demonstrate that our method can significantly outperforms some state-of-art DRL algorithms especially in some hard explored games.

## Probabilistic Mixture-of-Experts for Efficient Deep Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement Learning', 'Sample Efficiency', 'Gaussian Mixture Models', 'Mixture-of-Experts']

Deep reinforcement learning (DRL) has successfully solved various problems recently, typically with a unimodal policy representation. However, grasping the decomposable and hierarchical structures within a complex task can be essential for further improving its learning efficiency and performance, which may lead to a multimodal policy or a mixture-of-experts (MOE). To our best knowledge, present DRL algorithms for general utility do not deploy MOE methods as policy function approximators due to the lack of differentiability, or without explicit probabilistic representation. In this work, we propose a differentiable probabilistic mixture-of-experts (PMOE) embedded in the end-to-end training scheme for generic off-policy and on-policy algorithms using stochastic policies, e.g., Soft Actor-Critic (SAC) and Proximal Policy Optimisation (PPO). Experimental results testify the advantageous performance of our method over unimodal polices and three different MOE methods, as well as a method of option frameworks, based on two types of DRL algorithms. We also demonstrate the distinguishable primitives learned with PMOE in different environments.

## Representation Balancing Offline Model-based Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Model-based Reinforcement Learning', 'Offline Reinforcement Learning', 'Batch Reinforcement Learning', 'Off-policy policy evaluation']

One of the main challenges in offline and off-policy reinforcement learning is to cope with the distribution shift that arises from the mismatch between the target policy and the data collection policy. In this paper, we focus on a model-based approach, particularly on learning the representation for a robust model of the environment under the distribution shift, which has been first studied by Representation Balancing MDP (RepBM). Although this prior work has shown promising results, there are a number of shortcomings that still hinder its applicability to practical tasks. In particular, we address the curse of horizon exhibited by RepBM, rejecting most of the pre-collected data in long-term tasks. We present a new objective for model learning motivated by recent advances in the estimation of stationary distribution corrections. This effectively overcomes the aforementioned limitation of RepBM, as well as naturally extending to continuous action spaces and stochastic policies. We also present an offline model-based policy optimization using this new objective, yielding the state-of-the-art performance in a representative set of benchmark offline RL tasks.

## Rewriting by Generating: Learn Heuristics for Large-scale Vehicle Routing Problems

Authors: ['Anonymous']

Keywords: ['vehicle routing problem', 'reinforcement learning', 'optimization']

The large-scale vehicle routing problem is defined based on the classical VRP with usually more than one thousand customers. It is of great importance to find an efficient and high-qualified solution. However, current algorithms of VRP including heuristics and RL based methods, only performs well on small-scale instances with usually no more than a hundred customers, and fails to solve large-scale VRP due to either high computation cost or the explosive exploration space that results in model divergence. Based on the classical idea of Divide-and-Conquer, we present a novel Rewriting-by-Generating(RBG) framework with hierarchical RL agents to solve the large-scale VRP. RBG contains a rewriter agent that refines the customer division and an elementary generator to generate regional solutions individually. Extensive experiments demonstrate our RBG framework has significant performance on large-scale VRP. RBG outperforms LKH3, the state-of-the-art method in solving VRP, by 2.43% with the problem size of N=2000 and could infer solutions about 100 times faster.

## The Importance of Pessimism in Fixed-Dataset Policy Optimization

Authors: ['Anonymous']

Keywords: ['deep learning', 'reinforcement learning']

We study worst-case guarantees on the expected return of fixed-dataset policy optimization algorithms. Our core contribution is a unified conceptual and mathematical framework for the study of algorithms in this regime. This analysis reveals that for naive approaches, the possibility of erroneous value overestimation leads to a difficult-to-satisfy requirement: in order to guarantee that we select a policy which is near-optimal, we may need the dataset to be informative of the value of every policy. To avoid this, algorithms can follow the pessimism principle, which states that we should choose the policy which acts optimally in the worst possible world. We show why pessimistic algorithms can achieve good performance even when the dataset is not informative of every policy, and derive families of algorithms which follow this principle. These theoretical findings are validated by experiments on a tabular gridworld, and deep learning experiments on four MinAtar environments.

## Optimizing Information Bottleneck in Reinforcement Learning: A Stein Variational Approach

Authors: ['Anonymous']

Keywords: ['Information Bottleneck', 'Reinforcement Learning', 'Stein Variational Gradient']

The information bottleneck (IB) principle is an elegant and useful learning framework for extracting relevant information that an input feature contains about the target. The principle has been widely used in supervised and unsupervised learning. In this paper, we investigate the effectiveness of the IB framework in reinforcement learning (RL). We first derive the objective based on IB in reinforcement learning, then we analytically derive the optimal conditional distribution of the optimization problem. Following the variational information bottleneck (VIB), we provide a variational lower bound using a prior distribution. Unlike VIB, we propose to utilize the amortized Stein variational gradient method to optimize the lower bound. We incorporate this framework in two popular RL algorithms: the advantageous actor critic algorithm (A2C) and the proximal policy optimization algorithm (PPO). Our experimental results show that our framework can improve the sample efficiency of vanilla A2C and PPO. We also show that our method achieves better performance than VIB and mutual information neural estimation (MINE), two other popular approaches to optimize the information bottleneck framework in supervised learning.

## Empirical Sufficiency Featuring Reward Delay Calibration

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement Learning', 'Reward Calibration', 'Empirical Sufficiency', 'Overfitting.']

Appropriate credit assignment for delay rewards is a fundamental challenge in various deep reinforcement learning tasks. To tackle this problem, we introduce a delay reward calibration paradigm inspired from a classification perspective. We hypothesize that when an agent's behavior satisfies an equivalent sufficient condition to be awarded, well-represented state vectors should share similarities. To this end, we define an empirical sufficient distribution, where the state vectors within the distribution will lead agents to environmental reward signals in consequent steps. Therefore, an overfitting classifier is established to handle the distribution and generate calibrated rewards. We examine the correctness of sufficient state extraction by tracking the real-time extraction and building hybrid different reward functions in environments with different levels of awarding latency. The results demonstrate that the classifier could generate timely and accurate calibrated rewards, and the rewards could make the training more efficient. Finally, we find that the sufficient states extracted by our model resonate with observations of human cognition.

## Skill Transfer via Partially Amortized Hierarchical Planning

Authors: ['Anonymous']

Keywords: ['Model-based Reinforcement Learning', 'Partial Amortization', 'Planning']

To quickly solve new tasks in complex environments, intelligent agents need to build up reusable knowledge. For example, a learned world model captures knowledge about the environment that applies to new tasks. Similarly, skills capture general behaviors that can apply to new tasks. In this paper, we investigate how these two approaches can be integrated into a single reinforcement learning agent. Specifically, we leverage the idea of partial amortization for fast adaptation at test time. For this, actions are produced by a policy that is learned over time while the skills it conditions on are chosen using online planning. We demonstrate the benefits of our design decisions across a suite of challenging locomotion tasks and demonstrate improved sample efficiency in single tasks as well as in transfer from one task to another, as compared to competitive baselines. Videos are available at: \url{https://sites.google.com/view/partial-amortization/home}

## Compute- and Memory-Efficient Reinforcement Learning with Latent Experience Replay

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'deep learning', 'computational efficiency', 'memory efficiency']

Recent advances in off-policy deep reinforcement learning (RL) have led to impressive success in complex tasks from visual observations. Experience replay improves sample-efficiency by reusing experiences from the past, and convolutional neural networks (CNNs) process high-dimensional inputs effectively. However, such techniques demand high memory and computational bandwidth. In this paper, we present Latent Vector Experience Replay (LeVER), a simple modification of existing off-policy RL methods, to address these computational and memory requirements without sacrificing the performance of RL agents. To reduce the computational overhead of gradient updates in CNNs, we freeze the lower layers of CNN encoders early in training due to early convergence of their parameters. Additionally, we reduce memory requirements by storing the low-dimensional latent vectors for experience replay instead of high-dimensional images, enabling an adaptive increase in the replay buffer capacity, a useful technique in constrained-memory settings. In our experiments, we show that LeVER does not degrade the performance of RL agents while significantly saving computation and memory across a diverse set of DeepMind Control environments and Atari games. Finally, we show that LeVER is useful for computation-efficient transfer learning in RL because lower layers of CNNs extract generalizable features, which can be used for different tasks and domains.

## On the Estimation Bias in Double Q-Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement learning', 'Q-learning', 'Estimation bias']

Double Q-learning is a classical method for reducing overestimation bias, which is caused by taking maximum estimated values in the Bellman operator. Its variants in the deep Q-learning paradigm have shown great promise in producing reliable value prediction and improving learning performance. However, as shown by prior work, double Q-learning is not fully unbiased and still suffers from underestimation bias. In this paper, we show that such underestimation bias may lead to multiple non-optimal fixed points under an approximated Bellman operation. To address the concerns of converging to non-optimal stationary solutions, we propose a simple and effective approach as a partial fix for underestimation bias in double Q-learning. This approach leverages real returns to bound the target value. We extensively evaluate the proposed method in the Atari benchmark tasks and demonstrate its significant improvement over baseline algorithms.

## Practical Marginalized Importance Sampling with the Successor Representation

Authors: ['Anonymous']

Keywords: ['marginalized importance sampling', 'off-policy evaluation', 'deep reinforcement learning', 'successor representation']

Marginalized importance sampling (MIS), which measures the density ratio between the state-action occupancy of a target policy and that of a sampling distribution, is a promising approach for off-policy evaluation. However, current state-of-the-art MIS methods rely on complex optimization tricks and succeed mostly on simple toy problems. We bridge the gap between MIS and deep reinforcement learning by observing that the density ratio can be computed from the successor representation of the target policy. The successor representation can be trained through deep reinforcement learning methodology and decouples the reward optimization from the dynamics of the environment, making the resulting algorithm stable and applicable to high-dimensional domains. We evaluate the empirical performance of our approach on a variety of challenging Atari and MuJoCo environments.

## Visual Imitation with Reinforcement Learning using Recurrent Siamese Networks

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Imitation learning']

It would be desirable for a reinforcement learning (RL) based agent to learn behaviour by merely watching a demonstration. However, defining rewards that facilitate this goal within the RL paradigm remains a challenge. Here we address this problem with Siamese networks, trained to compute distances between observed behaviours and an agent's behaviours. We use an RNN-based comparator model to learn such distances in space and time between motion clips while training an RL policy to minimize this distance. Through experimentation, we have also found that the inclusion of multi-task data and an additional image encoding loss helps enforce temporal consistency and improve policy learning. These two components appear to balance reward for matching a specific instance of a behaviour versus that behaviour in general. Furthermore, we focus here on a particularly challenging form of this problem where only a single demonstration is provided for a given task -- the one-shot learning setting. We demonstrate our approach on humanoid, dog and raptor agents in 2D and a 3D quadruped and humanoid. In these environments, we show that our method outperforms the state-of-the-art, GAIfO (i.e. GAIL without access to actions) and TCNs.

## SMiRL: Surprise Minimizing Reinforcement Learning in Unstable Environments

Authors: ['Anonymous']

Keywords: ['Reinforcement learning']

Every living organism struggles against disruptive environmental forces to carve out and maintain an orderly niche. We propose that such a struggle to achieve and preserve order might offer a principle for the emergence of useful behaviors in artificial agents. We formalize this idea into an unsupervised reinforcement learning method called surprise minimizing reinforcement learning (SMiRL). SMiRL alternates between learning a density model to evaluate the surprise of a stimulus, and improving the policy to seek more predictable stimuli. The policy seeks out stable and repeatable situations that counteract the environment's prevailing sources of entropy. This might include avoiding other hostile agents, or finding a stable, balanced pose for a bipedal robot in the face of disturbance forces. We demonstrate that our surprise minimizing agents can successfully play Tetris, Doom, control a humanoid to avoid falls, and navigate to escape enemies in a maze without any task-specific reward supervision. We further show that SMiRL can be used together with standard task rewards to accelerate reward-driven learning.

## On Trade-offs of Image Prediction in Visual Model-Based Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['world models', 'model based reinforcement learning', 'latent planning', 'model-based reinforcement learning', 'model predictive control', 'video prediction']

Model-based reinforcement learning (MBRL) methods have shown strong sample efficiency and performance across a variety of tasks, including when faced with high-dimensional visual observations. These methods learn to predict the environment dynamics and expected reward from interaction and use this predictive model to plan and perform the task. However, MBRL methods vary in their fundamental design choices, and it there is no strong consensus in the literature on how these design decisions affect performance. In this paper, we study a number of design decisions for the predictive model in visual MBRL algorithms, focusing specifically on methods that use a predictive model for planning. We find that a range of design decisions that are often considered crucial, such as the use of latent spaces, have little effect on task performance. A big exception to this finding is that predicting future observations (i.e., images) leads to significant task performance improvement compared to only predicting rewards. We also empirically find that image prediction accuracy, somewhat surprisingly, correlates more strongly with downstream task performance than reward prediction accuracy. We show how this phenomenon is related to exploration and how some of the lower-scoring models on standard benchmarks (that require exploration) will perform the same as the best-performing models when trained on the same training data. Simultaneously, in the absence of exploration, models that fit the data better usually perform better on the down-stream task as well, but surprisingly, these are often not the same models that perform the best when learning and exploring from scratch. These findings suggest that performance and exploration place important and potentially contradictory requirements on the model.

## Accelerating Safe Reinforcement Learning with Constraint-mismatched Policies

Authors: ['Anonymous']

Keywords: ['Reinforcement learning with constraints', 'Safe reinforcement learning']

We consider the problem of reinforcement learning when provided with (1) a baseline control policy and (2) a set of constraints that the controlled system must satisfy. The baseline policy can arise from a teacher agent, demonstration data or even a heuristic while the constraints might encode safety, fairness or other application-specific requirements. Importantly, the baseline policy may be sub-optimal for the task at hand, and is not guaranteed to satisfy the specified constraints. The key challenge therefore lies in effectively leveraging the baseline policy for faster learning, while still ensuring that the constraints are minimally violated. To reconcile these potentially competing aspects, we propose an iterative policy optimization algorithm that alternates between maximizing expected return on the task, minimizing distance to the baseline policy, and projecting the policy onto the constraint-satisfying set. We analyze the convergence of our algorithm theoretically and provide a finite-sample guarantee. In our empirical experiments on five different control tasks, our algorithm consistently outperforms several state-of-the-art methods, achieving 10 times fewer constraint violations and 40% higher reward on average.

## XT2: Training an X-to-Text Typing Interface with Online Learning from Implicit Feedback

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'human-computer interaction']

We aim to help users communicate their intent to machines using flexible, adaptive interfaces that translate arbitrary user input into desired actions. In this work, we focus on assistive typing applications in which a user cannot operate a keyboard, but can instead supply other inputs, such as webcam images that capture eye gaze. Standard methods train a model on a fixed dataset of user inputs, then deploy a static interface that does not learn from its mistakes; in part, because extracting an error signal from user behavior can be challenging. We investigate a simple idea that would enable such interfaces to improve over time, with minimal additional effort from the user: online learning from implicit user feedback on the accuracy of the interface's actions. In the typing domain, we leverage backspaces as implicit feedback that the interface did not perform the desired action. We propose an algorithm called x-to-text (XT2) that trains a predictive model of this implicit feedback signal, and uses this model to fine-tune any existing, default interface for translating user input into actions that select words or characters. We evaluate XT2 through a small-scale online user study with 12 participants who type sentences by gazing at their desired words, and a large-scale observational study on handwriting samples from 60 users. The results show that XT2 learns to outperform a non-adaptive default interface, stimulates user co-adaptation to the interface, personalizes the interface to individual users, and can leverage offline data collected from the default interface to improve its initial performance and accelerate online learning.

## Optimistic Exploration with Backward Bootstrapped Bonus for Deep Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['optimistic exploration', 'backward bootstrapped bonus', 'posterior sampling', 'reinforcement learning']

Optimism in the face of uncertainty is a principled approach for provably efficient exploration for reinforcement learning in tabular and linear settings. However, such an approach is challenging in developing practical exploration algorithms for Deep Reinforcement Learning (DRL). To address this problem, we propose an Optimistic Exploration algorithm with Backward Bootstrapped Bonus (OEB3) for DRL by following these two principles. OEB3 is built on bootstrapped deep $Q$-learning, a non-parametric posterior sampling method for temporally-extended exploration. Based on such a temporally-extended exploration, we construct an UCB-bonus indicating the uncertainty of $Q$-functions. The UCB-bonus is further utilized to estimate an optimistic $Q$-value, which encourages the agent to explore the scarcely visited states and actions to reduce uncertainty. In the estimation of $Q$-function, we adopt an episodic backward update strategy to propagate the future uncertainty to the estimated $Q$-function consistently. Extensive evaluations show that OEB3 outperforms several state-of-the-art exploration approaches in Mnist maze and 49 Atari games.

## Interactive Visualization for Debugging RL

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Interpretability', 'Visualization']

Visualization tools for supervised learning (SL) allow users to interpret, introspect, and gain an intuition for the successes and failures of their models. While reinforcement learning (RL) practitioners ask many of the same questions while debugging agent policies, existing tools aren't a great fit for the RL setting as these tools address challenges typically found in the SL regime. Whereas SL involves a static dataset, RL often entails collecting new data in challenging environments with partial observability, stochasticity, and non-stationary data distributions. This necessitates the creation of alternate visual interfaces to help us better understand agent policies trained using RL. In this work, we design and implement an interactive visualization tool for debugging and interpreting RL. Our system identifies and addresses important aspects missing from existing tools such as (1) visualizing alternate state representations (different from those seen by the agent) that researchers could use while debugging RL policies; (2) interactive interfaces tailored to metadata stored while training RL agents (3) a conducive workflow designed around RL policy debugging. We provide an example workflow of how this system could be used, along with ideas for future extensions.

## Goal-Auxiliary Actor-Critic for 6D Robotic Grasping with Point Clouds

Authors: ['Anonymous']

Keywords: ['Robotics', 'Reinforcement Learning', 'Learning from Demonstration']

6D robotic grasping beyond top-down bin-picking scenarios is a challenging task. Previous solutions based on 6D grasp synthesis with robot motion planning usually operate in an open-loop setting without considering the dynamics and contacts of objects, which makes them sensitive to grasp synthesis errors. In this work, we propose a novel method for learning closed-loop control policies for 6D robotic grasping using point clouds from an egocentric camera. We combine imitation learning and reinforcement learning in order to grasp unseen objects and handle the continuous 6D action space, where expert demonstrations are obtained from a joint motion and grasp planner. We introduce a goal-auxiliary actor-critic algorithm, which uses grasping goal prediction as an auxiliary task to facilitate policy learning. The supervision on grasping goals can be obtained from the expert planner for known objects or from hindsight goals for unknown objects. Overall, our learned closed-loop policy achieves over 90% success rates on grasping various ShapeNet objects and YCB objects in simulation.

## C-Learning: Learning to Achieve Goals via Recursive Classification

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'goal reaching', 'density estimation', 'Q-learning', 'hindsight relabeling']

We study the problem of predicting and controlling the future state distribution of an autonomous agent. This problem, which can be viewed as a reframing of goal-conditioned reinforcement learning (RL), is centered around learning a conditional probability density function over future states. Instead of directly estimating this density function, we indirectly estimate this density function by training a classifier to predict whether an observation comes from the future. Via Bayes' rule, predictions from our classifier can be transformed into predictions over future states. Importantly, an off-policy variant of our algorithm allows us to predict the future state distribution of a new policy, without collecting new experience. This variant allows us to optimize functionals of a policy's future state distribution, such as the density of reaching a particular goal state. While conceptually similar to Q-learning, our work lays a principled foundation for goal-conditioned RL as density estimation, providing justification for goal-conditioned methods used in prior work. This foundation makes hypotheses about Q-learning, including the optimal goal-sampling ratio, which we confirm experimentally. Moreover, our proposed method is competitive with prior goal-conditioned RL methods.

## Off-Dynamics Reinforcement Learning: Training for Transfer with Domain Classifiers

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'transfer learning', 'domain adaptation']

We propose a simple, practical, and intuitive approach for domain adaptation in reinforcement learning. Our approach stems from the idea that the agent's experience in the source domain should look similar to its experience in the target domain. Building off of a probabilistic view of RL, we achieve this goal by compensating for the difference in dynamics by modifying the reward function. This modified reward function is simple to estimate by learning auxiliary classifiers that distinguish source-domain transitions from target-domain transitions. Intuitively, the agent is penalized for transitions that would indicate that the agent is interacting with the source domain, rather than the target domain. Formally, we prove that applying our method in the source domain is guaranteed to obtain a near-optimal policy for the target domain, provided that the source and target domains satisfy a lightweight assumption. Our approach is applicable to domains with continuous states and actions and does not require learning an explicit model of the dynamics. On discrete and continuous control tasks, we illustrate the mechanics of our approach and demonstrate its scalability to high-dimensional~tasks.

## Learning to Reach Goals via Iterated Supervised Learning

Authors: ['Anonymous']

Keywords: ['goal reaching', 'reinforcement learning', 'behavior cloning', 'goal-conditioned RL']

Current reinforcement learning (RL) algorithms can be brittle and difficult to use, especially when learning goal-reaching behaviors from sparse rewards. Although supervised imitation learning provides a simple and stable alternative, it requires access to demonstrations from a human supervisor. In this paper, we study RL algorithms that use imitation learning to acquire goal reaching policies from scratch, without the need for expert demonstrations or a value function. In lieu of demonstrations, we leverage the property that any trajectory is a successful demonstration for reaching the final state in that same trajectory. We propose a simple algorithm in which an agent continually relabels and imitates the trajectories it generates to progressively learn goal-reaching behaviors from scratch. Each iteration, the agent collects new trajectories using the latest policy, and maximizes the likelihood of the actions along these trajectories under the goal that was actually reached, so as to improve the policy. We formally show that this iterated supervised learning procedure optimizes a bound on the RL objective, derive performance bounds of the learned policy, and empirically demonstrate improved goal-reaching performance and robustness over current RL algorithms in several benchmark tasks. 

## Model-Based Visual Planning with Self-Supervised Functional Distances

Authors: ['Anonymous']

Keywords: ['planning', 'model learning', 'distance learning', 'reinforcement learning', 'robotics']

A generalist robot must be able to complete a variety of tasks in its environment. One appealing way to specify each task is in terms of a goal observation. However, learning goal-reaching policies with reinforcement learning remains a challenging problem, particularly when rewards are not provided and distances in the observation space are not meaningful. Learned dynamics models are a promising approach for learning about the environment without rewards or task-directed data, but planning to reach goals with such a model requires a notion of functional similarity between observations and goal states. We present a self-supervised method for model-based visual goal reaching, which uses both a visual dynamics model as well as a dynamical distance function learned using model-free reinforcement learning. This approach trains entirely using offline, unlabeled data, making it practical to scale to large and diverse datasets. On several challenging robotic manipulation tasks with only offline, unlabeled data, we find that our algorithm compares favorably to prior model-based and model-free reinforcement learning methods. In ablation experiments, we additionally identify important factors for learning effective distances.

## Iterative Amortized Policy Optimization

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Policy Optimization', 'Amortization', 'Variational Inference']

Policy networks are a central feature of deep reinforcement learning (RL) algorithms for continuous control, enabling the estimation and sampling of high-value actions. From the variational inference perspective on RL, policy networks, when employed with entropy or KL regularization, are a form of amortized optimization, optimizing network parameters rather than the policy distributions directly. However, this direct amortized mapping can empirically yield suboptimal policy estimates. Given this perspective, we consider the more flexible class of iterative amortized optimizers. We demonstrate that the resulting technique, iterative amortized policy optimization, yields performance improvements over conventional direct amortization methods on benchmark continuous control tasks.

## DeepAveragers: Offline Reinforcement Learning By Solving Derived Non-Parametric MDPs

Authors: ['Anonymous']

Keywords: ['Offline Reinforcement Learning', 'Planning']

We study an approach to offline reinforcement learning (RL) based on optimally solving  finitely-represented  MDPs  derived  from  a  static  dataset  of  experience. This approach can be applied on top of any learned representation and has the potential to easily support multiple solution objectives as well as zero-shot adjustment to changing environments and goals.  Our main contribution is to introduce the Deep Averagers with Costs MDP (DAC-MDP) and to investigate its solutions for offline RL.  DAC-MDPs are a non-parametric model that can leverage deep representations and account for limited data by introducing costs for exploiting under-represented parts of the model.  In theory, we show conditions that allow for lower-bounding the performance of DAC-MDP solutions. We also investigate the empirical behavior in a number of environments, including those with image-based observations. Overall, the experiments demonstrate that the framework can work in practice and scale to large complex offline RL problems.

## Uncertainty Weighted Offline Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'offline', 'batch reinforcement learning', 'off-policy', 'uncertainty estimation', 'dropout', 'actor-critic', 'bootstrap error']

Offline Reinforcement Learning promises to learn effective policies from previously-collected, static datasets without the need for exploration. However, existing Q-learning and actor-critic based off-policy RL algorithms fail when bootstrapping from out-of-distribution (OOD) actions or states. We hypothesize that a key missing ingredient from the existing methods is a proper treatment of uncertainty in the offline setting. We propose Uncertainty Weighted Actor-Critic (UWAC), an algorithm that models the epistemic uncertainty to detect OOD state-action pairs and down-weights their contribution in the training objectives accordingly. Implementation-wise, we adopt a practical and effective dropout-based uncertainty estimation method that introduces very little overhead over existing RL algorithms. Empirically, we observe that UWAC substantially improves model stability during training. In addition, UWAC out-performs existing offline RL methods on a variety of competitive tasks, and achieves significant performance gains over the state-of-the-art baseline on datasets with sparse demonstrations collected from human experts.

## My Body is a Cage: the Role of Morphology in Graph-Based Incompatible Control

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement Learning', 'Multitask Reinforcement Learning', 'Graph Neural Networks', 'Continuous Control']

Multitask Reinforcement Learning is a promising way to obtain models with better performance, generalisation, data efficiency, and robustness. Most existing work is limited to compatible settings, where the state and action space dimensions are the same across tasks. Graph Neural Networks (GNN) are one way to address incompatible environments, because they can process graphs of arbitrary size. They also allow practitioners to inject biases encoded in the structure of the input graph. Existing work in graph-based continuous control uses the physical morphology of the agent to construct the input graph, i.e., encoding limb features as node labels and using edges to connect the nodes if their corresponded limbs are physically connected.
In this work, we present a series of ablations on existing methods that show that morphological information encoded in the graph does not improve their performance. Motivated by the hypothesis that any benefits GNNs extract from the graph structure are outweighed by difficulties they create for message passing, we also propose Amorpheus, a transformer-based approach. Further results show that, while Amorpheus ignores the morphological information that GNNs encode, it nonetheless substantially outperforms GNN-based methods.

## MDP Playground: Controlling Dimensions of Hardness in Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement learning', 'Benchmarks', 'Efficiency', 'Reproducibility', 'Core issues', 'Algorithm analysis', 'Dimensions of hardness', 'OpenAI Gym']

We present MDP Playground, an efficient benchmark for Reinforcement Learning (RL) algorithms with various dimensions of hardness that can be controlled independently to challenge algorithms in different ways and to obtain varying degrees of hardness in generated environments. We consider and allow control over a wide variety of key hardness dimensions, including delayed rewards, rewardable sequences, sparsity of rewards, stochasticity, image representations, irrelevant features, time unit, and action max. While it is very time consuming to run RL algorithms on standard benchmarks, we define a parameterised collection of fast-to-run toy benchmarks in OpenAI Gym by varying these dimensions. Despite their toy nature and low compute requirements, we show that these benchmarks present substantial challenges to current RL algorithms. Furthermore, since we can generate environments with a desired value for each of the dimensions, in addition to having fine-grained control over the environments' hardness, we also have the ground truth available for evaluating algorithms. Finally, we evaluate the kinds of transfer for these dimensions that may be expected from our benchmarks to more complex benchmarks. We believe that MDP Playground is a valuable testbed for researchers designing new, adaptive and intelligent RL algorithms and those wanting to unit test their algorithms.


## An Examination of Preference-based Reinforcement Learning for Treatment Recommendation

Authors: ['Anonymous']

Keywords: ['Preference-based Reinforcement Learning', 'Treatment Recommendation', 'healthcare']

Treatment recommendation is a complex multi-faceted problem with many conflicting objectives, e.g., optimizing the survival rate (or expected lifetime), mitigating negative impacts, reducing financial expenses and time costs, avoiding over-treatment, etc. While this complicates the hand-engineering of a reward function for learning treatment policies, fortunately, qualitative feedback from human experts is readily available and can be easily exploited.  Since direct estimation of rewards via inverse reinforcement learning is a challenging task and requires the existence of an optimal human policy, the field of treatment recommendation has recently witnessed the development of the preference-based ReinforcementLearning (PRL) framework, which infers a reward function from only qualitative and imperfect human feedback to ensure that a human expert’s preferred policy has a higher expected return over a less preferred policy. In this paper, we first present an open simulation platform to model the progression of two diseases, namely Cancer and Sepsis, and the reactions of the affected individuals to the received treatment. Secondly, we investigate important problems in adopting preference-basedRL approaches for treatment recommendation, such as advantages of learning from preference over hand-engineered reward, addressing incomparable policies, reward interpretability, and agent design via simulated experiments. The designed simulation platform and insights obtained for preference-based RL approaches are beneficial for achieving the right trade-off between various human objectives during treatment recommendation.

## Risk-Averse Offline Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['offline', 'reinforcement learning', 'risk-averse', 'risk sensitive', 'robust', 'safety', 'safe']

Training Reinforcement Learning (RL) agents in high-stakes applications might be too prohibitive due to the risk associated to exploration. Thus, the agent can only use data previously collected by safe policies. While previous work considers optimizing the average performance using offline data, we focus on optimizing a risk-averse criteria, namely the CVaR. In particular, we present the Offline Risk-Averse Actor-Critic (O-RAAC), a model-free RL algorithm that is able to learn risk-averse policies in a fully offline setting. We show that O-RAAC learns policies with higher CVaR than risk-neutral approaches in different robot control tasks. Furthermore, considering risk-averse criteria guarantees distributional robustness of the average performance with respect to particular distribution shifts. We demonstrate empirically that in the presence of natural distribution-shifts, O-RAAC learns policies with good average performance. 


## Guiding Representation Learning in Deep Generative Models with Policy Gradients

Authors: ['Anonymous']

Keywords: ['VAE', 'RL', 'PPO']

Variational Auto Encoder (VAE) provide an efficient latent space representation of complex data distributions which is learned in an unsupervised fashion.
Using such a representation as input to Reinforcement Learning (RL) approaches may reduce learning time, enable domain transfer or improve interpretability of the model.
However, current state-of-the-art approaches that combine VAE with RL fail at learning good performing policies on certain RL domains.
Typically, the VAE is pre-trained in isolation and may omit the embedding of task-relevant features due to insufficiencies of its loss.
As a result, the RL approach can not successfully maximize the reward on these domains.
Therefore, this paper investigates the issues of joint training approaches and explores incorporation of policy gradients from RL into the VAE's latent space to find a task-specific latent space representation.
We show that using pre-trained representations can lead to policies being unable to learn any rewarding behaviour in these environments.
Subsequently, we introduce two types of models which overcome this deficiency by using policy gradients to learn the representation.
Thereby the models are able to embed features into its representation that are crucial for performance on the RL task but would not have been learned with previous methods.

## Self-Supervised Continuous Control without Policy Gradient

Authors: ['Anonymous']

Keywords: ['Self-Supervised Reinforcement Learning', 'Continuous Control', 'Zeroth-Order Optimization']

Despite the remarkable progress made by the policy gradient algorithms in reinforcement learning (RL), sub-optimal policies usually result from the local exploration property of the policy gradient update. In this work, we propose a method called Zeroth-Order Supervised Policy Improvement (ZOSPI) that exploits the estimated value function $Q$ globally while preserving the local exploitation of the policy gradient methods. Experiments show that ZOSPI achieves competitive results on the MuJoCo benchmarks with a remarkable sample efficiency. Moreover, different from the conventional policy gradient methods, the policy learning of ZOSPI is conducted in a self-supervised manner. We show such a self-supervised learning paradigm has the flexibility of including optimistic exploration as well as adopting a non-parametric policy.

## Novel Policy Seeking with Constrained Optimization

Authors: ['Anonymous']

Keywords: ['Novel Policy Seeking', 'Reinforcement Learning', 'Constrained Optimization']

We address the problem of seeking novel policies in reinforcement learning tasks. Instead of following the multi-objective framework commonly used in existing methods, we propose to rethink the problem under a novel perspective of constrained optimization. We at first introduce a new metric to evaluate the difference between policies, and then design two practical novel policy seeking methods following the new perspective, namely the Constrained Task Novel Bisector (CTNB), and the Interior Policy Differentiation (IPD), corresponding to the feasible direction method and the interior point method commonly known in the constrained optimization literature. Experimental comparisons on the MuJuCo control suite show our methods can achieve substantial improvements over previous novelty-seeking methods in terms of both the novelty of policies and their performances in the primal task.

## Towards Understanding Linear Value Decomposition in Cooperative Multi-Agent Q-Learning

Authors: ['Anonymous']

Keywords: ['Multi-agent reinforcement learning', 'Fitted Q-iteration', 'Value factorization', 'Credit assignment']

Value decomposition is a popular and promising approach to scaling up multi-agent reinforcement learning in cooperative settings. However, the theoretical understanding of such methods is limited. In this paper, we introduce a variant of the fitted Q-iteration framework for analyzing multi-agent Q-learning with value decomposition. Based on this framework, we derive a closed-form solution to the empirical Bellman error minimization with linear value decomposition. With this novel solution, we further reveal two interesting insights: 1) linear value decomposition implicitly implements a classical multi-agent credit assignment called counterfactual difference rewards; and 2) On-policy data distribution or richer Q function classes can improve the training stability of multi-agent Q-learning. In the empirical study, our experiments demonstrate the realizability of our theoretical closed-form formulation and implications in the didactic examples and a broad set of StarCraft II unit micromanagement tasks, respectively. 

## Learning Discrete Adaptive Receptive Fields for Graph Convolutional Networks

Authors: ['Anonymous']

Keywords: ['Graph Neural Networks', 'Reinforcement Learning', 'Attention Mechanism', 'Adaptive Receptive Fields']

Different nodes in a graph neighborhood generally yield different importance. In previous work of Graph Convolutional Networks (GCNs), such differences are typically modeled with attention mechanisms. However, as we prove in our paper, soft attention weights suffer from over-smoothness in large neighborhoods. To address this weakness, we introduce a novel framework of conducting graph convolutions, where nodes are discretely selected among multi-hop neighborhoods to construct adaptive receptive fields (ARFs). ARFs enable GCNs to get rid of the over-smoothness of soft attention weights, as well as to efficiently explore long-distance dependencies in graphs. We further propose GRARF (GCN with Reinforced Adaptive Receptive Fields) as an instance, where an optimal policy of constructing ARFs is learned with reinforcement learning. GRARF achieves or matches state-of-the-art performances on public datasets from different domains. Our further analysis corroborates that GRARF is more robust than attention models against neighborhood noises.

## Diversity Actor-Critic: Sample-Aware Entropy Regularization for Sample-Efficient Exploration

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Entropy Regularization', 'Exploration']

Policy entropy regularization is commonly used for better exploration in deep reinforcement learning (RL). However, policy entropy regularization is sample-inefficient in off-policy learning since it does not take the distribution of previous samples stored in the replay buffer into account. In order to take advantage of the previous sample distribution from the replay buffer for sample-efficient exploration, we propose sample-aware entropy regularization which maximizes the entropy of weighted sum of the policy action distribution and the sample action distribution from the replay buffer. We formulate the problem of sample-aware entropy regularized policy iteration,  prove its convergence, and provide a practical algorithm named diversity actor-critic (DAC) which is a generalization of soft actor-critic (SAC). Numerical results show that DAC significantly outperforms SAC baselines and other state-of-the-art RL algorithms.

## Reinforcement Learning for Flexibility Design Problems

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'flexibility design', 'policy gradient', 'combinatorial optimization']

Flexibility design problems are a class of problems that appear in strategic decision-making across industries, where the objective is to design a (e.g., manufacturing) network that affords flexibility and adaptivity. The underlying combinatorial nature and stochastic objectives make flexibility design problems challenging for standard optimization methods. In this paper, we develop a reinforcement learning (RL) framework for flexibility design problems. Specifically, we carefully design mechanisms with noisy exploration and variance reduction to ensure empirical success and show the unique advantage of RL in terms of fast-adaptation. Empirical results show that the RL-based method consistently finds better solutions compared to classical heuristics.

## Constrained Reinforcement Learning With Learned Constraints

Authors: ['Anonymous']

Keywords: ['Constrained reinforcement learning', 'constrain inference', 'safe reinforcement learning']

Standard reinforcement learning (RL) algorithms train agents to maximize given reward functions. However, many real-world applications of RL require agents to also satisfy certain constraints which may, for example, be motivated by safety concerns. Constrained RL algorithms approach this problem by training agents to maximize given reward functions while respecting \textit{explicitly} defined constraints. However, in many cases, manually designing accurate constraints is a challenging task. In this work, given a reward function and a set of demonstrations from an expert that maximizes this reward function while respecting \textit{unknown} constraints, we propose a framework to learn the most likely constraints that the expert respects. We then train agents to maximize the given reward function subject to the learned constraints. Previous works in this regard have either mainly been restricted to tabular settings or specific types of constraints or assume the ability to modify the environment. In contrast, we empirically show that our framework is able to learn arbitrary constraints in high-dimensions in a model-free setting.

## Unsupervised Task Clustering for Multi-Task Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Multi-Task Learning', 'Clustering', 'Expectation-Maximization']

Meta-learning, transfer learning and multi-task learning have recently laid a path towards more generally applicable reinforcement learning agents that are not limited to a single task. However, most existing approaches implicitly assume a uniform similarity between tasks. We argue that this assumption is limiting in settings where the relationship between tasks is unknown a-priori. In this work, we propose a general approach to automatically cluster together similar tasks during training. Our method, inspired by the expectation-maximization algorithm, succeeds at finding clusters of related tasks and uses these to improve sample complexity. In the expectation step, we evaluate the performance of a set of policies on all tasks and assign each task to the best performing policy. In the maximization step, each policy trains by sampling tasks from its assigned set. This method is intuitive, simple to implement and orthogonal to other multi-task learning algorithms. We show the generality of our approach by evaluating on simple discrete and continuous control tasks, as well as complex bipedal walker tasks and Atari games. Results show improvements in sample complexity as well as a more general applicability when compared to other approaches.


## Drift Detection in Episodic Data: Detect When Your Agent Starts Faltering

Authors: ['Anonymous']

Keywords: ['Reinforcement learning reliability', 'Reinforcement learning stability', 'Drift detection', 'Degradation test', 'Bootstrapping']

Detection of deterioration of agent performance in dynamic environments is challenging due to the non-i.i.d nature of the observed performance. We consider an episodic framework, where the objective is to detect when an agent begins to falter. We devise a hypothesis testing procedure for non-i.i.d rewards, which is optimal under certain conditions. To apply the procedure sequentially in an online manner, we also suggest a novel Bootstrap mechanism for False Alarm Rate control (BFAR). We demonstrate our procedure in problems where the rewards are not independent, nor identically-distributed, nor normally-distributed. The statistical power of the new testing procedure is shown to outperform alternative tests - often by orders of magnitude - for a variety of environment modifications (which cause deterioration in agent performance). Our detection method is entirely external to the agent, and in particular does not require model-based learning. Furthermore, it can be applied to detect changes or drifts in any episodic signal.

## DOP: Off-Policy Multi-Agent Decomposed Policy Gradients

Authors: ['Anonymous']

Keywords: ['Multi-Agent Reinforcement Learning', 'Multi-Agent Policy Gradients']

Multi-agent policy gradient (MAPG) methods recently witness vigorous progress. However, there is a significant performance discrepancy between MAPG methods and state-of-the-art multi-agent value-based approaches. In this paper, we investigate causes that hinder the performance of MAPG algorithms and present a multi-agent decomposed policy gradient method (DOP). This method introduces the idea of value function decomposition into the multi-agent actor-critic framework. Based on this idea, DOP supports efficient off-policy learning and addresses the issue of centralized-decentralized mismatch and credit assignment in both discrete and continuous action spaces. We formally show that DOP critics have sufficient representational capability to guarantee convergence. In addition, empirical evaluations on the StarCraft II micromanagement benchmark and multi-agent particle environments demonstrate that DOP significantly outperforms both state-of-the-art value-based and policy-based multi-agent reinforcement learning algorithms. Demonstrative videos are available at https://sites.google.com/view/dop-mapg/.

## Learning Predictive Communication by Imagination in Networked System Control

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Multi-agent Reinforcement Learning', 'Networked System Control']

Dealing with multi-agent control in networked systems is one of the biggest challenges in Reinforcement Learning (RL) and limited success has been presented compared to recent deep reinforcement learning in single-agent domain. However, obstacles remain in addressing the delayed global information where each agent learns a decentralized control policy based on local observations and messages from connected neighbors. This paper first considers delayed global information sharing by combining the delayed global information and latent imagination of farsighted states in differentiable communication. Our model allows an agent to imagine its future states and communicate that with its neighbors. The predictive message sent to the connected neighbors reduces the delay in global information.  On the tasks of networked multi-agent traffic control, experimental results show that our model helps stabilize the training of each local agent and outperforms existing algorithms for networked system control.

## Learning to Represent Action Values as a Hypergraph on the Action Vertices

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'action representation learning', 'multidimensional action space', 'relational inductive bias']

Action values are ubiquitous in reinforcement learning (RL) methods, with the sample complexity of such methods relying heavily on how fast a good estimator for action value can be learned. Viewing this problem through the lens of representation learning, naturally good representations of both state and action can facilitate action-value estimation. While advances in deep learning have seamlessly driven progress in learning state representations, given the specificity of the notion of agency to RL, little attention has been paid to learning action representations. We conjecture that leveraging the combinatorial structure of multidimensional action spaces is a key ingredient for learning good representations of action. In order to test this, we set forth the action hypergraph networks framework---a class of functions for learning action representations with a relational inductive bias. Using this framework we realise an agent class based on a combination with deep Q-networks, which we dub hypergraph Q-networks. We show the effectiveness of our approach on a myriad of domains: illustrative prediction problems under minimal confounding effects, Atari 2600 games, and physical control benchmarks.

## Local Information Opponent Modelling Using Variational Autoencoders

Authors: ['Anonymous']

Keywords: ['multi-agent systems', 'opponent modelling', 'reinforcement learning']

Modelling the behaviours of other agents (opponents) is essential for understanding how agents interact and making effective decisions. Existing methods for opponent modelling commonly assume knowledge of the local observations and chosen actions of the modelled opponents, which can significantly limit their applicability. We propose a new modelling technique based on variational autoencoders, which are trained to reconstruct the local actions and observations of the opponent based on embeddings which depend only on the local observations of the modelling agent (its observed world state, chosen actions, and received rewards). The embeddings are used to augment the modelling agent's decision policy which is trained via deep reinforcement learning; thus the policy does not require access to opponent observations.  We provide a comprehensive evaluation and ablation study in diverse multi-agent tasks, showing that our method achieves comparable performance to an ideal baseline which has full access to opponent's information, and significantly higher returns than a baseline method which does not use the learned embeddings.

## AUBER: Automated BERT Regularization

Authors: ['Anonymous']

Keywords: ['BERT Regularization', 'Reinforcement Learning', 'Automated Regularization']

How can we effectively regularize BERT? Although BERT proves its effectiveness in various downstream natural language processing tasks, it often overﬁts when there are only a small number of training instances. A promising direction to regularize BERT is based on pruning its attention heads based on a proxy score for head importance. However, heuristic-based methods are usually suboptimal since they predetermine the order by which attention heads are pruned. In order to overcome such a limitation, we propose AUBER, an effective regularization method that leverages reinforcement learning to automatically prune attention heads from BERT. Instead of depending on heuristics or rule-based policies, AUBER learns a pruning policy that determines which attention heads should or should not be pruned for regularization. Experimental results show that AUBER outperforms existing pruning methods by achieving up to 10% better accuracy. In addition, our ablation study empirically demonstrates the effectiveness of our design choices for AUBER.

## What Preserves the Emergence of Language?

Authors: ['Anonymous']

Keywords: ['emergence of language', 'reinforcement learning']

The emergence of language is a mystery. One dominant theory is that cooperation boosts language to emerge. However, as a means of giving out information, language seems not to be an evolutionarily stable strategy. To ensure the survival advantage of many competitors, animals are selfish in nature. From the perspective of Darwinian, if an individual can obtain a higher benefit by deceiving the other party, why not deceive? For those who are cheated, once bitten and twice shy, cooperation will no longer be a good option. As a result, motivation for communication, as well as the emergence of language would perish. Then, what preserves the emergence of language? We aim to answer this question in a brand new framework of agent community, reinforcement learning, and natural selection. Empirically, we reveal that lying indeed dispels cooperation. Even with individual resistance to lying behaviors, liars can easily defeat truth tellers and survive during natural selection. However, social resistance eventually constrains lying and makes the emergence of language possible.

## Understanding and Leveraging Causal Relations in Deep Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Causal Relation']

Reinforcement Learning (RL) has shown great potential to deal with sequential decision-making problems. However, most RL algorithms do not explicitly consider the relations between entities in the environment. This makes the policy learning suffer from the problems of efficiency, effectivity and interpretability.  In this paper, we propose a novel deep reinforcement learning algorithm, which firstly learns the causal structure of the environment and then leverages the learned causal information to assist policy learning. The proposed algorithm learns a graph to encode the environmental structure by calculating Average Causal Effect (ACE) between different categories of entities, and an intrinsic reward is given to encourage the agent to interact more with critical entities of the causal graph, which leads to significant speed up of policy learning.  Several experiments are conducted on a number of simulation environments to demonstrate the effectiveness and better interpretability of our proposed method.

## Bridging the Imitation Gap by Adaptive Insubordination

Authors: ['Anonymous']

Keywords: ['Privileged Experts', 'Imitation Learning', 'Reinforcement Learning', 'Actor-Critic', 'Behavior Cloning', 'MiniGrid', 'Knowledge Distillation']

When expert supervision is available, practitioners often use imitation learning with varying degrees of success. We show that when an expert has access to privileged information that is unavailable to the student, this information is marginalized in the student policy during imitation learning resulting in an ''imitation gap'' and, potentially, poor results. Prior work bridges this gap via a progression from imitation learning to reinforcement learning. While often successful, gradual progression fails for tasks that require frequent switches between exploration and memorization skills. To better address these tasks and alleviate the imitation gap we propose 'Adaptive Insubordination' (ADVISOR), which dynamically weights imitation and reward-based reinforcement learning losses during training, enabling switching between imitation and exploration. On a suite of challenging didactic and MiniGrid tasks, we show that ADVISOR outperforms pure imitation, pure reinforcement learning, as well as their sequential and parallel combinations.

## Policy Gradient with Expected Quadratic Utility Maximization: A New Mean-Variance Approach in Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['mean-variance reinforcement learning', 'finance']

In real-world decision-making problems, risk management is critical. Among various risk management approaches, the mean-variance criterion is one of the most widely used in practice. In this paper, we suggest expected quadratic utility maximization (EQUM) as a new framework for policy gradient style reinforcement learning (RL) algorithms with mean-variance control. The quadratic utility function is a common objective of risk management in finance and economics. The proposed EQUM framework has several interpretations, such as reward-constrained variance minimization and regularization, as well as agent utility maximization. In addition, the computation of the EQUM framework is easier than that of existing mean-variance RL methods, which require double sampling. In experiments, we demonstrate the effectiveness of the proposed framework in benchmark setting of RL and financial data.

## Efficient Exploration for Model-based Reinforcement Learning with Continuous States and Actions

Authors: ['Anonymous']

Keywords: ['Model-based reinforcement learning', 'posterior sampling', 'Bayesian reinforcement learning']

Balancing exploration and exploitation is crucial in reinforcement learning (RL). In this paper, we study the model-based posterior sampling algorithm in continuous state-action spaces theoretically and empirically. First, we improve the regret bound: with the assumption that reward and transition functions can be modeled as Gaussian Processes with linear kernels, we develop a Bayesian regret bound of $\tilde{O}(H^{3/2}d\sqrt{T})$, where $H$ is the episode length, $d$ is the dimension of the state-action space, and $T$ is the time elapsed. Our bound can be extended to nonlinear cases as well: using linear kernels on the feature representation $\phi$, the Bayesian regret bound becomes $\tilde{O}(H^{3/2}d_{\phi}T)$, where $d_\phi$ is the dimension of the representation space. Moreover, we present MPC-PSRL, a model-based posterior sampling algorithm with model predictive control for action selection. To capture the uncertainty in models and realize posterior sampling, we use Bayesian linear regression on the penultimate layer (the feature representation layer $\phi$) of neural networks. Empirical results show that our algorithm achieves the best sample efficiency in benchmark control tasks compared to prior model-based algorithms, and matches the asymptotic performance of model-free algorithms. 

## Enforcing robust control guarantees within neural network policies

Authors: ['Anonymous']

Keywords: ['robust control', 'reinforcement learning', 'differentiable optimization']

When designing controllers for safety-critical systems, practitioners often face a challenging tradeoff between robustness and performance. While robust control methods provide rigorous guarantees on system stability under certain worst-case disturbances, they often result in simple controllers that perform poorly in the average (non-worst) case. In contrast, nonlinear control methods trained using deep learning have achieved state-of-the-art performance on many control tasks, but often lack robustness guarantees. In this paper, we propose a technique that combines the strengths of these two approaches: a generic nonlinear control policy class, parameterized by neural networks, that nonetheless enforces the same provable robustness criteria as robust control. Specifically, we show that by integrating custom convex-optimization-based projection layers into a nonlinear policy, we can construct a provably robust neural network policy class that outperforms robust control methods in the average (non-adversarial) setting. We demonstrate the power of this approach on several domains, improving in performance over existing robust control methods and in stability over (non-robust) RL methods.

## Rank the Episodes: A Simple Approach for Exploration in Procedurally-Generated Environments

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Exploration', 'Generalization of Reinforcement Learning', 'Self-Imitation']

Exploration under sparse reward is a long-standing challenge of model-free reinforcement learning. The state-of-the-art methods address this challenge by introducing intrinsic rewards to encourage exploration in novel states or uncertain environment dynamics. Unfortunately, the methods based on intrinsic rewards often fall short in procedurally-generated environments, where a different environment is generated in each episode so that the agent is not likely to visit the same state more than once. Motivated by how humans distinguish good exploration behaviors by looking into the entire episode, we introduce RAPID, a simple yet effective episode-level exploration method for procedurally-generated environments. RAPID regards each episode as a whole and gives an episodic exploration score from both per-episode and long-term views. Those highly scored episodes are treated as good exploration behaviors and are stored in a small ranking buffer. The agent then imitates the episodes in the buffer to reproduce the past good exploration behaviors. We demonstrate our method on several procedurally-generated MiniGrid environments, a first-person-view 3D Maze navigation task from MiniWorld, and several sparse MuJoCo tasks. The results show that RAPID significantly outperforms the state-of-the-art intrinsic reward strategies in terms of sample efficiency and final performance. The code is available at https://github.com/anonymousubmition/rapid

## Intrinsically Guided Exploration in Meta Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Meta reinforcement learning', 'Exploration', 'Information gain']

Deep reinforcement learning algorithms generally require large amounts of data to solve a single task. Meta reinforcement learning (meta-RL) agents learn to adapt to novel unseen tasks with high sample efficiency by extracting useful prior knowledge from previous tasks. Despite recent progress, efficient exploration in meta-training and adaptation remains a key challenge in sparse-reward meta-RL tasks. We propose a novel off-policy meta-RL algorithm to address this problem, which disentangles exploration and exploitation policies and learns intrinsically motivated exploration behaviors. We design novel intrinsic rewards derived from information gain to reduce task uncertainty and encourage the explorer to collect informative trajectories about the current task. Experimental evaluation shows that our algorithm achieves state-of-the-art performance on various sparse-reward MuJoCo locomotion tasks and more complex Meta-World tasks.

## Learning Movement Strategies for Moving Target Defense

Authors: ['Anonymous']

Keywords: ['Multi-agent Reinforcement Learning', 'Moving Target Defense', 'Stackelberg Security']

The field of cybersecurity has mostly been a cat-and-mouse game with the discovery of new attacks leading the way. To take away an attacker's advantage of reconnaissance, researchers have proposed proactive defense methods such as Moving Target Defense (MTD). To find good movement strategies, researchers have modeled MTD as leader-follower games between the defender and a cyber-adversary. We argue that existing models are inadequate in sequential settings when there is incomplete information about rational adversary and yield sub-optimal movement strategies. Further, while there exists an array of work on learning defense policies in sequential settings for cyber-security, they are either unpopular due to scalability issues arising out of incomplete information or tend to ignore the strategic nature of the adversary simplifying the scenario to use single-agent reinforcement learning techniques. To address these concerns, we propose (1) a unifying game-theoretic model, called the Bayesian Stackelberg Markov Games (BSMGs), that can model uncertainty over attacker types and the nuances of an MTD system and (2) a Bayesian Strong Stackelberg Q-learning (BSS-Q) approach that can, via interaction, learn the optimal movement policy for BSMGs within a reasonable time. We situate BSMGs in the landscape of incomplete-information Markov games and characterize the notion of Strong Stackelberg Equilibrium (SSE) in them. We show that our learning approach converges to an SSE of a BSMG and then highlight that the learned movement policy (1) improves the state-of-the-art in MTD for web-application security and (2) converges to an optimal policy in MTD domains with incomplete information about adversaries even when prior information about rewards and transitions is absent.

## Adaptive Discretization for Continuous Control using Particle Filtering Policy Network

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Continuous Control', 'Action Space Discretization', 'Policy Gradient']

Controlling the movements of highly articulated agents and robots has been a long-standing challenge to model-free deep reinforcement learning. In this paper, we propose a simple, yet general, framework for improving the performance of policy gradient algorithms by discretizing the continuous action space. Instead of using a fixed set of predetermined atomic actions, we exploit particle filtering to adaptively discretize actions during training and track the posterior policy distribution represented as a mixture of Gaussians. The resulting policy can replace the original continuous policy of any given policy gradient algorithm without changing its underlying model architecture. We demonstrate the applicability of our approach to state-of-the-art on-policy and off-policy baselines in challenging control tasks. Baselines using our particle-based policies achieve better final performance and speed of convergence as compared to corresponding continuous implementations and implementations that rely on fixed discretization schemes. 

## Learning Flexible Visual Representations via Interactive Gameplay

Authors: ['Anonymous']

Keywords: ['representation learning', 'deep reinforcement learning', 'computer vision']

A growing body of research suggests that embodied gameplay, prevalent not just in human cultures but across a variety of animal species including turtles and ravens, is critical in developing the neural flexibility for creative problem solving, decision making and socialization. Comparatively little is known regarding the impact of embodied gameplay upon artificial agents. While recent work has produced agents proficient in abstract games, these environments are far removed the real world and thus these agents can provide little insight into the advantages of embodied play. Hiding games, such as hide-and-seek, played universally, provide a rich ground for studying the impact of embodied gameplay on representation learning in the context of perspective taking, secret keeping, and false belief understanding. Here we are the first to show that embodied adversarial reinforcement learning agents playing Cache, a variant of hide-and-seek, in a high fidelity, interactive, environment, learn flexible representations of their observations encoding information such as object permanence, free space, and containment. Moving closer to biologically motivated learning strategies, our agents' representations, enhanced by intentionality and memory, are developed through interaction and play. These results serve as a model for studying how facets of vision develop through interaction, provide an experimental framework for assessing what is learned by artificial agents, and demonstrates the value of moving from large, static, datasets towards experiential, interactive, representation learning.

## Average Reward Reinforcement Learning with Monotonic Policy Improvement

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Average Reward', 'Policy Optimization', 'Model Free RL']

In continuing control tasks, an agent’s average reward per time step is a more natural performance measure compared to the commonly used discounting framework as it can better capture an agent’s long-term behavior.  We derive a novel lower bound on the difference of the average rewards for two policies, where the lower bound depends on the average divergence between the policies.  We show that previous work based on the discounted return (Schulman et al., 2015; Achiam et al.,2017) result in a trivial lower bound in the average reward setting.  We develop an iterative procedure based on our lower bound which produces a sequence of monotonically improved policies for the average reward criterion. When combined with deep reinforcement learning methods, the procedure leads to scalable and efficient algorithms aimed at maximizing an agent’s average reward performance. Empirically, we demonstrate the efficacy of our algorithms through a series of high-dimensional control tasks with long time horizons and show that discounting can lead to unsatisfactory performance on continuing control tasks.

## ERMAS: Learning Policies Robust to Reality Gaps in Multi-Agent Simulations

Authors: ['Anonymous']

Keywords: ['Robustness', 'Multi-Agent Learning', 'Sim2Real', 'Reinforcement Learning']

Policies for real-world multi-agent problems, such as optimal taxation, can be learned in multi-agent simulations with AI agents that emulate humans. However, simulations can suffer from reality gaps as humans often act suboptimally or optimize for different objectives (i.e., bounded rationality). We introduce $\epsilon$-Robust Multi-Agent Simulation (ERMAS), a robust optimization framework to learn AI policies that are robust to such multi-agent reality gaps. The objective of ERMAS theoretically guarantees robustness to the $\epsilon$-Nash equilibria of other agents – that is, robustness to behavioral deviations with a regret of at most$\epsilon$. ERMAS efficiently solves a first-order approximation of the robustness objective using meta-learning methods. We show that ERMAS yields robust policies for repeated bimatrix games and optimal adaptive taxation in economic simulations, even when baseline notions of robustness are uninformative or intractable. In particular, we show ERMAS can learn tax policies that are robust to changes in agent risk aversion, improving policy objectives (social welfare) by up to 15% in complex spatiotemporal simulations using the AI Economist (Zheng et al., 2020).

## FactoredRL: Leveraging Factored Graphs for Deep Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'factored mdp', 'factored rl']

We propose a simple class of deep reinforcement learning (RL) methods, called FactoredRL, that can leverage factored environment structures to improve the sample efficiency of existing model-based and model-free RL algorithms. In tabular and linear approximation settings, the factored Markov decision process literature has shown exponential improvements in sample efficiency by leveraging factored environment structures. We extend this to deep RL algorithms that use neural networks. For model-based algorithms, we use the factored structure to inform the state transition network architecture and for model-free algorithms we use the factored structure to inform the Q network or the policy network architecture.  We demonstrate that doing this significantly improves sample efficiency in both discrete and continuous state-action space settings.

## Asymmetric self-play for automatic goal discovery in robotic manipulation

Authors: ['Anonymous']

Keywords: ['self-play', 'asymmetric self-play', 'automatic curriculum', 'automatic goal generation', 'robotic learning', 'robotic manipulation', 'reinforcement learning']

We train a single, goal-conditioned policy that can solve many robotic manipulation tasks, including tasks with previously unseen goals and objects. To do so, we rely on asymmetric self-play for goal discovery, where two agents, Alice and Bob, play a game.  Alice is asked to propose challenging goals and Bob aims to solve them.  We show that this method is able to discover highly diverse and complex goals without any human priors.  We further show that Bob can be trained with only sparse rewards, because the interaction between Alice and Bob results in a natural curriculum and Bob can learn from Alice's trajectory when relabeled as a goal-conditioned demonstration.  Finally, we show that our method scales, resulting in a single policy that can transfer to many unseen hold-out tasks such as setting a table, stacking blocks, and solving simple puzzles.

## Maximum Reward Formulation In Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Theoretical Reinforcement Learning', 'Drug Discovery', 'Molecule Generation', 'de novo drug design']

Reinforcement learning (RL) algorithms typically deal with maximizing the expected cumulative return (discounted or undiscounted, finite or infinite horizon). However, several crucial applications in the real world, such as drug discovery, do not fit within this framework because an RL agent only needs to identify states (molecules) that achieve the highest reward within a trajectory and does not need to optimize for the expected cumulative return. In this work, we formulate an objective function to maximize the expected maximum reward along a trajectory, derive a novel functional form of the Bellman equation, introduce the corresponding Bellman operators, and provide a proof of convergence. Using this formulation, we achieve state-of-the-art results on the task of molecule generation that mimics a real-world drug discovery pipeline.

## D2RL: Deep Dense Architectures in Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement learning', 'Policy architectures']

While improvements in deep learning architectures have played a crucial role in improving the state of supervised and unsupervised learning in computer vision and natural language processing, neural network architecture choices for reinforcement learning remain relatively under-explored. We take inspiration from successful architectural choices in computer vision and generative modeling, and investigate the use of deeper networks and dense connections for reinforcement learning on a variety of simulated robotic learning benchmark environments. Our findings reveal that current methods benefit significantly from dense connections and deeper networks, across a suite of manipulation and locomotion tasks, for both proprioceptive and image-based observations. We hope that our results can serve as a strong baseline and further motivate future research into neural network architectures for reinforcement learning. 

## Conservative Safety Critics for Exploration

Authors: ['Anonymous']

Keywords: ['Safe exploration', 'Reinforcement Learning']

Safe exploration presents a major challenge in reinforcement learning (RL): when active data collection requires deploying partially trained policies, we must ensure that these policies avoid catastrophically unsafe regions, while still enabling trial and error learning. In this paper, we target the problem of safe exploration in RL, by learning a conservative safety estimate of environment states through a critic, and provably upper bound the likelihood of catastrophic failures at every training iteration. We theoretically characterize the tradeoff between safety and policy improvement, show that the safety constraints are satisfied with high probability during training, derive provable convergence guarantees for our approach which is no worse asymptotically then standard RL, and empirically demonstrate the efficacy of the proposed approach on a suite of challenging navigation, manipulation, and locomotion tasks. Our results demonstrate that the proposed approach can achieve competitive task performance, while incurring significantly lower catastrophic failure rates during training as compared to prior methods. Videos are at this URL https://sites.google.com/view/safe-exploration/

## Adaptive N-step Bootstrapping with Off-policy Data

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'policy evaluation']

The  definition  of  the  update  target  is  a  crucial  design  choice  in  reinforcement learning.  Due to the low computation cost and empirical high performance,  n-step returns with off-policy data is a widely used update target to bootstrap from scratch. A critical issue of applying n-step returns is to identify the optimal value of n.  In practice, n is often set to a fixed value, which is either determined by an empirical guess or by some hyper-parameter search.  In this work,  we point out that the optimal value of n actually differs on each data point, while the fixed value n is a rough average of them. The estimation error can be decomposed into two sources, off-policy bias and approximation error, and the fixed value of n is a trade-off between them.  Based on that observation, we introduce a new metric, policy age, to quantify the off-policyness of each data point.  We propose the Adaptive N-step Bootstrapping, which calculates the value of n for each data point by its policy age instead of the empirical guess. We conduct experiments on both MuJoCo and Atari games.  The results show that adaptive n-step bootstrap-ping achieves state-of-the-art performance in terms of both final reward and data efficiency.

## OPAL: Offline Primitive Discovery for Accelerating Offline Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Offline Reinforcement Learning', 'Primitive Discovery', 'Unsupervised Learning']

Reinforcement learning (RL) has achieved impressive performance in a variety of online settings in which an agent's ability to query the environment for transitions and rewards is effectively unlimited. However, in many practical applications, the situation is reversed: an agent may have access to large amounts of undirected offline experience data, while access to the online environment is severely limited. In this work, we focus on this offline setting. Our main insight is that, when presented with offline data composed of a variety of behaviors, an effective way to leverage this data is to extract a continuous space of recurring and temporally extended primitive behaviors before using these primitives for downstream task learning. Primitives extracted in this way serve two purposes: they delineate the behaviors that are supported by the data from those that are not, making them useful for avoiding distributional shift in offline RL; and they provide a degree of temporal abstraction, which reduces the effective horizon yielding better learning in theory, and improved offline RL in practice. In addition to benefiting offline policy optimization, we show that performing offline primitive learning in this way can also be leveraged for improving few-shot imitation learning as well as exploration and transfer in online RL on a variety of benchmark domains. Visualizations are available at https://sites.google.com/view/opal-iclr.

## Lyapunov Barrier Policy Optimization

Authors: ['Anonymous']

Keywords: ['Safe Reinforcement Learning', 'Deep Reinforcement Learning']

Deploying Reinforcement Learning (RL) agents in the real-world require that the agents satisfy safety constraints. Current RL agents explore the environment without considering these constraints, which can lead to damage to the hardware or even other agents in the environment. We propose a new method, LBPO, that uses a Lyapunov-based barrier function to restrict the policy update to a safe set for each training iteration. Our method also allows the user to control the conservativeness of the agent with respect to the constraints in the environment. LBPO significantly outperforms state-of-the-art baselines in terms of the number of constraint violations during training while being competitive in terms of performance.  Further, our analysis reveals that baselines like CPO and SDDPG rely mostly on backtracking to ensure safety rather than safe projection, which provides insight into why previous methods might not have effectively limit the number of constraint violations.

## Efficient Architecture Search for Continual Learning

Authors: ['Anonymous']

Keywords: ['continual learning', 'reinforcement learning', 'recurrent neural network', 'deep neural network']

Continual learning with neural networks is an important learning framework in AI that aims to learn a sequence of tasks well. However, it is often confronted with three challenges: (1) overcome the catastrophic forgetting problem, (2) adapt the current network to new tasks, and meanwhile (3) control its model complexity. To reach these goals, we propose a novel approach named as Continual Learning with Efficient Architecture Search, or CLEAS in short. CLEAS works closely with neural architecture search (NAS) which leverages reinforcement learning techniques to search for the best neural architecture that fits a new task. In particular, we design a neuron-level NAS controller that decides which old neurons from previous tasks should be reused (knowledge transfer), and which new neurons should be added (to learn new knowledge). Such a fine-grained controller allows finding a very concise architecture that can fit each new task well. Meanwhile, since we do not alter the weights of the reused neurons, we perfectly memorize the knowledge learned from previous tasks. We evaluate CLEAS on numerous sequential classification tasks, and the results demonstrate that CLEAS outperforms other state-of-the-art alternative methods, achieving higher classification accuracy while using simpler neural architectures.


## RODE: Learning Roles to Decompose Multi-Agent Tasks

Authors: ['Anonymous']

Keywords: ['Multi-Agent Reinforcement Learning', 'Role-Based Learning', 'Hierarchical Multi-Agent Learning', 'Multi-Agent Transfer Learning']

Role-based learning holds the promise of achieving scalable multi-agent learning by decomposing complex tasks using roles. However, it is largely unclear how to efficiently discover such a set of roles. To solve this problem, we propose to first decompose joint action spaces into restricted role action spaces by clustering actions according to their effects on the environment and other agents. Learning a role selector based on action effects makes role discovery much easier because it forms a bi-level learning hierarchy -- the role selector searches in a smaller role space and at a lower temporal resolution, while role policies learn in significantly reduced primitive action-observation spaces. We further integrate information about action effects into the role policies to boost learning efficiency and policy generalization. By virtue of these advances, our method (1) outperforms the current state-of-the-art MARL algorithms on 10 of the 14 scenarios that comprise the challenging StarCraft II micromanagement benchmark and (2) achieves rapid transfer to new environments with three times the number of agents. Demonstrative videos are available at https://sites.google.com/view/rode-marl .

## Scalable Bayesian Inverse Reinforcement Learning by Auto-Encoding Reward

Authors: ['Anonymous']

Keywords: ['Bayesian', 'Inverse reinforcement learning', 'Imitation Learning']

Bayesian inference over the reward presents an ideal solution to the ill-posed nature of the inverse reinforcement learning problem. Unfortunately current methods generally do not scale well beyond the small tabular setting due to the need for an inner-loop MDP solver, and even non-Bayesian methods that do themselves scale often require extensive interaction with the environment to perform well, being inappropriate for high stakes or costly applications such as healthcare. In this paper we introduce our method, Auto-encoded Variational Reward Imitation Learning (AVRIL), that addresses both of these issues by jointly learning an approximate posterior distribution over the reward that scales to arbitrarily complicated state spaces alongside an appropriate policy in a completely offline manner through a variational approach to said latent reward. Applying our method to real medical data alongside classic control simulations, we demonstrate Bayesian reward inference in environments beyond the scope of current methods, as well as task performance competitive with focused offline imitation learning algorithms.

## A Distributional Perspective on Actor-Critic Framework

Authors: ['Anonymous']

Keywords: ['Value distribution learning', 'reinforcement learning', 'deep learning', 'distributional reinforcement learning', 'distributional actor-critic']

Recent distributional reinforcement learning methods, despite their successes, still contain fundamental problems that can lead to inaccurate representations of value distributions, such as distributional instability, action type restriction, and biased approximation. In this paper, we present a novel distributional actor-critic frame-work, GMAC, to address such problems.  Adopting a stochastic policy removes the first two problems, and the bias in approximation is alleviated by minimizing the Cramer distance between the value distribution and its Bellman target distribution. In addition, GMAC improves data efficiency by generating the Bellman target distribution through Sample-Replacement algorithm, denoted by SR(λ), which provides a distributional generalization of multi-step policy evaluation algorithms.We empirically show that our method captures the multimodality of value distributions and improves the performance of conventional actor-critic methods with low computational cost in both discrete and continuous action spaces, using ArcadeLearning Environment (ALE) and PyBullet environment.

## Learning Safe Policies with Cost-sensitive Advantage Estimation

Authors: ['Anonymous']

Keywords: ['Safe Reinforcement Learning']

Reinforcement Learning (RL) with safety guarantee is critical for agents performing tasks in risky environments. Recent safe RL algorithms, developed based on Constrained Markov Decision Process (CMDP), mostly take the safety requirement as additional constraints when learning to maximize the return. However, they usually make unnecessary compromises in return for safety and only learn sub-optimal policies, due to the inability of differentiating safe and unsafe state-actions with high rewards. To address this, we propose Cost-sensitive Advantage Estimation (CSAE), which is simple to deploy for policy optimization and effective for guiding the agents to avoid unsafe state-actions by penalizing their advantage value properly. Moreover, for stronger safety guarantees, we develop a Worst-case Constrained Markov Decision Process (WCMDP) method to augment CMDP by constraining the worst-case safety cost instead of the average one. With CSAE and WCMDP, we develop new safe RL algorithms with theoretical justifications on their benefits for safety and performance of the obtained policies. Extensive experiments clearly demonstrate the superiority of our algorithms in learning safer and better agents under multiple settings. 

## TEAC: Intergrating Trust Region and Max Entropy Actor Critic for Continuous Control

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Trust region methods', 'Maximum Entropy Reinforcement Learning', 'Deep Reinforcement Learning']

Trust region methods and maximum entropy methods are two state-of-the-art branches used in reinforcement learning (RL) for the benefits of stability and exploration in continuous environments, respectively. This paper proposes to integrate both branches in a unified framework, thus benefiting from both sides. We first transform the original RL objective to a constraint optimization problem and then proposes trust entropy actor critic (TEAC), an off-policy algorithm to learn stable and sufficiently explored policies for continuous states and actions. TEAC trains the critic by minimizing the refined Bellman error and updates the actor by minimizing KL-divergence loss derived from the closed-form solution to the Lagrangian. 
We prove that the policy evaluation and policy improvement in TEAC is guaranteed to converge.
Extensive experiments on the tasks in the MuJoCo environment show that TEAC outperforms state-of-the-art solutions in terms of efficiency and effectiveness.


## Causal Curiosity: RL Agents Discovering Self-supervised Experiments for Causal Representation Learning

Authors: ['Anonymous']

Keywords: ['Causal Representation Learning', 'Unsupervised/Self-Supervised Reinforcement Learning']

Humans show an innate ability to learn the regularities of the world through interaction. By performing experiments in our environment, we are able to discern the causal factors of variation and infer how they affect the dynamics of our world. Analogously, here we attempt to equip reinforcement learning agents with the ability to perform experiments that facilitate a categorization of the rolled-out trajectories, and to subsequently infer the causal factors of the environment in a hierarchical manner. We introduce a novel intrinsic reward, called causal curiosity, and show that it allows our agents to learn optimal sequences of actions, and to discover causal factors in the dynamics. The learned behavior allows the agent to infer a binary quantized representation for the ground-truth causal factors in every environment. Additionally, we find that these experimental behaviors are semantically meaningful (e.g., to differentiate between heavy and light blocks, our agents learn to lift them), and are learnt in a self-supervised manner with approximately 2.5 times less data than conventional supervised planners. We show that these behaviors can be re-purposed and fine-tuned (e.g., from lifting to pushing or other downstream tasks). Finally, we show that the knowledge of causal factor representations aids zero-shot learning for more complex tasks.

## Embedding a random graph via GNN: mean-field inference theory and RL applications to NP-Hard multi-robot/machine scheduling

Authors: ['Anonymous']

Keywords: ['Graph neural network', 'graph embedding', 'multi-robot/machine scheduling', 'Reinforcement learning', 'Mean-field inference']

We develop a theory for embedding a random graph using graph neural networks (GNN) and illustrate its capability to solve NP-hard scheduling problems. We apply the theory to address the challenge of developing a near-optimal learning algorithm to solve the NP-hard problem of scheduling multiple robots/machines with time-varying rewards. In particular, we consider a class of reward collection problems called Multi-Robot Reward Collection (MRRC). Such MRRC problems well model ride-sharing, pickup-and-delivery, and a variety of related problems. We consider the classic identical parallel machine scheduling problem (IPMS) in the Appendix.  

For the theory, we first observe that MRRC system state can be represented as an extension of probabilistic graphical models (PGMs), which we refer to as random PGMs. We then develop a mean-field inference method for random PGMs. 
We prove that a simple modification of a typical GNN embedding is sufficient to embed a random graph even when the edge presence probabilities are interdependent.

Our theory enables a two-step hierarchical inference for precise and transferable Q-function estimation for MRRC and IPMS. For scalable computation, we show that the transferability of Q-function estimation enables us to design a polynomial-time algorithm with 1-1/e optimality bound. 
Experimental results on solving NP-hard MRRC problems (and IMPS in the Appendix) highlight the near-optimality and transferability of the proposed methods.  


## Greedy-GQ with Variance Reduction: Finite-time Analysis and Improved Complexity

Authors: ['Anonymous']

Keywords: ['Optimization', 'Reinforcement Learning', 'Machine Learning']

Greedy-GQ is a value-based reinforcement learning (RL) algorithm for optimal control. Recently, the finite-time analysis of Greedy-GQ has been developed under linear function approximation and Markovian sampling, and the algorithm is shown to achieve an $\epsilon$-stationary point with a sample complexity in the order of $\mathcal{O}(\epsilon^{-3})$. Such a high sample complexity is due to the large variance induced by the Markovian samples. In this paper, we propose a variance-reduced Greedy-GQ (VR-Greedy-GQ) algorithm for off-policy optimal control. In particular, the algorithm applies the SVRG-based variance reduction scheme to reduce the stochastic variance of the two time-scale updates. We study the finite-time convergence of VR-Greedy-GQ under linear function approximation and Markovian sampling and show that the algorithm achieves a much smaller bias and variance error than the original Greedy-GQ. In particular, we prove that VR-Greedy-GQ achieves an improved sample complexity that is in the order of $\mathcal{O}(\epsilon^{-2})$. We further compare the performance of VR-Greedy-GQ with that of Greedy-GQ in various RL experiments to corroborate our theoretical findings.

## Large Batch Simulation for Deep Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'simulation']

We introduce a reinforcement learning system that trains agents in photo-realistic 3D simulated environments at over 19,000 frames of experience per second on a single GPU (and up to 72,000 frames per second on a single eight-GPU machine) – more than 100x faster than prior work in the same environments.  Our core innovation is a custom simulator design that accepts and executes large batches of simulation requests simultaneously.  Batch simulation allows in-memory storage of scene assets, rendering work, and synchronization costs to be shared/amortized across many simulation requests, dramatically improving the number of simulated agents per GPU and overall simulation throughput.  To keep up with faster simulation, we build a computationally efficient policy DNN that maintains high task performance with reduced inference and training costs, and modify training algorithms to maintain sample efficiency when training with large mini-batches.  By combining batch simulation,  inference,  and training performance optimizations with multi-GPU parallelism, we demonstrate that PointGoal navigation agents can be trained in photo-realistic 3D environments on a single GPU in 1.5 days to 97% of the accuracy of agents trained on a prior state-of-the-art system using a 64-GPU cluster over three days. This increase in performance enables rapid exploration of reinforcement learning algorithms in regimes that were previously prohibitively slow and computationally expensive.

## Efficient Reinforcement Learning in Factored MDPs with Application to Constrained RL

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'factored MDP', 'constrained RL', 'learning theory']

Reinforcement learning (RL) in episodic, factored Markov decision processes (FMDPs) is studied. We propose an algorithm called FMDP-BF, which leverages the factorization structure of FMDP.  The regret of FMDP-BF is shown to be exponentially smaller than that of optimal algorithms designed for non-factored MDPs, and improves on the best previous result for FMDPs~\citep{osband2014near} by a factor of $\sqrt{nH|\mathcal{S}_i|}$, where $|\mathcal{S}_i|$ is the cardinality of the factored state subspace, $H$ is the planning horizon and $n$ is the number of factored transition. To show the optimality of our bounds, we also provide a lower bound for FMDP, which indicates that our algorithm is near-optimal w.r.t. timestep $T$, horizon $H$ and factored state-action subspace cardinality. Finally, as an application, we study a new formulation of constrained RL, known as RL with knapsack constraints (RLwK), and provides the first sample-efficient algorithm based on FMDP-BF.

## Human-Level Performance in No-Press Diplomacy via Equilibrium Search

Authors: ['Anonymous']

Keywords: ['multi-agent systems', 'regret minimization', 'no-regret learning', 'game theory', 'reinforcement learning']

Prior AI breakthroughs in complex games have focused on either the purely adversarial or purely cooperative settings. In contrast, Diplomacy is a game of shifting alliances that involves both cooperation and competition. For this reason, Diplomacy has proven to be a formidable research challenge. In this paper we describe an agent for the no-press variant of Diplomacy that combines supervised learning on human data with one-step lookahead search via external regret minimization. External regret minimization techniques have been behind previous AI successes in adversarial games, most notably poker, but have not previously been shown to be successful in large-scale games involving cooperation. We show that our agent greatly exceeds the performance of past no-press Diplomacy bots, is unexploitable by expert humans, and achieves a rank of 23 out of 1,128 human players when playing anonymous games on a popular Diplomacy website.

## FORK: A FORward-looKing Actor for Model-Free Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Actor Critic', 'Policy Gradient', 'Model Free']

In this paper, we propose a new type of Actor, named forward-looking Actor or FORK for short, for Actor-Critic algorithms. FORK can be easily integrated into a model-free Actor-Critic algorithm. Our experiments on six Box2D and MuJoCo environments with continuous state and action spaces demonstrate significant performance improvement FORK can bring to the state-of-the-art algorithms. A variation of FORK can further solve BipedalWalkerHardcore in as few as four hours using a single GPU.

## Offline Adaptive Policy Leaning in Real-World Sequential Recommendation Systems

Authors: ['Anonymous']

Keywords: ['Reinforcement learning', 'Recommendation system']

The training process of RL requires many trial-and-errors that are costly in real-world applications. To avoid the cost, a promising solution is to learn the policy from an offline dataset, e.g., to learn a simulator from the dataset, and train optimal policies in the simulator. By this approach, the quality of policies highly relies on the fidelity of the simulator. Unfortunately, due to the stochasticity and unsteadiness of the real-world and the unavailability of online sampling, the distortion of the simulator is inevitable. In this paper, based on the model learning technique, we propose a new paradigm to learn an RL policy from offline data in the real-world sequential recommendation system (SRS). Instead of increasing the fidelity of models for policy learning, we handle the distortion issue via learning to adapt to diverse simulators generated by the offline dataset. The adaptive policy is suitable to real-world environments where dynamics are changing and have stochasticity in the offline setting. Experiments are conducted in synthetic environments and a real-world ride-hailing platform. The results show that the method overcomes the distortion problem and produces robust recommendations in the unseen real-world.

## Convex Regularization in Monte-Carlo Tree Search

Authors: ['Anonymous']

Keywords: ['Monte-Carlo Tree Search', 'Entropy regularization', 'Reinforcement Learning']

Monte-Carlo planning and Reinforcement Learning (RL) are essential to sequential decision making. The recent AlphaGo and AlphaZero algorithms have shown how to successfully combine these two paradigms in order to solve large scale sequential decision problems. These methodologies exploit a variant of the well-known UCT algorithm to trade off exploitation of good actions and exploration of unvisited states, but their empirical success comes at the cost of poor sample-efficiency and high computation time. In this paper, we overcome these limitations by studying the benefit of convex regularization in Monte-Carlo Tree Search (MCTS) to efficiently drive exploration and to improve policy updates, as already observed in RL. First, we introduce a unifying theory on the use of generic convex regularizers in MCTS, deriving the first regret analysis of regularized MCTS and showing that it guarantees exponential convergence rate. Second, we exploit our theoretical framework to introduce novel regularized backup operators for MCTS, based on the relative entropy of the policy update, and on the Tsallis entropy of the policy. Finally, we show how our framework can easily be incorporated in AlphaGo and AlphaZero, and we empirically show the superiority of convex regularization w.r.t. representative baselines, on well-known RL problems and across several Atari games.

## Learning with AMIGo: Adversarially Motivated Intrinsic Goals

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'exploration', 'meta-learning']

A key challenge for reinforcement learning (RL) consists of learning in environments with sparse extrinsic rewards. In contrast to current RL methods, humans are able to learn new skills with little or no reward by using various forms of intrinsic motivation. We propose AMIGo, a novel agent incorporating a goal-generating teacher that proposes Adversarially Motivated Intrinsic Goals to train a goal-conditioned student'' policy in the absence of (or alongside) environment reward. Specifically, through a simple but effective ``constructively adversarial'' objective, the teacher learns to propose increasingly challenging---yet achievable---goals that allow the student to learn general skills for acting in a new environment, independent of the task to be solved. We show that our method generates a natural curriculum of self-proposed goals which ultimately allows the agent to solve challenging procedurally-generated tasks where other forms of intrinsic motivation and state-of-the-art RL methods fail.

## Meta-Reinforcement Learning With Informed Policy Regularization

Authors: ['Anonymous']

Keywords: ['meta-reinforcement learning', 'reinforcement learning', 'multi-task', 'non-stationary', 'task representations', 'regularization']

Meta-reinforcement learning aims at finding a policy able to generalize to new environments. When facing a new environment, this policy must explore to identify its particular characteristics and then exploit this information for collecting reward.  Even though policies based on recurrent neural networks can be used in this setting by training them on multiple environments, they often fail to model this trade-off, or solve it at a very high computational cost. In this paper, we propose a new algorithm that uses privileged information in the form of a task descriptor at train time to improve the learning of recurrent policies.  Our method learns an informed policy (i.e., a policy receiving as input the description of the current task) that is used to both construct task embeddings from the descriptors, and to regularize the training of the recurrent policy through parameters sharing and an auxiliary objective. This approach significantly reduces the learning sample complexity without altering the representational power of RNNs, by focusing on the relevant characteristics of the task, and by exploiting them efficiently.  We evaluate our algorithm in a variety of environments that require sophisticated exploration/exploitation strategies and show that it outperforms vanilla RNNs, Thompson sampling and the task-inference approaches to meta-reinforcement learning.

## Measuring Progress in Deep Reinforcement Learning Sample Efficiency 

Authors: ['Anonymous']

Keywords: ['Deep Reinforcement Learning', 'Sample Efficiency']

Sampled environment transitions are a critical input to deep reinforcement learning (DRL) algorithms. Current DRL benchmarks often allow for the cheap and easy generation of large amounts of samples such that perceived progress in DRL does not necessarily correspond to improved sample efficiency. As simulating real world processes is often prohibitively hard and collecting real world experience is costly, sample efficiency is an important indicator for economically relevant applications of DRL. We investigate progress in sample efficiency on Atari games and continuous control tasks by comparing the amount of samples that a variety of algorithms need to reach a given performance level according to training curves in the corresponding publications. We find exponential progress in sample efficiency with estimated doubling times of around 10 to 18 months on Atari, 5 to 24 months on state-based continuous control and of around 4 to 9 months on pixel-based continuous control depending on the specific task and performance level.

## Prioritized Level Replay

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Procedurally Generated Environments', 'Curriculum Learning', 'Procgen Benchmark']

Simulated environments with procedurally generated content have become popular benchmarks for testing systematic generalization of reinforcement learning agents. Every level in such an environment is algorithmically created, thereby exhibiting a unique configuration of underlying factors of variation, such as layout, positions of entities, asset appearances, or even the rules governing environment transitions. Fixed sets of training levels can be determined to aid comparison and reproducibility, and test levels can be held out to evaluate the generalization and robustness of agents. While prior work samples training levels in a direct way (e.g. uniformly) for the agent to learn from, we investigate the hypothesis that different levels provide different learning progress for an agent at specific times during training. We introduce Prioritized Level Replay, a general framework for estimating the future learning potential of a level given the current state of the agent's policy.  We find that temporal-difference (TD) errors, while previously used to selectively sample past transitions, also proves effective for scoring a level's future learning potential of entire episodes that an agent would experience when replaying it. We report significantly improved sample-efficiency and generalization on the majority of Procgen Benchmark environments as well as two challenging MiniGrid environments. Lastly, we present a qualitative analysis showing that Prioritized Level Replay induces an implicit curriculum, taking the agent gradually from easier to harder levels. 

## Fast MNAS: Uncertainty-aware Neural Architecture Search with Lifelong Learning

Authors: ['Anonymous']

Keywords: ['Neural Architecture Search', 'AutoML', 'Reinforcement Learning (RL)']

Sampling-based neural architecture search (NAS) always guarantees better convergence yet suffers from huge computational resources compared with gradient-based approaches, due to the rollout bottleneck -- exhaustive training for each sampled generation on proxy tasks. This work provides a general pipeline to accelerate the convergence of the rollout process as well as the RL learning process in sampling-based NAS. It is motivated by the interesting observation that both the architecture and the parameter knowledge can be transferred between different experiments and even different tasks. We first introduce an uncertainty-aware critic (value function) in PPO to utilize the architecture knowledge in previous experiments, which stabilizes the training process and reduces the searching time by 4 times. Further, a life-long knowledge pool together with a block similarity function is proposed to utilize the lifelong parameter knowledge and reduces the searching time by 2 times. It is the first to introduce block-level weight sharing in RL-based NAS. The block similarity function guarantees a 100% hitting ratio with strict fairness.  Besides, we show a simply designed off-policy correction factor that enables 'replay buffer' in RL optimization and further reduces half of the searching time. Experiments on the MNAS search space show the proposed FNAS accelerates standard RL-based NAS process by $\sim$10x (e.g. $\sim$256 2x2 TPUv2*days / 20,000 GPU*hour $\rightarrow$ 2,000 GPU*hour for MNAS), and guarantees better performance on various vision tasks.

## Error Controlled Actor-Critic Method to Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'actor-critic', 'function approximation', 'approximation error', 'KL divergence']

In the reinforcement learning (RL) algorithms which incorporate function approximation methods, the approximation error of value function inevitably cause overestimation phenomenon and have a negative impact on the convergence of the algorithms. To mitigate the negative effects of approximation error, we propose a new actor-critic algorithm called Error Controlled Actor-critic which ensures confining the approximation error in value function. In this paper, we firstly present an analysis of how the approximation error can hinder the optimization process of actor-critic methods. Then, we derive an upper boundary of the approximation error of $Q$ function approximator, and found that the error can be lowered by placing restrictions on the KL-divergence between every two consecutive policies during the training phase of the policy. The results of experiments on a range of continuous control tasks from OpenAI gym suite demonstrate that the proposed actor-critic algorithm apparently reduces the approximation error and significantly outperforms other model-free RL algorithms.

## Adaptive Dataset Sampling by Deep Policy Gradient

Authors: ['Anonymous']

Keywords: ['dataset sampling', 'batch selection', 'mini-batch SGD', 'reinforcement learning', 'policy gradient', 'optimal sample sequence']

Mini-batch SGD is a predominant optimization method in deep learning. Several works aim to improve naïve random dataset sampling, which appears typically in deep learning literature, with additional prior to allow faster and better performing optimization. This includes, but not limited to, importance sampling and curriculum learning. In this work, we propose an alternative way: we think of sampling as a trainable agent and let this external model learn to sample mini-batches of training set items based on the current status and recent history of the learned model. The resulting adaptive dataset sampler, named RLSampler, is a policy network implemented with simple recurrent neural networks trained by a policy gradient algorithm. We demonstrate RLSampler on image classification benchmarks with several different learner architectures and show consistent performance gain over the originally reported scores. Moreover, either a pre-sampled sequence of indices or a pre-trained RLSampler turns out to be more effective than naïve random sampling regardless of the network initialization and model architectures. Our analysis reveals the possible existence of a model-agnostic sample sequence that best represents the dataset under mini-batch SGD optimization framework.

## What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study

Authors: ['Anonymous']

Keywords: ['Reinforcement learning', 'continuous control']

In recent years, reinforcement learning (RL) has been successfully applied to many different continuous control tasks. While RL algorithms are often conceptually simple, their state-of-the-art implementations take numerous low- and high-level design decisions that strongly affect the performance of the resulting agents. Those choices are usually not extensively discussed in the literature, leading to discrepancy between published descriptions of algorithms and their implementations. This makes it hard to attribute progress in RL and slows down overall progress [Engstrom'20]. As a step towards filling that gap, we implement >50 such ``"choices" in a unified on-policy deep actor-critic framework, allowing us to investigate their impact in a large-scale empirical study. We train over 250'000 agents in five continuous control environments of different complexity and provide insights and practical recommendations for the training of on-policy deep actor-critic RL agents.

## Diverse Exploration via InfoMax Options

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Hierachical Reinforcement Learning', 'Exploration']

In this paper, we study the problem of autonomously discovering temporally abstracted actions, or options, for exploration in reinforcement learning. For learning diverse options suitable for exploration, we introduce the infomax termination objective defined as the mutual information between options and their corresponding state transitions. We derive a scalable optimization scheme for maximizing this objective via the termination condition of options, yielding the InfoMax Option Critic (IMOC) algorithm. Through illustrative experiments, we empirically show that IMOC learns diverse options and utilizes them for exploration. Moreover, we show that IMOC scales well to continuous control tasks.


## Hierarchical Reinforcement Learning by Discovering Intrinsic Options

Authors: ['Anonymous']

Keywords: ['hierarchical reinforcement learning', 'reinforcement learning', 'options', 'unsupervised skill discovery', 'exploration']

We propose a hierarchical reinforcement learning method, HIDIO, that can learn task-agnostic options in a self-supervised manner while jointly learning to utilize them to solve sparse-reward tasks. Unlike current hierarchical RL approaches that tend to formulate a goal-reaching framework or pre-define ad hoc lower-level policies, HIDIO encourages lower-level option learning that is independent of the task at hand, requiring no assumptions or knowledge about the task structure. These options are learned through an intrinsic entropy minimization objective conditioned on the option trajectories. The learned options are diverse, shortsighted, and non task-driven. In experiments on robotic manipulation and navigation tasks with sparse rewards, HIDIO achieves higher success rates with greater sample efficiency than regular RL baselines and two state-of-the-art hierarchical RL methods.

## Robust Reinforcement Learning using Adversarial Populations

Authors: ['Anonymous']

Keywords: ['Robust Control', 'Reinforcement Learning', 'Multiagent Systems']

Reinforcement Learning (RL) is an effective tool for controller design but can struggle with issues of robustness, failing catastrophically when the underlying system dynamics are perturbed. The Robust RL formulation tackles this by adding worst-case adversarial noise to the dynamics and constructing the noise distribution as the solution to a zero-sum minimax game. However, existing work on learning solutions to the Robust RL formulation has primarily focused on training a single RL agent against a single adversary. In this work, we demonstrate that using a single adversary does not consistently yield robustness to dynamics variations under standard parametrizations of the adversary; the resulting policy is highly exploitable by new adversaries. We propose a population-based augmentation to the Robust RL formulation in which we randomly initialize a population of adversaries and sample from the population uniformly during training. We empirically validate across robotics benchmarks that the use of an adversarial population results in a less exploitable, more robust policy.  Finally, we demonstrate that this approach provides comparable robustness and generalization as domain randomization on these benchmarks while avoiding a ubiquitous domain randomization failure mode.

## Preventing Value Function Collapse in Ensemble  Q-Learning by Maximizing Representation Diversity

Authors: ['Anonymous']

Keywords: ['Ensemble Q-Learning', 'Representation Diversity', 'Reinforcement Learning']

The first deep RL algorithm, DQN, was limited by the overestimation bias of the learned Q-function. Subsequent algorithms proposed techniques to reduce this problem, without fully eliminating it. Recently, the Maxmin and Ensemble Q-learning algorithms used the different estimates provided by ensembles of learners to reduce the bias. Unfortunately, these learners can converge to the same point in the parametric or representation space, falling back to the classic single neural network DQN. In this paper, we describe a regularization technique to maximize diversity in the representation space in these algorithms. We propose and compare five regularization functions inspired from economics theory and consensus optimization. We show that the resulting approach significantly outperforms the Maxmin and Ensemble Q-learning algorithms as well as non-ensemble baselines.

## Cross-State Self-Constraint for Feature Generalization in Deep Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'generalization', 'regularization']

Representation learning on visualized input is an important yet challenging task for deep reinforcement learning (RL). The feature space learned from visualized input not only dominates the agent's generalization ability in new environments but also affect the data efficiency during training. To help the RL agent learn general and discriminative representation among various states, we present cross-state self-constraint(CSSC), a novel constraint that regularizes the representation feature space by comparing similarity of different pairs of representations. Based on the representation-behavior connection derived from the agent's experience, this constraint helps reinforce the general feature recognition during the learning process and thus enhance the generalization to unseen environment. We test our proposed method on the OpenAI ProcGen benchmark and see significant improvement on generalization performance across most of ProcGen games.

## Rapid Task-Solving in Novel Environments

Authors: ['Anonymous']

Keywords: ['deep reinforcement learning', 'meta learning', 'deep learning', 'exploration', 'planning']

We propose the challenge of rapid task-solving in novel environments (RTS), wherein an agent must solve a series of tasks as rapidly as possible in an unfamiliar environment. An effective RTS agent must balance between exploring the unfamiliar environment and solving its current task, all while building a model of the new environment over which it can plan when faced with later tasks. While modern deep RL agents exhibit some of these abilities in isolation, none are suitable for the full RTS challenge. To enable progress toward RTS, we introduce two challenge domains: (1) a minimal RTS challenge called the Memory\&Planning Game and (2) One-Shot StreetLearn Navigation, which introduces scale and complexity from real-world data. We demonstrate that state-of-the-art deep RL agents fail at RTS in both domains, and that this failure is due to an inability to plan over gathered knowledge. We develop Episodic Planning Networks (EPNs) and show that deep-RL agents with EPNs excel at RTS, outperforming the nearest baseline by factors of 2-3 and learning to navigate held-out StreetLearn maps within a single episode. We show that EPNs learn to execute a value iteration-like planning algorithm and that they generalize to situations beyond their training experience.

## Autonomous Learning of Object-Centric Abstractions for High-Level Planning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'planning', 'PDDL', 'multitask', 'transfer', 'objects']

We propose a method for autonomously learning an object-centric representation of a continuous and high-dimensional environment that is suitable for planning. Such representations can immediately be transferred between tasks that share the same types of objects, resulting in agents that require fewer samples to learn a model of a new task. We first demonstrate our approach on a simple domain where the agent learns a compact, lifted representation that generalises across objects. We then apply it to a series of Minecraft tasks to learn object-centric representations, including object types—directly from pixel data—that can be leveraged to solve new tasks quickly. The resulting learned representations enable the use of a task-level planner, resulting in an agent capable of forming complex, long-term plans with considerably fewer environment interactions.

## Temporal Difference Uncertainties as a Signal for Exploration

Authors: ['Anonymous']

Keywords: ['deep reinforcement learning', 'deep-rl', 'exploration']

An effective approach to exploration in reinforcement learning is to rely on an agent's uncertainty over the optimal policy, which can yield near-optimal exploration strategies in tabular settings. However, in non-tabular settings that involve function approximators, obtaining accurate uncertainty estimates is almost as challenging a problem. In this paper, we highlight that value estimates are easily biased and temporally inconsistent. In light of this, we propose a novel method for estimating uncertainty over the value function that relies on inducing a distribution over temporal difference errors. This exploration signal controls for state-action transitions so as to isolate uncertainty in value that is due to uncertainty over the agent's parameters. Because our measure of uncertainty conditions on state-action transitions, we cannot act on this measure directly. Instead, we incorporate it as an intrinsic reward and treat exploration as a separate learning problem, induced by the agent's temporal difference uncertainties. We introduce a distinct exploration policy that learns to collect data with high estimated uncertainty, which gives rise to a curriculum that smoothly changes throughout learning and vanishes in the limit of perfect value estimates. We evaluate our method on hard exploration tasks, including Deep Sea and Atari 2600 environments and find that our proposed form of exploration facilitates both diverse and deep exploration.

## Dream and Search to Control: Latent Space Planning for Continuous Control

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Model Based RL', 'Continuous Control', 'Search', 'Planning', 'MCTS']

Learning and planning with latent space dynamics has been shown to be useful for sample efficiency in model-based reinforcement learning (MBRL) for discrete and continuous control tasks. In particular, recent work, for discrete action spaces, demonstrated the effectiveness of latent-space planning via Monte-Carlo Tree Search (MCTS) for bootstrapping MBRL during learning and at test time. However, the potential gains from latent-space tree search have not yet been demonstrated for environments with continuous action spaces.  In this work, we propose and explore an MBRL approach for continuous action spaces based on tree-based planning over learned latent dynamics.  We show that it is possible to demonstrate the types of bootstrapping benefits as previously shown for discrete spaces. In particular, the approach achieves improved sample efficiency and performance on a majority of challenging continuous-control benchmarks compared to the state-of-the-art. 

## A REINFORCEMENT LEARNING FRAMEWORK FOR TIME DEPENDENT CAUSAL EFFECTS EVALUATION IN A/B TESTING

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'A/B testing', 'causal inference', 'sequential testing']

A/B testing, or online experiment is a standard business strategy to compare a new product with an old one in pharmaceutical, technological, and traditional industries. The aim of this paper is to introduce a reinforcement learn- ing framework for carrying A/B testing in two-sided marketplace platforms, while characterizing the long-term treatment effects. Our proposed testing procedure allows for sequential monitoring and online updating. It is generally applicable to a variety of treatment designs in different industries. In addition, we systematically investigate the theoretical properties (e.g., size and power) of our testing procedure. Finally, we apply our framework to both synthetic data and a real-world data example obtained from a technological company to illustrate its advantage over the current practice. 


## Learning Intrinsic Symbolic Rewards in Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Intrinsic Rewards', 'Symbolic Regression']

Learning effective policies for sparse objectives is a key challenge in Deep Reinforcement Learning (RL). A common approach is to design task-related dense rewards to improve task learnability. While such rewards are easily interpreted, they rely on heuristics and domain expertise. Alternate approaches that train neural networks to discover dense surrogate rewards avoid heuristics, but are high-dimensional, black-box solutions offering little interpretability. In this paper, we present a method that discovers dense rewards in the form of low-dimensional symbolic trees - thus making them more tractable for analysis. The trees use simple functional operators to map an agent's observations to a scalar reward, which then supervises the policy gradient learning of a neural network policy. We test our method on continuous action spaces in Mujoco and discrete action spaces in Atari and Pygames environments. We show that the discovered dense rewards are an effective signal for an RL policy to solve the benchmark tasks. Notably, we significantly outperform a widely used, contemporary neural-network based reward-discovery algorithm in all environments considered.

## Safety Aware Reinforcement Learning (SARL)

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Safe RL', 'Probabilistic Distance Metrics']

As reinforcement learning agents become increasingly integrated into complex, real-world environments, designing for safety becomes a critical consideration. We specifically focus on researching scenarios where agents can cause undesired side effects while executing a policy on a primary task. Since one can define multiple tasks for a given environment dynamics, there are two important challenges. First, we need to abstract the concept of safety that applies broadly to that environment independent of the specific task being executed. Second, we need a mechanism for the abstracted notion of safety to modulate the actions of agents executing different policies to minimize their side-effects. In this work, we propose Safety Aware Reinforcement Learning (SARL) - a framework where a virtual safe agent modulates the actions of a main reward-based agent to minimize side effects. The safe agent learns a task-independent notion of safety for a given environment. The main agent is then trained with a regularization loss given by the distance between the native action probabilities of the two agents. Since the safe agent effectively abstracts a task-independent notion of safety via its action probabilities, it can be ported to modulate multiple policies solving different tasks within the given environment without further training. We contrast this with solutions that rely on task-specific regularization metrics and test our framework on the SafeLife Suite, based on Conway's Game of Life, comprising a number of complex tasks in dynamic environments. We show that our solution is able to match the performance of solutions that rely on task-specific side-effect penalties on both the primary and safety objectives while additionally providing the benefit of generalizability and portability. 

## Which Mutual-Information Representation Learning Objectives are Sufficient for Control?

Authors: ['Anonymous']

Keywords: ['representation learning', 'reinforcement learning', 'information theory']

Mutual information maximization provides an appealing formalism for learning representations of data. In the context of reinforcement learning, such representations can accelerate learning by discarding irrelevant and redundant information, while retaining the information necessary for control. Much of the prior work on these methods has addressed the practical difficulties of estimating mutual information from samples of high-dimensional observations, while comparatively less is understood about \emph{which} mutual information objectives are sufficient for RL from a theoretical perspective. In this paper we identify conditions under which representations that maximize specific mutual-information objectives are theoretically sufficient for learning and representing the optimal policy. Somewhat surprisingly, we find that several popular objectives can yield insufficient representations given mild and common assumptions on the structure of the MDP. We corroborate our theoretical results with deep RL experiments on a simulated game environment with visual observations.

## Model-Based Offline Planning

Authors: ['Anonymous']

Keywords: ['off-line reinforcement learning', 'model-based reinforcement learning', 'model-based control', 'reinforcement learning', 'model predictive control', 'robotics']

Offline learning is a key part of making reinforcement learning (RL) useable in real systems. Offline RL looks at scenarios where there is data from a system's operation, but no direct access to the system when learning a policy. Recent work on training RL policies from offline data has shown results both with model-free policies learned directly from the data, or with planning on top of learnt models of the data. Model-free policies tend to be more performant, but are more opaque, harder to command externally, and less easy to integrate into larger systems. We propose an offline learner that generates a model that can be used to control the system directly through planning. This allows us to have easily controllable policies directly from data, without ever interacting with the system. We show the performance of our algorithm, Model-Based Offline Planning (MBOP) on a series of robotics-inspired tasks, and demonstrate its ability leverage planning to respect environmental constraints. We are able to find near-optimal polices for certain simulated systems from as little as 50 seconds of real-time system interaction, and create zero-shot goal-conditioned policies on a series of environments.

## Modelling Hierarchical Structure between Dialogue Policy and Natural Language Generator with Option Framework for Task-oriented Dialogue System

Authors: ['Anonymous']

Keywords: ['Task-oriented Dialogue System', 'Natural Language Processing', 'Policy Optimization', 'Hierarchical Reinforcement Learning']

Designing task-oriented dialogue systems is a challenging research topic, since it needs not only to generate utterances fulfilling user requests but also to guarantee the comprehensibility. Many previous works trained end-to-end (E2E) models with supervised learning (SL), however, the bias in annotated system utterances remains as a bottleneck. Reinforcement learning (RL) deals with the problem through using non-differentiable evaluation metrics (e.g., the success rate) as rewards. Nonetheless, existing works with RL showed that the comprehensibility of generated system utterances could be corrupted when improving the performance on fulfilling user requests. In our work, we (1) propose modelling the hierarchical structure between dialogue policy and natural language generator (NLG) with the option framework, called HDNO, where the latent dialogue act is applied to avoid designing specific dialogue act representations; (2) train HDNO via hierarchical reinforcement learning (HRL), as well as suggest the asynchronous updates between dialogue policy and NLG during training to theoretically guarantee their convergence to a local maximizer; and (3) propose using a discriminator modelled with language models as an additional reward to further improve the comprehensibility. We test HDNO on MultiWoz 2.0 and MultiWoz 2.1, the datasets on multi-domain dialogues, in comparison with word-level E2E model trained with RL, LaRL and HDSA, showing improvements on the performance evaluated by automatic evaluation metrics and human evaluation. Finally, we demonstrate the semantic meanings of latent dialogue acts to show the ability of explanation.

## Toward Synergism in Macro Action Ensembles

Authors: ['Anonymous']

Keywords: ['Automated machine Learning', 'Reinforcement learning', 'Macro action ensemble']

Macro actions have been demonstrated to be beneficial for the learning processes of an agent. A variety of techniques have been developed to construct more effective macro actions. However, they usually fail to provide an approach for combining macro actions to form a synergistic macro action ensemble. A synergistic macro action ensemble performs better than individual macro actions within it. Motivated by the recent advances of neural architecture search, we formulate the construction of a synergistic macro action ensemble as a sequential decision problem and evaluate the ensemble in a task. The formulation of sequential decision problem enables coherency in the macro actions to be considered. Also, our evaluation procedure takes synergism into account since the synergism among the macro action ensemble exhibits when jointly used by an agent. The experimental results show that our framework is able to discover synergistic macro action ensembles. We further perform experiments to validate the synergism property among the macro actions in an ensemble.

## UneVEn: Universal Value Exploration for Multi-Agent Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['multi-agent reinforcement learning', 'deep Q-learning', 'universal value functions', 'successor features', 'relative overgeneralization']

This paper focuses on cooperative value-based multi-agent reinforcement learning (MARL) in the paradigm of centralized training with decentralized execution (CTDE). Current state-of-the-art value-based MARL methods leverage CTDE to learn a centralized joint-action value function as a monotonic mixing of each agent's utility function, which enables easy decentralization. However, this monotonic restriction leads to inefficient exploration in tasks with nonmonotonic returns due to suboptimal approximations of the values of joint actions. To address this, we present a novel MARL approach called Universal Value Exploration (UneVEn), which uses universal successor features (USFs) to learn policies of tasks related to the target task, but with simpler reward functions in a sample efficient manner. UneVEn uses novel action-selection schemes between randomly sampled related tasks during exploration, which enables the monotonic joint-action value function of the target task to place more importance on useful joint actions. Empirical results on a challenging cooperative predator-prey task requiring significant coordination amongst agents show that UneVEn significantly outperforms state-of-the-art baselines.

## Learning from Demonstrations with Energy based Generative Adversarial Imitation Learning

Authors: ['Anonymous']

Keywords: ['Learning from Demonstrations', 'Energy based Models', 'Inverse Reinforcement Learning', 'Imitation Learning']

Traditional reinforcement learning methods usually deal with the tasks with explicit reward signals. However, for vast majority of cases, the environment wouldn't feedback a reward signal immediately. It turns out to be a bottleneck for modern reinforcement learning approaches to be applied into more realistic scenarios. Recently, inverse reinforcement learning has made great progress in making full use of the expert demonstrations to recover the reward signal for reinforcement learning. And generative adversarial imitation learning is one promising approach. In this paper, we propose a new architecture for training generative adversarial imitation learning which is so called energy based generative adversarial imitation learning (EB-GAIL). It views the discriminator as an energy function that attributes low energies to the regions near the expert demonstrations and high energies to other regions. Therefore, a generator can be seen as a reinforcement learning procedure to sample trajectories with minimal energies (cost), while the discriminator is trained to assign high energies to these generated trajectories. In detail, EB-GAIL uses an auto-encoder architecture in place of the discriminator, with the energy being the reconstruction error. Theoretical analysis shows our EB-GAIL could match the occupancy measure with expert policy during the training process. Meanwhile, the experiments depict that EB-GAIL outperforms other SoTA methods while the training process for EB-GAIL can be more stable.

## Quantifying Differences in Reward Functions

Authors: ['Anonymous']

Keywords: ['rl', 'irl', 'reward learning', 'distance', 'benchmarks']

For many tasks, the reward function is too complex to be specified procedurally, and must instead be learned from user data. Prior work has evaluated learned reward functions by examining rollouts from a policy optimized for the learned reward. However, this method cannot distinguish between the learned reward function failing to reflect user preferences, and the reinforcement learning algorithm failing to optimize the learned reward. Moreover, the rollout method is highly sensitive to details of the environment the learned reward is evaluated in, which often differ in the deployment environment. To address these problems, we introduce the Equivalent-Policy Invariant Comparison (EPIC) distance to quantify the difference between two reward functions directly, without training a policy. We prove EPIC is invariant on an equivalence class of reward functions that always induce the same optimal policy. Furthermore, we find EPIC can be precisely approximated and is more robust than baselines to the choice of visitation distribution. Finally, we show that EPIC distance bounds the regret of optimal policies even under different transition dynamics, and confirm empirically that it predicts policy training success. Our source code is available at — double blind: supplementary material —.

## Efficient Fully-Offline Meta-Reinforcement Learning via Distance Metric Learning and Behavior Regularization

Authors: ['Anonymous']

Keywords: ['offline/batch reinforcement learning', 'meta-reinforcement learning', 'distance metric learning']

We study the offline meta-reinforcement learning (OMRL) problem, a paradigm which enables reinforcement learning (RL) algorithms to quickly adapt to unseen tasks without any interactions with the environments, making RL truly practical in many real-world applications. This problem is still not fully understood, for which two major challenges need to be addressed. First, offline RL often suffers from bootstrapping errors of out-of-distribution state-actions which leads to divergence of value functions. Second, meta-RL requires efficient and robust task inference learned jointly with control policy. In this work, we enforce behavior regularization on learned policy as a general approach to offline RL, combined with a deterministic context encoder for efficient task inference. We propose a novel negative-power distance metric on bounded context embedding space, whose gradients propagation is detached from that of the Bellman backup. We provide analysis and insight showing that some simple design choices can yield substantial improvements over recent approaches involving meta-RL and distance metric learning.  To the best of our knowledge, our method is the first model-free and end-to-end OMRL algorithm, which is computationally efficient and demonstrated to outperform prior algorithms on several meta-RL benchmarks.


## Deep Reinforcement Learning With Adaptive Combined Critics

Authors: ['Anonymous']

Keywords: ['overestimation', 'continuous control', 'deep reinforcement learning', 'policy improvement']

The overestimation problem has long been popular in deep value learning, because function approximation errors may lead to amplified value estimates and suboptimal policies. There have been several methods to deal with the overestimation problem, however, further problems may be induced, for example, the underestimation bias and instability. In this paper, we focus on the overestimation issues on continuous control through deep reinforcement learning, and propose a novel algorithm that can minimize the overestimation, avoid the underestimation bias and retain the policy improvement during the whole training process. Specifically, we add a weight factor to adjust the influence of two independent critics, and use the combined value of weighted critics to update the policy. Then the updated policy is involved in the update of the weight factor, in which we propose a novel method to provide theoretical and experimental guarantee for future policy improvement. We evaluate our method on a set of classical control tasks, and the results show that the proposed algorithms are more computationally efficient and stable than several existing algorithms for continuous control.

## BRAC+: Going Deeper with Behavior Regularized Offline Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['offline reinforcement learning', 'behavior regularization']

Online interactions with the environment to collect data samples for training a Reinforcement Learning agent is not always feasible due to economic and safety concerns. The goal of Offline Reinforcement Learning (RL) is to address this problem by learning effective policies using previously collected datasets. Standard off-policy RL algorithms are prone to overestimations of the values of out of distribution (less explored) actions and are hence unsuitable for Offline RL. Behavior regularization, which constraints the learned policy within the support set of the dataset, has been proposed to tackle the limitations of standard off-policy algorithms. In this paper, we improve the behavior regularized offline reinforcement learning and propose \emph{BRAC+}. We use an analytical upper bound on KL divergence as the behavior regularizor to reduce variance associated with sample based estimations. Additionally, we employ state-dependent Lagrange multipliers for the regularization term to avoid distributing KL divergence penalty across all states of the sampled batch. The proposed Lagrange multipliers allow more freedom of deviation to high probability (more explored) states leading to better rewards while simultaneously restricting low probability (least explored) states to prevent out of distribution actions. We also propose several practical enhancements to further improve the performance. On challenging locomotion offline RL benchmarks, BRAC+ matches the state-of-the-art approaches on single-modal datasets and outperforms them on multi-modal datasets. 

## Transfer among Agents: An Efficient Multiagent Transfer Learning Framework

Authors: ['Anonymous']

Keywords: ['Multiagent learning', 'transfer learning', 'reinforcement learning']

Transfer Learning has shown great potential to enhance the single-agent Reinforcement Learning (RL) efficiency, by sharing learned policies of previous tasks. Similarly, in multiagent settings, the learning performance can also be promoted if agents can share knowledge between each other. However, it remains an open question of how an agent should learn from other agents' knowledge. In this paper, we propose a novel multiagent option-based policy transfer (MAOPT) framework to improve multiagent learning efficiency. Our framework learns what advice to give to each agent and when to terminate it by modeling multiagent policy transfer as the option learning problem. MAOPT provides different kinds of variants which can be classified into two types in terms of the experience used during training. One type is the MAOPT with the Global Option Advisor which has the access to the global information of the environment. However, in many realistic scenarios, we can only obtain each agent's local information due to the partial observation. The other type contains MAOPT with the Local Option Advisor and MAOPT with the Successor Representation Option (SRO) which are suitable for this setting and collect each agent's local experience for the update. In many cases, each agent's experience is inconsistent with each other which causes the option-value estimation to oscillate and to become inaccurate. SRO is used to handle the experience inconsistency by decoupling the dynamics of the environment from the rewards to learn the option-value function under each agent's preference. MAOPT can be easily combined with existing deep RL approaches. Experimental results show it significantly boosts the performance of existing deep RL methods in both discrete and continuous state spaces.

## A Sharp Analysis of Model-based Reinforcement Learning with Self-Play

Authors: ['Anonymous']

Keywords: ['Reinforcement learning theory', 'Markov games', 'model-based RL', 'task-agnostic RL', 'multi-agent RL']

Model-based algorithms---algorithms that explore the environment through building and utilizing an estimated model---are widely used in reinforcement learning practice and theoretically shown to achieve optimal sample efficiency for single-agent reinforcement learning in Markov Decision Processes (MDPs). However, for multi-agent reinforcement learning in Markov games, the current best known sample complexity for model-based algorithms is rather suboptimal and compares unfavorably against recent model-free approaches. In this paper, we present a sharp analysis of model-based self-play algorithms for multi-agent Markov games. We design an algorithm \emph{Optimistic Nash Value Iteration} (Nash-VI) for two-player zero-sum Markov games that is able to output an $\epsilon$-approximate Nash policy in $\tilde{\mathcal{O}}(H^3SAB/\epsilon^2)$ episodes of game playing, where $S$ is the number of states, $A,B$ are the number of actions for the two players respectively, and $H$ is the horizon length. This significantly improves over the best known model-based guarantee of $\tilde{\mathcal{O}}(H^4S^2AB/\epsilon^2)$, and is the first that matches the information-theoretic lower bound $\Omega(H^3S(A+B)/\epsilon^2)$ except for a $\min\{A,B\}$ factor. In addition, our guarantee compares favorably against the best known model-free algorithm if $\min\{A,B\}=o(H^3)$, and outputs a single Markov policy while existing sample-efficient model-free algorithms output a nested mixture of Markov policies that is in general non-Markov and rather inconvenient to store and execute. We further adapt our analysis to designing a provably efficient task-agnostic algorithm for zero-sum Markov games, and designing the first line of provably sample-efficient algorithms for multi-player general-sum Markov games.

## Learning to Search for Fast Maximum Common Subgraph Detection

Authors: ['Anonymous']

Keywords: ['graph matching', 'maximum common subgraph', 'graph neural network', 'reinforcement learning', 'search']

Detecting the Maximum Common Subgraph (MCS) between two input graphs is fundamental for applications in biomedical analysis, malware detection, cloud computing, etc. This is especially important in the task of drug design, where the successful extraction of common substructures in compounds can reduce the number of experiments needed to be conducted by humans. However, MCS computation is NP-hard, and state-of-the-art MCS solvers rely on heuristics in search which in practice cannot find good solution for large graph pairs under a limited search budget. Here we propose GLSearch, a Graph Neural Network based model for MCS detection, which learns to search. Our model uses a state-of-the-art branch and bound algorithm as the backbone search algorithm to extract subgraphs by selecting one node pair at a time. In order to make better node selection decision at each step, we replace the node selection heuristics with a novel task-specific Deep Q-Network (DQN), allowing the search process to find larger common subgraphs faster. To enhance the training of DQN, we leverage the search process to provide supervision in a pre-training stage and guide our agent during an imitation learning stage. Therefore, our framework allows search and reinforcement learning to mutually benefit each other. Experiments on synthetic and real-world large graph pairs demonstrate that our model outperforms state-of-the-art MCS solvers and neural graph matching network models.

## Explainable Reinforcement Learning Through Goal-Based Explanations

Authors: ['Anonymous']

Keywords: ['explainable reinforcement learning', 'hierarchical reinforcement learning', 'goal-based explanations']

Many algorithms in Reinforcement Learning rely on neural networks to achieve state-of-the-art performance, but this has the cost of making the agents black-boxes, hard to interpret and understand, making their use difficult in trusted applications, such as robotics or industrial applications. Our key contribution to improve explainability is introducing goal-based explanations, a new explanation mechanism where the agent produces goals and attempts to reach those goals one-by-one while maximizing the collected reward. These goals form the agent's plan to solve the task, explaining the purpose of its current actions (reach the current goal) and predicting its future behavior. To obtain the agent's goals without domain knowledge, we use 2-layer hierarchical agents where the top layer produces goals and the bottom layer attempts to reach those goals. The goals produced by trained hierarchical agent form clear and reliable explanations that can be visualized to make them easier to understand for non-experts.

Hierarchical agents are more explainable but are difficult to train: Hindsight Actor-Critic (HAC), a  state-of-the-art algorithm, fails to train the agent in many environments. As an additional contribution, we generalize it and create HAC-General with Teacher, which maximizes the rewards collected from the environment, does not require the environment to provide an end-goal, and vastly improves training by leveraging a black-box agent and using more complex goals composed of a state $s$ to be reached and a reward $r$ to be collected. Our experiments show HAC-General with Teacher can train agents successfully in environments where HAC fails (even if it is helped by knowing the desired end-goal), making it possible to create explainable agents in more settings. 

## Mutual Information State Intrinsic Control

Authors: ['Anonymous']

Keywords: ['Intrinsically Motivated Reinforcement Learning', 'Intrinsic Reward', 'Intrinsic Motivation', 'Deep Reinforcement Learning', 'Reinforcement Learning']

Reinforcement learning has been shown to be highly successful at many challenging tasks. However, success heavily relies on well-shaped rewards. Intrinsically motivated RL attempts to remove this constraint by defining an intrinsic reward function. Motivated by the self-consciousness concept in psychology, we make a natural assumption that the agent knows what constitutes itself, and propose a new intrinsic objective that encourages the agent to have maximum control on the environment. We mathematically formalize this reward as the mutual information between the agent state and the surrounding state under the current agent policy. With this new intrinsic motivation, we are able to outperform previous methods, including being able to complete the pick-and-place task for the first time without using any task reward. A video showing experimental results is available in the supplementary material.

## UPDeT: Universal Multi-agent RL via Policy Decoupling with Transformers

Authors: ['Anonymous']

Keywords: ['Multi-agent Reinforcement Learning', 'Transfer Learning']

Recent advances in multi-agent reinforcement learning have been largely limited in training one model from scratch for every new task. The limitation is due to the restricted model architecture related to fixed input and output dimensions. This hinders the experience accumulation and transfer of the learned agent over tasks with diverse levels of difficulty (e.g. 3 vs 3 or 5 vs 6 multi-agent games).  In this paper, we make the first attempt to explore a universal multi-agent reinforcement learning pipeline, designing one single architecture to fit tasks with the requirement of different observation and action configurations. Unlike previous RNN-based models, we utilize a transformer-based model to generate a flexible policy by decoupling the policy distribution from the intertwined input observation with an importance weight measured by the merits of the self-attention mechanism. Compared to a standard transformer block, the proposed model, named as Universal Policy Decoupling Transformer (UPDeT), further relaxes the action restriction and makes the multi-agent task's decision process more explainable. UPDeT is general enough to be plugged into any multi-agent reinforcement learning pipeline and equip them with strong generalization abilities that enables the handling of multiple tasks at a time. Extensive experiments on large-scale SMAC multi-agent competitive games demonstrate that the proposed UPDeT-based multi-agent reinforcement learning achieves significant results relative to state-of-the-art approaches, demonstrating advantageous transfer capability in terms of both performance and training speed (10 times faster).

## Learning Subgoal Representations with Slow Dynamics

Authors: ['Anonymous']

Keywords: ['Hierarchical Reinforcement Learning', 'Representation Learning', 'Exploration']

In goal-conditioned Hierarchical Reinforcement Learning (HRL), a high-level policy periodically sets subgoals for a low-level policy, and the low-level policy is trained to reach those subgoals. A proper subgoal representation function, which abstracts a state space to a latent subgoal space, is crucial for effective goal-conditioned HRL, since different low-level behaviors are induced by reaching subgoals in the compressed representation space. Observing that the high-level agent operates at an abstract temporal scale, we propose a slowness objective to effectively learn the subgoal representation (i.e., the high-level action space). We provide a theoretical grounding for the slowness objective. That is, selecting slow features as the subgoal space can achieve efficient hierarchical exploration. As a result of better exploration ability, our approach significantly outperforms state-of-the-art HRL and exploration methods on a number of benchmark continuous-control tasks. Thanks to the generality of the proposed subgoal representation learning method, empirical results also demonstrate that the learned representation and corresponding low-level policies can be transferred between distinct tasks.

## NeurWIN: Neural Whittle Index Network for Restless Bandits via Deep RL

Authors: ['Anonymous']

Keywords: ['deep reinforcement learning', 'restless bandits', 'Whittle index']

Whittle index policy is a powerful tool to obtain asymptotically optimal solutions for the notoriously intractable problem of restless bandits. However, finding the Whittle indices remains a difficult problem for many practical restless bandits with convoluted transition kernels. This paper proposes NeurWIN, a neural Whittle index network that seeks to learn the Whittle indices for any restless bandits by leveraging mathematical properties of the Whittle indices. We show that a neural network that produces the Whittle index is also one that produces the optimal control for a set of Markov decision problems. This property motivates using deep reinforcement learning for the training of NeurWIN. We demonstrate the utility of NeurWIN by evaluating its performance for three recently studied restless bandit problems. Our experiment results show that the performance of NeurWIN is either better than, or as good as, state-of-the-art policies for all three problems.

## Cross-Modal Domain Adaptation for Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Domain Adaptation', 'Reinforcement Learning']

Domain adaptation is a promising direction for deploying RL agents in real-world applications, where vision-based robotics tasks constitute an important part. Cur-rent methods that train polices on simulated images not only require a delicately crafted simulator, but also add extra burdens to the training process. In this paper, we propose a method that can learn a mapping from high-dimensional images to low-level simulator states, allowing agents trained on the source domain of state input to transfer well to the target domain of image input. By fully leveraging the sequential information in the trajectories and incorporating the policy to guide the training process, our method overcomes the intrinsic ill-posedness in cross-modal domain adaptation when structural constraints from the same modality are unavailable. Experiments on MuJoCo environments show that the policy, once combined with the mapping function,  can be deployed directly in the target domain with only a small performance gap, while current methods designed for same-modal domain adaptation fail on this problem.

## Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'policy search', 'offline RL', 'control']

In this work, we aim to develop a simple and scalable reinforcement learning algorithm that uses standard supervised learning methods as subroutines, while also being able to leverage off-policy data. Our proposed approach, which we refer to as advantage-weighted regression (AWR), consists of two standard supervised learning steps: one to regress onto target values for a value function, and another to regress onto weighted target actions for the policy. The method is simple and general, can accommodate continuous and discrete actions, and can be implemented in just a few lines of code on top of standard supervised learning methods. We provide a theoretical motivation for AWR and analyze its properties when incorporating off-policy data from experience replay. We evaluate AWR on a suite of standard OpenAI Gym benchmark tasks, and show that it achieves competitive performance compared to a number of well-established state-of-the-art RL algorithms. AWR is also able to acquire more effective policies than most off-policy algorithms when learning from purely static datasets with no additional environmental interactions. Furthermore, we demonstrate our algorithm on challenging continuous control tasks with highly complex simulated characters.

## Correcting experience replay for multi-agent communication

Authors: ['Anonymous']

Keywords: ['multi-agent reinforcement learning', 'experience replay', 'communication', 'relabelling']

We consider the problem of learning to communicate using multi-agent reinforcement learning (MARL). A common approach is to learn off-policy, using data sampled from a replay buffer. However, messages received in the past may not accurately reflect the current communication policy of each agent, and this complicates learning. We therefore introduce a 'communication correction' which accounts for the non-stationarity of observed communication induced by multi-agent learning. It works by relabelling the received message to make it likely under the communicator's current policy, and thus be a better reflection of the receiver's current environment. To account for cases in which agents are both senders and receivers, we introduce an ordered relabelling scheme. Our correction is computationally efficient and can be integrated with a range of off-policy algorithms. It substantially improves the ability of communicating MARL systems to learn across a variety of cooperative and competitive tasks.

## Multi-Agent Trust Region Learning

Authors: ['Anonymous']

Keywords: ['multi-agent learning', 'reinforcement learning', 'empirical game-theoretic analysis']

Trust-region methods are widely used in single-agent reinforcement learning.  One advantage is that they guarantee a lower bound of monotonic payoff improvement for policy optimization at each iteration. Nonetheless, when applied in multi-agent settings, such guarantee is lost because an agent's payoff is also determined by other agents' adaptive behaviors.  In fact, measuring agents' payoff improvements in multi-agent reinforcement learning (MARL) scenarios is still challenging. Although game-theoretical solution concepts such as Nash equilibrium can be applied, the algorithm (e.g., Nash-Q learning) suffers from poor scalability beyond two-player discrete games. To mitigate the above measurability and tractability issues, in this paper, we propose Multi-Agent Trust Region Learning (MATRL) method. MATRL augments the single-agent trust-region optimization process with the multi-agent solution concept of stable fixed point that is computed at the policy-space meta-game level. When multiple agents learn simultaneously, stable fixed points at the meta-game level can effectively measure agents' payoff improvements, and, importantly, a meta-game representation enjoys better scalability for multi-player games.  We derive the lower bound of agents' payoff improvements for MATRL methods, and also prove the convergence of our method on the meta-game fixed points. We evaluate the MATRL method on both discrete and continuous multi-player general-sum games; results suggest that MATRL significantly outperforms strong MARL baselines on grid worlds, multi-agent MuJoCo, and Atari games. 

## Batch Reinforcement Learning Through Continuation Method

Authors: ['Anonymous']

Keywords: ['batch reinforcement learning', 'continuation method', 'relaxed regularization']

Many real-world applications of reinforcement learning (RL) require the agent to learn from a fixed set of trajectories, without collecting new interactions.  Policy optimization under this setting is extremely challenging as: 1) the geometry of the objective function is hard to optimize efficiently; 2) the shift of data distributions causes high noise in the value estimation. In this work, we propose a simple yet effective policy iteration approach to batch RL using global optimization techniques known as continuation.  By constraining the difference between the learned policy and the behavior policy that generates the fixed trajectories, and continuously relaxing the constraint, our method 1) helps the agent escape local optima; 2) reduces the error in policy evaluation in the optimization procedure.   We present results on a variety of control tasks, game environments, and a recommendation task to empirically demonstrate the efficacy of our proposed method.

## RMIX: Risk-Sensitive Multi-Agent Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['Risk-sensitive learning', 'cooperative multi-agent reinforcement learning', 'reinforcement learning']

Centralized training with decentralized execution (CTDE) has become an important paradigm in multi-agent reinforcement learning (MARL). Current CTDE-based methods rely on restrictive decompositions of the centralized value function across agents, which decomposes the global Q-value into individual Q values to guide individuals' behaviours. However, such expected, i.e., risk-neutral, Q value decomposition is not sufficient even with CTDE due to the randomness of rewards and the uncertainty in environments, which causes the failure of these methods to train coordinating agents in complex environments. To address these issues, we propose RMIX, a novel cooperative MARL method with the Conditional Value at Risk (CVaR) measure over the learned distributions of individuals' Q values. Our main contributions are in three folds: (i) We first learn the return distributions of individuals to analytically calculate CVaR for decentralized execution; (ii) We then propose a dynamic risk level predictor for CVaR calculation to handle the temporal nature of the stochastic outcomes during executions; (iii) We finally propose risk-sensitive Bellman equation along with Individual-Global-MAX (IGM) for MARL training. Empirically, we show that our method significantly outperforms state-of-the-art methods on many challenging StarCraft II tasks, demonstrating significantly enhanced coordination and high sample efficiency.

## Greedy Multi-Step Off-Policy Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['off-policy learning', 'multi-step reinforcement learning', 'Q learning']

This paper presents a novel multi-step reinforcement learning algorithms, named Greedy Multi-Step Value Iteration (GM-VI), under off-policy setting. GM-VI iteratively approximates the optimal value function of a given environment using a newly proposed multi-step bootstrapping technique, in which the step size is adaptively adjusted along each trajectory according to a greedy principle. With the improved multi-step information propagation mechanism, we show that the resulted VI process is capable of safely learning from arbitrary behavior policy without additional off-policy correction. We further analyze the theoretical properties of the corresponding operator, showing that it is able to converge to globally optimal value function, with a rate faster than traditional Bellman Optimality Operator. Experiments reveal that the proposed methods is reliable, easy to implement and achieves state-of-the-art performance on a series of standard benchmark datasets.

## Decoupling Representation Learning from Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'representation learning', 'unsupervised learning']

In an effort to overcome limitations of reward-driven feature learning in deep reinforcement learning (RL) from images, we propose decoupling representation learning from policy learning.  To this end, we introduce a new unsupervised learning (UL) task, called Augmented Temporal Contrast (ATC), which trains a convolutional encoder to associate pairs of observations separated by a short time difference, under image augmentations and using a contrastive loss.  In online RL experiments, we show that training the encoder exclusively using ATC matches or outperforms end-to-end RL in most environments.  Additionally, we benchmark several leading UL algorithms by pre-training encoders on expert demonstrations and using them, with weights frozen, in RL agents; we find that agents using ATC-trained encoders outperform all others.  We also train multi-task encoders on data from multiple environments and show generalization to different downstream RL tasks.  Finally, we ablate components of ATC, and introduce a new data augmentation to enable replay of (compressed) latent images from pre-trained encoders when RL requires augmentation.  Our experiments span visually diverse RL benchmarks in DeepMind Control, DeepMind Lab, and Atari, and our complete code is available at \url{hidden url}.

## REPAINT: Knowledge Transfer in Deep Actor-Critic Reinforcement Learning

Authors: ['Anonymous']

Keywords: ['reinforcement learning', 'transfer learning', 'actor-critic RL', 'representation transfer', 'instance transfer', 'task similarity', 'MuJoCo', 'DeepRacer']

Accelerating the learning processes for complex tasks by leveraging previously learned tasks has been one of the most challenging problems in reinforcement learning, especially when the similarity between source and target tasks is low or unknown. In this work, we propose a REPresentation-And-INstance Transfer algorithm (REPAINT) for deep actor-critic reinforcement learning paradigm. In representation transfer, we adopt a kickstarted training method using a pre-trained teacher policy by introducing an auxiliary cross-entropy loss. In instance transfer, we develop a sampling approach, i.e., advantage-based experience replay, on transitions collected following the teacher policy, where only the samples with high advantage estimates are retained for policy update. We consider both learning an unseen target task by transferring from previously learned teacher tasks and learning a partially unseen task composed of multiple sub-tasks by transferring from a pre-learned teacher sub-task. In several benchmark experiments, REPAINT significantly reduces the total training time and improves the asymptotic performance compared to training with no prior knowledge and other baselines.

## Policy-Driven Attack: Learning to Query for Hard-label Black-box Adversarial Examples

Authors: ['Anonymous']

Keywords: ['hard-label attack', 'black-box attack', 'adversarial attack', 'reinforcement learning']

To craft black-box adversarial examples, adversaries need to query the victim model and take proper advantage of its feedback. Existing black-box attacks generally suffer from high query complexity, especially when only the top-1 decision (i.e., the hard-label prediction) of the victim model is available. In this paper, we propose a novel hard-label black-box attack named Policy-Driven Attack, to reduce the query complexity. Our core idea is to learn promising search directions of the adversarial examples using a well-designed policy network in a novel reinforcement learning formulation, in which the queries become more sensible. Experimental results demonstrate that our method can significantly reduce the query complexity in comparison with existing state-of-the-art hard-label black-box attacks on various image classification benchmark datasets. Anonymous code and models for reproducing our results are available in the supplementary material.

## Hierarchical Meta Reinforcement Learning for Multi-Task Environments

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning', 'Multi-task', 'Hierarchical', 'Meta Learning']

Deep reinforcement learning algorithms aim to achieve human-level intelligence by solving practical decisions-making problems, which are often composed of multiple sub-tasks. Complex and subtle relationships between sub-tasks make traditional methods hard to give a promising solution. We implement a first-person shooting environment with random spatial structures to illustrate a typical representative of this kind. A desirable agent should be capable of balancing between different sub-tasks: navigation to find enemies and shooting to kill them. To address the problem brought by the environment, we propose a Meta Soft Hierarchical reinforcement learning framework (MeSH), in which each low-level sub-policy focuses on a specific sub-task respectively and high-level policy automatically learns to utilize low-level sub-policies through meta-gradients. The proposed framework is able to disentangle multiple sub-tasks and discover proper low-level policies under different situations. The effectiveness and efficiency of the framework are shown by a series of comparison experiments. Both environment and algorithm code will be provided for open source to encourage further research.

## Stochastic Inverse Reinforcement Learning 

Authors: ['Anonymous']

Keywords: ['Inverse Reinforcement Learning', 'Stochastic Methods', 'MCEM']

The goal of the inverse reinforcement learning (IRL) problem is to recover the reward functions from expert demonstrations. However, the IRL problem like any ill-posed inverse problem suffers the congenital defect that the policy may be optimal for many reward functions, and expert demonstrations may be optimal for many policies. In this work, we generalize the IRL problem to a well-posed expectation optimization problem stochastic inverse reinforcement learning (SIRL) to recover the probability distribution over reward functions. We adopt the Monte Carlo expectation-maximization (MCEM) method to estimate the parameter of the probability distribution as the first solution to the SIRL problem. The solution is succinct, robust, and transferable for a learning task and can generate alternative solutions to the IRL problem. Through our formulation, it is possible to observe the intrinsic property for the IRL problem from a global viewpoint, and our approach achieves a considerable performance on the objectworld. 

## Mixture of Step Returns in Bootstrapped DQN

Authors: ['Anonymous']

Keywords: ['Reinforcement Learning']

The concept of utilizing multi-step returns for updating value functions has been adopted in deep reinforcement learning (DRL) for a number of years. Updating value functions with different backup lengths provides advantages in different aspects, including bias and variance of value estimates, convergence speed, and exploration behavior of the agent. Conventional methods such as TD-lambda leverage these advantages by using a target value equivalent to an exponential average of different step returns. Nevertheless, integrating step returns into a single target sacrifices the diversity of the advantages offered by different step return targets. To address this issue, we propose Mixture Bootstrapped DQN (MB-DQN) built on top of bootstrapped DQN, and uses different backup lengths for different bootstrapped heads. MB-DQN enables heterogeneity of the target values that is unavailable in approaches relying only on a single target value. As a result, it is able to maintain the advantages offered by different backup lengths. In this paper, we first discuss the motivational insights through a simple maze environment. In order to validate the effectiveness of MB-DQN, we perform experiments on the Atari 2600 benchmark environments and demonstrate the performance improvement of MB-DQN over a number of baseline methods. We further provide a set of ablation studies to examine the impacts of different design configurations of MB-DQN.

