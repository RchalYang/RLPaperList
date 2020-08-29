# Spotlight

## Influence-Based Multi-Agent Exploration

Authors: ['Tonghan Wang*', 'Jianhao Wang*', 'Yi Wu', 'Chongjie Zhang']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['Multi-agent reinforcement learning', 'Exploration']

Intrinsically motivated reinforcement learning aims to address the exploration challenge for sparse-reward tasks. However, the study of exploration methods in transition-dependent multi-agent settings is largely absent from the literature. We aim to take a step towards solving this problem. We present two exploration methods: exploration via information-theoretic influence (EITI) and exploration via decision-theoretic influence (EDTI), by exploiting the role of interaction in coordinated behaviors of agents. EITI uses mutual information to capture the interdependence between the transition dynamics of agents. EDTI uses a novel intrinsic reward, called Value of Interaction (VoI), to characterize and quantify the influence of one agent's behavior on expected returns of other agents. By optimizing EITI or EDTI objective as a regularizer, agents are encouraged to coordinate their exploration and learn policies to optimize the team performance. We show how to optimize these regularizers so that they can be easily integrated with policy gradient reinforcement learning. The resulting update rule draws a connection between coordinated exploration and intrinsic reward distribution. Finally, we empirically demonstrate the significant strength of our methods in a variety of multi-agent scenarios.

## Doubly Robust Bias Reduction in Infinite Horizon Off-Policy Estimation

Authors: ['Ziyang Tang*', 'Yihao Feng*', 'Lihong Li', 'Dengyong Zhou', 'Qiang Liu']

Ratings: ['6: Weak Accept', '8: Accept', '8: Accept']

Keywords: ['off-policy evaluation', 'infinite horizon', 'doubly robust', 'reinforcement learning']

Infinite horizon off-policy policy evaluation is a highly challenging task due to the excessively large variance of typical importance sampling (IS) estimators. Recently, Liu et al. (2018) proposed an approach that significantly reduces the variance of infinite-horizon off-policy evaluation by estimating the stationary density ratio, but at the cost of introducing potentially high risks due to the error in density ratio estimation. In this paper, we develop a bias-reduced augmentation of their method, which can take advantage of a learned value function to obtain higher accuracy. Our method is doubly robust in that the bias vanishes when either the density ratio or value function estimation is perfect.  In general, when either of them is accurate, the bias can also be reduced. Both theoretical and empirical results show that our method yields significant advantages over previous methods.

## Learning to Plan in High Dimensions via Neural Exploration-Exploitation Trees

Authors: ['Binghong Chen', 'Bo Dai', 'Qinjie Lin', 'Guo Ye', 'Han Liu', 'Le Song']

Ratings: ['6: Weak Accept', '8: Accept', '8: Accept']

Keywords: ['learning to plan', 'representation learning', 'learning to design algorithm', 'reinforcement learning', 'meta learning']

We propose a meta path planning algorithm named \emph{Neural Exploration-Exploitation Trees~(NEXT)} for learning from prior experience for solving new path planning problems in high dimensional continuous state and action spaces. Compared to more classical sampling-based methods like RRT, our approach achieves much better sample efficiency in  high-dimensions and can benefit from prior experience of planning in similar environments. More specifically, NEXT exploits a novel neural architecture which can learn promising search directions from problem structures. The learned prior is then integrated into a UCB-type algorithm to achieve an online balance between \emph{exploration} and \emph{exploitation} when solving a new problem. We conduct thorough experiments to show that NEXT accomplishes new planning problems with more compact search trees and significantly outperforms state-of-the-art methods on several benchmarks.

## Is a Good Representation Sufficient for Sample Efficient Reinforcement Learning?

Authors: ['Simon S. Du', 'Sham M. Kakade', 'Ruosong Wang', 'Lin F. Yang']

Ratings: ['6: Weak Accept', '8: Accept', '8: Accept']

Keywords: ['reinforcement learning', 'function approximation', 'lower bound', 'representation']

Modern deep learning methods provide effective means to learn good representations. However, is a good representation itself sufficient for sample efficient reinforcement learning? This question has largely been studied only with respect to (worst-case) approximation error, in the more classical approximate dynamic programming literature. With regards to the statistical viewpoint, this question is largely unexplored, and the extant body of literature mainly focuses on conditions which \emph{permit} sample efficient reinforcement learning with little understanding of what are \emph{necessary} conditions for efficient reinforcement learning.
This work shows that, from the statistical viewpoint, the situation is far subtler than suggested by the more traditional approximation viewpoint, where the requirements on the representation that suffice for sample efficient RL are even more stringent. Our main results provide sharp thresholds for reinforcement learning methods, showing that there are hard limitations on what constitutes good function approximation (in terms of the dimensionality of the representation), where we focus on natural representational conditions relevant to value-based, model-based, and policy-based learning. These lower bounds highlight that having a good (value-based, model-based, or policy-based) representation in and of itself is insufficient for efficient reinforcement learning, unless the quality of this approximation passes certain hard thresholds. Furthermore, our lower bounds also imply exponential separations on the sample complexity between 1) value-based learning with perfect representation and value-based learning with a good-but-not-perfect representation, 2) value-based learning and policy-based learning, 3) policy-based learning and supervised learning and 4) reinforcement learning and imitation learning.   

## Dream to Control: Learning Behaviors by Latent Imagination

Authors: ['Danijar Hafner', 'Timothy Lillicrap', 'Jimmy Ba', 'Mohammad Norouzi']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept', '8: Accept']

Keywords: ['world model', 'latent dynamics', 'imagination', 'planning by backprop', 'policy optimization', 'planning', 'reinforcement learning', 'control', 'representations', 'latent variable model', 'visual control', 'value function']

Learned world models summarize an agent's experience to facilitate learning complex behaviors. While learning world models from high-dimensional sensory inputs is becoming feasible through deep learning, there are many potential ways for deriving behaviors from them. We present Dreamer, a reinforcement learning agent that solves long-horizon tasks from images purely by latent imagination. We efficiently learn behaviors by propagating analytic gradients of learned state values back through trajectories imagined in the compact state space of a learned world model. On 20 challenging visual control tasks, Dreamer exceeds existing approaches in data-efficiency, computation time, and final performance.

## Behaviour Suite for Reinforcement Learning

Authors: ['Ian Osband', 'Yotam Doron', 'Matteo Hessel', 'John Aslanides', 'Eren Sezener', 'Andre Saraiva', 'Katrina McKinney', 'Tor Lattimore', 'Csaba Szepesvari', 'Satinder Singh', 'Benjamin Van Roy', 'Richard Sutton', 'David Silver', 'Hado Van Hasselt']

Ratings: ['3: Weak Reject', '6: Weak Accept', '8: Accept']

Keywords: ['reinforcement learning', 'benchmark', 'core issues', 'scalability', 'reproducibility']

This paper introduces the Behaviour Suite for Reinforcement Learning, or bsuite for short. bsuite is a collection of carefully-designed experiments that investigate core capabilities of reinforcement learning (RL) agents with two objectives. First, to collect clear, informative and scalable problems that capture key issues in the design of general and efficient learning algorithms. Second, to study agent behaviour through their performance on these shared benchmarks. To complement this effort, we open source this http URL, which automates evaluation and analysis of any agent on bsuite. This library facilitates reproducible and accessible research on the core issues in RL, and ultimately the design of superior learning algorithms. Our code is Python, and easy to use within existing projects. We include examples with OpenAI Baselines, Dopamine as well as new reference implementations. Going forward, we hope to incorporate more excellent experiments from the research community, and commit to a periodic review of bsuite from a committee of prominent researchers.

## Model Based Reinforcement Learning for Atari

Authors: ['Łukasz Kaiser', 'Mohammad Babaeizadeh', 'Piotr Miłos', 'Błażej Osiński', 'Roy H Campbell', 'Konrad Czechowski', 'Dumitru Erhan', 'Chelsea Finn', 'Piotr Kozakowski', 'Sergey Levine', 'Afroz Mohiuddin', 'Ryan Sepassi', 'George Tucker', 'Henryk Michalewski']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['reinforcement learning', 'model based rl', 'video prediction model', 'atari']

Model-free reinforcement learning (RL) can be used to learn effective policies for complex tasks, such as Atari games, even from image observations. However, this typically requires very large amounts of interaction -- substantially more, in fact, than a human would need to learn the same games. How can people learn so quickly? Part of the answer may be that people can learn how the game works and predict which actions will lead to desirable outcomes. In this paper, we explore how video prediction models can similarly enable agents to solve Atari games with fewer interactions than model-free methods. We describe Simulated Policy Learning (SimPLe), a complete model-based deep RL algorithm based on video prediction models and present a comparison of several model architectures, including a novel architecture that yields the best results in our setting. Our experiments evaluate SimPLe on a range of Atari games in low data regime of 100k interactions between the agent and the environment, which corresponds to two hours of real-time play. In most games SimPLe outperforms state-of-the-art model-free algorithms, in some games by over an order of magnitude.

## Disagreement-Regularized Imitation Learning

Authors: ['Kiante Brantley', 'Wen Sun', 'Mikael Henaff']

Ratings: ['6: Weak Accept', '8: Accept', '8: Accept']

Keywords: ['imitation learning', 'reinforcement learning', 'uncertainty']

We present a simple and effective algorithm designed to address the covariate shift problem in imitation learning. It operates by training an ensemble of policies on the expert demonstration data, and using the variance of their predictions as a cost which is minimized with RL together with a supervised behavioral cloning cost. Unlike adversarial imitation methods, it uses a fixed reward function which is easy to optimize. We prove a regret bound for the algorithm which is linear in the time horizon multiplied by a coefficient which we show to be low for certain problems in which behavioral cloning fails. We evaluate our algorithm empirically across multiple pixel-based Atari environments and continuous control tasks, and show that it matches or significantly outperforms behavioral cloning and generative adversarial imitation learning.

## Measuring the Reliability of Reinforcement Learning Algorithms

Authors: ['Stephanie C.Y. Chan', 'Samuel Fishman', 'Anoop Korattikara', 'John Canny', 'Sergio Guadarrama']

Ratings: ['6: Weak Accept', '8: Accept', '8: Accept']

Keywords: ['reinforcement learning', 'metrics', 'statistics', 'reliability']

Lack of reliability is a well-known issue for reinforcement learning (RL) algorithms. This problem has gained increasing attention in recent years, and efforts to improve it have grown substantially. To aid RL researchers and production users with the evaluation and improvement of reliability, we propose a set of metrics that quantitatively measure different aspects of reliability. In this work, we focus on variability and risk, both during training and after learning (on a fixed policy). We designed these metrics to be general-purpose, and we also designed complementary statistical tests to enable rigorous comparisons on these metrics. In this paper, we first describe the desired properties of the metrics and their design, the aspects of reliability that they measure, and their applicability to different scenarios. We then describe the statistical tests and make additional practical recommendations for reporting results. The metrics and accompanying statistical tools have been made available as an open-source library. We apply our metrics to a set of common RL algorithms and environments, compare them, and analyze the results.

## Maximum Likelihood Constraint Inference for Inverse Reinforcement Learning

Authors: ['Dexter R.R. Scobee', 'S. Shankar Sastry']

Ratings: ['3: Weak Reject', '6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['learning from demonstration', 'inverse reinforcement learning', 'constraint inference']

While most approaches to the problem of Inverse Reinforcement Learning (IRL) focus on estimating a reward function that best explains an expert agent’s policy or demonstrated behavior on a control task, it is often the case that such behavior is more succinctly represented by a simple reward combined with a set of hard constraints. In this setting, the agent is attempting to maximize cumulative rewards subject to these given constraints on their behavior. We reformulate the problem of IRL on Markov Decision Processes (MDPs) such that, given a nominal model of the environment and a nominal reward function, we seek to estimate state, action, and feature constraints in the environment that motivate an agent’s behavior. Our approach is based on the Maximum Entropy IRL framework, which allows us to reason about the likelihood of an expert agent’s demonstrations given our knowledge of an MDP. Using our method, we can infer which constraints can be added to the MDP to most increase the likelihood of observing these demonstrations. We present an algorithm which iteratively infers the Maximum Likelihood Constraint to best explain observed behavior, and we evaluate its efficacy using both simulated behavior and recorded data of humans navigating around an obstacle.

## Improving Generalization in Meta Reinforcement Learning using Learned Objectives

Authors: ['Louis Kirsch', 'Sjoerd van Steenkiste', 'Juergen Schmidhuber']

Ratings: ['6: Weak Accept', '6: Weak Accept', '8: Accept']

Keywords: ['meta reinforcement learning', 'meta learning', 'reinforcement learning']

Biological evolution has distilled the experiences of many learners into the general learning algorithms of humans. Our novel meta reinforcement learning algorithm MetaGenRL is inspired by this process. MetaGenRL distills the experiences of many complex agents to meta-learn a low-complexity neural objective function that decides how future individuals will learn. Unlike recent meta-RL algorithms, MetaGenRL can generalize to new environments that are entirely different from those used for meta-training. In some cases, it even outperforms human-engineered RL algorithms. MetaGenRL uses off-policy second-order gradients during meta-training that greatly increase its sample efficiency.

