# Oral

## Learning to Reach Goals via Iterated Supervised Learning

Authors: ['Dibya Ghosh', 'Abhishek Gupta', 'Ashwin Reddy', 'Justin Fu', 'Coline Manon Devin', 'Benjamin Eysenbach', 'Sergey Levine']

Ratings: ['7: Good paper, accept', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['goal reaching', 'reinforcement learning', 'behavior cloning', 'goal-conditioned RL']

Current reinforcement learning (RL) algorithms can be brittle and difficult to use, especially when learning goal-reaching behaviors from sparse rewards. Although supervised imitation learning provides a simple and stable alternative, it requires access to demonstrations from a human supervisor. In this paper, we study RL algorithms that use imitation learning to acquire goal reaching policies from scratch, without the need for expert demonstrations or a value function. In lieu of demonstrations, we leverage the property that any trajectory is a successful demonstration for reaching the final state in that same trajectory. We propose a simple algorithm in which an agent continually relabels and imitates the trajectories it generates to progressively learn goal-reaching behaviors from scratch. Each iteration, the agent collects new trajectories using the latest policy, and maximizes the likelihood of the actions along these trajectories under the goal that was actually reached, so as to improve the policy. We formally show that this iterated supervised learning procedure optimizes a bound on the RL objective, derive performance bounds of the learned policy, and empirically demonstrate improved goal-reaching performance and robustness over current RL algorithms in several benchmark tasks. 

## Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients

Authors: ['Brenden K Petersen', 'Mikel Landajuela Larma', 'Terrell N. Mundhenk', 'Claudio Prata Santiago', 'Soo Kyung Kim', 'Joanne Taery Kim']

Ratings: ['7: Good paper, accept', '8: Top 50% of accepted papers, clear accept', '8: Top 50% of accepted papers, clear accept', '9: Top 15% of accepted papers, strong accept']

Keywords: ['symbolic regression', 'reinforcement learning', 'automated machine learning']

Discovering the underlying mathematical expressions describing a dataset is a core challenge for artificial intelligence. This is the problem of $\textit{symbolic regression}$. Despite recent advances in training neural networks to solve complex tasks, deep learning approaches to symbolic regression are underexplored. We propose a framework that leverages deep learning for symbolic regression via a simple idea: use a large model to search the space of small models. Specifically, we use a recurrent neural network to emit a distribution over tractable mathematical expressions and employ a novel risk-seeking policy gradient to train the network to generate better-fitting expressions. Our algorithm outperforms several baseline methods (including Eureqa, the gold standard for symbolic regression) in its ability to exactly recover symbolic expressions on a series of benchmark problems, both with and without added noise. More broadly, our contributions include a framework that can be applied to optimize hierarchical, variable-length objects under a black-box performance metric, with the ability to incorporate constraints in situ, and a risk-seeking policy gradient formulation that optimizes for best-case performance instead of expected performance.

## Learning Generalizable Visual Representations via Interactive Gameplay

Authors: ['Luca Weihs', 'Aniruddha Kembhavi', 'Kiana Ehsani', 'Sarah M Pratt', 'Winson Han', 'Alvaro Herrasti', 'Eric Kolve', 'Dustin Schwenk', 'Roozbeh Mottaghi', 'Ali Farhadi']

Ratings: ['8: Top 50% of accepted papers, clear accept', '8: Top 50% of accepted papers, clear accept', '8: Top 50% of accepted papers, clear accept', '9: Top 15% of accepted papers, strong accept']

Keywords: ['representation learning', 'deep reinforcement learning', 'computer vision']

A growing body of research suggests that embodied gameplay, prevalent not just in human cultures but across a variety of animal species including turtles and ravens, is critical in developing the neural flexibility for creative problem solving, decision making, and socialization. Comparatively little is known regarding the impact of embodied gameplay upon artificial agents. While recent work has produced agents proficient in abstract games, these environments are far removed the real world and thus these agents can provide little insight into the advantages of embodied play. Hiding games, such as hide-and-seek, played universally, provide a rich ground for studying the impact of embodied gameplay on representation learning in the context of perspective taking, secret keeping, and false belief understanding. Here we are the first to show that embodied adversarial reinforcement learning agents playing Cache, a variant of hide-and-seek, in a high fidelity, interactive, environment, learn generalizable representations of their observations encoding information such as object permanence, free space, and containment. Moving closer to biologically motivated learning strategies, our agents' representations, enhanced by intentionality and memory, are developed through interaction and play. These results serve as a model for studying how facets of vision develop through interaction, provide an experimental framework for assessing what is learned by artificial agents, and demonstrates the value of moving from large, static, datasets towards experiential, interactive, representation learning.

## Human-Level Performance in No-Press Diplomacy via Equilibrium Search

Authors: ['Jonathan Gray', 'Adam Lerer', 'Anton Bakhtin', 'Noam Brown']

Ratings: ['7: Good paper, accept', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept', '8: Top 50% of accepted papers, clear accept']

Keywords: ['multi-agent systems', 'regret minimization', 'no-regret learning', 'game theory', 'reinforcement learning']

Prior AI breakthroughs in complex games have focused on either the purely adversarial or purely cooperative settings. In contrast, Diplomacy is a game of shifting alliances that involves both cooperation and competition. For this reason, Diplomacy has proven to be a formidable research challenge. In this paper we describe an agent for the no-press variant of Diplomacy that combines supervised learning on human data with one-step lookahead search via regret minimization. Regret minimization techniques have been behind previous AI successes in adversarial games, most notably poker, but have not previously been shown to be successful in large-scale games involving cooperation. We show that our agent greatly exceeds the performance of past no-press Diplomacy bots, is unexploitable by expert humans, and ranks in the top 2% of human players when playing anonymous games on a popular Diplomacy website.

## Parrot: Data-Driven Behavioral Priors for Reinforcement Learning

Authors: ['Avi Singh', 'Huihan Liu', 'Gaoyue Zhou', 'Albert Yu', 'Nicholas Rhinehart', 'Sergey Levine']

Ratings: ['6: Marginally above acceptance threshold', '7: Good paper, accept', '8: Top 50% of accepted papers, clear accept', '9: Top 15% of accepted papers, strong accept']

Keywords: ['reinforcement learning', 'imitation learning']

Reinforcement learning provides a general framework for flexible decision making and control, but requires extensive data collection for each new task that an agent needs to learn. In other machine learning fields, such as natural language processing or computer vision, pre-training on large, previously collected datasets to bootstrap learning for new tasks has emerged as a powerful paradigm to reduce data requirements when learning a new task. In this paper, we ask the following question: how can we enable similarly useful pre-training for RL agents? We propose a method for pre-training behavioral priors that can capture complex input-output relationships observed in successful trials from a wide range of previously seen tasks, and we show how this learned prior can be used for rapidly learning new tasks without impeding the RL agent's ability to try out novel behaviors. We demonstrate the effectiveness of our approach in challenging robotic manipulation domains involving image observations and sparse reward functions, where our method outperforms prior works by a substantial margin. Additional materials can be found on our project website: https://sites.google.com/view/parrot-rl

## Evolving Reinforcement Learning Algorithms

Authors: ['John D Co-Reyes', 'Yingjie Miao', 'Daiyi Peng', 'Esteban Real', 'Quoc V Le', 'Sergey Levine', 'Honglak Lee', 'Aleksandra Faust']

Ratings: ['6: Marginally above acceptance threshold', '7: Good paper, accept', '9: Top 15% of accepted papers, strong accept']

Keywords: ['reinforcement learning', 'evolutionary algorithms', 'meta-learning', 'genetic programming']

We propose a method for meta-learning reinforcement learning algorithms by searching over the space of computational graphs which compute the loss function for a value-based model-free RL agent to optimize. The learned algorithms are domain-agnostic and can generalize to new environments not seen during training. Our method can both learn from scratch and bootstrap off known existing algorithms, like DQN, enabling interpretable modifications which improve performance. Learning from scratch on simple classical control and gridworld tasks, our method rediscovers the temporal-difference (TD) algorithm. Bootstrapped from DQN, we highlight two learned algorithms which obtain good generalization performance over other classical control tasks, gridworld type tasks, and Atari games. The analysis of the learned algorithm behavior shows resemblance to recently proposed RL algorithms that address overestimation in value-based methods.

