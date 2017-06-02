Recent changes:
 - Addition of learning_botnet.py, common interface for all botnets. The inheritance tree is as follows:
   LearningBotnet --> Botnet --> QStar
                   |          |
                   |          |-> Fast, FastIncr, GreedyFast, RewardIncr
                   |
                   -> QLearning --> Thompson --> ModelBased --> FullModelBased
 - Update of all the botnets to match the interface, along with tests and examples.
 - A Botnet must provide three methods:
    - exploration(), to gather information
    - exploitation(), to receive the highest reward
    - receive_reward(), to acknowledge the result of an action during training (and update internal values...)
 - A state is now immutable, to be hashable properly
 - Some changes in network.py, to reflect the convention that the state is always before the actions in parameters
 - Little refactoring to make methods available where they should be


Files:
 - state.py:                implementation of integer set
 - network.py:              operations on the input
 - policy.py:               performs statistic analysis of a policy
 - botnet.py:               abstract class containing rewards-related functions and holds the state
 - markov.py:               q-star and such
 - fast.py:                 algorithm minimizing expected time of hijacking the whole network
====Not yet to standards====
 - GUI.py:                  well, I guess you guessed, right? (TODO: update, integrate in network/botnet?)
 - thompson_sampling.py:    self-explanatory (TODO: inheritate from botnet, or create botnet_learning that doesn't have access to attributes)
 - tests.py:                could declare some networks at the top, and call a Botnet.compute_policy().expected_time/reward()...


On vocabulary:
 - the reward is the immediate reward following taking action a in state s
 - the value is the R function (on all actions)
 - the power is the sum of the proselytism of the current state
 - the resistance measures how much it is hard to capture a given node
 - the proselytism is the value that will be added to the power of the network once the node is hijacked
 - the cost is how an attempt at capturing affects the immediate reward (not very relevant I think...)
 - a node is a computer, which is said to be either free or hijacked
 - an action is also a node, but which is being hijacked
 - a state is a subset of the nodes of the network (the hijacked one)
 - a policy is a sequence of actions (the order of the hijackings)
 - a botnet is the conjunction of a network, a state, and optimization parameters (gamma, ...)


Biblio :
https://en.wikipedia.org/wiki/Markov_decision_process

https://en.wikipedia.org/wiki/Q-learning
Temporal-Difference learning : http://www.bkgm.com/articles/tesauro/tdl.html
http://papers.nips.cc/paper/4251-speedy-q-learning.pdf
Deep Reinforcement learning with double Q-learning : https://arxiv.org/pdf/1509.06461.pdf

https://en.wikipedia.org/wiki/Learning_automata
