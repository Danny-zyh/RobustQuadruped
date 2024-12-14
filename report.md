### Big Picture and Introduction

Achieving robust locomotion abilities—such as walking, running, and jumping—that resemble those of animals has been a longstanding challenge in robotics. Traditional approaches depend heavily on manually modeling physical dynamics and designing control strategies. While these methods have shown significant success in enabling robots to navigate and move across various terrains, they demand extensive expertise in controller design and involve labor-intensive tuning to enhance adaptability across different environments and robot structures.

Recently, learning-based methods, particularly reinforcement learning (RL), have garnered substantial attention. A common approach involves training an RL policy in a simulator and then transferring this learned policy to real-world robots using sim-to-real (Sim2Real) techniques. However, this transfer process is challenging due to fundamental differences between simulated and real-world environments. The reality gap can stem from multiple factors, such as (a) discrepancies in the robot's physical properties between the simulation and real world, (b) substantial differences in real-world terrains compared to those modeled in simulators, and (c) limitations in physics simulators, such as their inability to accurately model contact forces, deformable surfaces, and moving rigid bodies [Kumar et al., 2022].

To address these challenges, techniques like domain randomization and domain adaptation are commonly employed to mitigate the reality gap. This project specifically focuses on domain randomization for developing robust locomotion policies within simulators. Domain randomization has shown success in facilitating Sim2Real transfer in robotic manipulation [Tobin et al., 2017], and here, we aim to investigate its application to locomotion by exploring the following key questions:

- How can we design a randomization scheme in simulation that accurately captures the discrepancies introduced by the reality gap?
- How can we develop an RL algorithm and training curriculum that effectively learns robust policies under a given randomization scheme?
- How can we build a performance evaluation benchmark in the simulator that correlates reliably with the agent's performance in real-world scenarios?

Due to time limit and inaccessibility to a real robot, we hope to answer the first two question and leave the third question for future works.

### Project Scope

Our project specifically focuses on locomotion for a quadruped robot, the Unitree Go2, which comes equipped with a built-in gyroscope, level sensor, and front-facing camera. In this context, locomotion is defined as the robot's ability to reliably follow specified velocity commands when navigating challenging terrains such as sand, hiking trails, tall grass, and stairs. 

To build such a locomotive agent in reality, we plan to design a domain randomization scheme to train a robust locomotive agent simulator and finally evaluate the result in the reality (if we could hands on a real robot), to see how far reinforcement learning trained entirely in simulator can bring us.

The performance of locomotive agent can be measured by 
- **success rate**: the percentage of trails in which the robot complete the navigation task successfully
- **time-to-fall**: maximum walking time under disturbance
- **average forward reward**: average reward the agent accumulated when completing the locomotion task  

### Broader Impact

We hope this project could provide insights on mitigating the reality gap with domain randomization and help us understand how we could leverage simulators to build reliable robots rapidly and cost-effectively. Besides, we want to shed light on distributionally robust training procedures by setting up and comparing different curriculum and RL algorithms to evaluate their performance on previously unseen simulated environments.

### Related Works and References

There is extensive prior work on reinforcement learning and Sim2Real transfer for quadruped locomotion.

In early work, [Peng et al., 2018] developed a locomotion policy through imitation learning, using motion data recorded from a real dog and subsequently fine-tuning the policy in simulation. To facilitate real-world deployment, they designed a domain adaptation pipeline incorporating a stochastic encoder to map observations from both real and simulated environments to a shared latent space, which was then fed into the policy. Later, [Kumar et al., 2022] extended this approach with an adaptation module that estimates extrinsic environmental parameters (e.g., terrain height, friction, and motor strength). Their policy could condition on these estimates to generate context-appropriate actions. While both approaches successfully tackled locomotion tasks, they required significant real-world adaptation and fine-tuning to be fully effective.

Our project is largely inspired by the work of [Rudin et al., 2022], which trained locomotion policies using massively parallel deep reinforcement learning in simulation. This approach demonstrated the viability of Sim2Real transfer by deploying policies trained solely in simulation onto real robots. However, Rudin et al. did not provide detailed rationale for their choices of simulation parameter randomization or the design of their training curriculum. Additionally, they only used the PPO algorithm [Schulman et al., 2017] without comparing its performance to other state-of-the-art RL algorithms. Their experiments were also limited to lab settings, without testing on complex real-world terrains. This project aims to build on Rudin et al.'s work by addressing these gaps and extending the investigation into diverse environments and training configurations.

### Capability, Deliverables, and Tasks

Our system aims to improve on the result of Rudin et al., 2022. The core deliverable consists of
- A methodology of randomizing simulation parameters and justification, as well as the code base.
- A RL pipeline that can train the quadruped in the simulator with reasonable performance under the aforementioned randomized environment.
- A navigation test bench that effectively evaluates the performance of locomotion agents trained with different curriculum and RL algorithms.
- A report of the performance of different agents as well as discussions of their locomotion capability on unseen terrain distributions.

The breakdown of actionable items are

| Week | Description                                                                             | Deliverable                                                                                  |
|------|-----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| 5    | Investigate the parallel deep RL training code base                                     | A computer with dependency installed that can reproduce the paper's code                     |
| 6    | Investigate randomization parameters and training pipeline design                       | Documents about tunable parameters in simulator and how they could affect agent performance  |
| 7    | Constructing randomization and training pipeline, generate preliminary training results | Code base of training and playing with the learned policy, report of performance             |
| 8    | Identify evaluation tasks and build codebase, report performance                        | Code base for the evaluation task                                                            |
| 9    | Make teasure trailer video, test the policy on real Unitree Go2 Robot                   | Teasure trailer video and results of real robots                                             |
| 10   | Consolidate project documents and images, draft final report                            | Project final report                                                                         |
| 11   | Preparing for project demo                                                              | Demo that enables people to play with the policy in simulator                                |

### Works Cited

Peng, Xue Bin, et al. "Learning agile robotic locomotion skills by imitating animals." arXiv preprint arXiv:2004.00784 (2020).

Kumar, Ashish, et al. "Adapting rapid motor adaptation for bipedal robots." 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022.

Rudin, Nikita, et al. "Learning to walk in minutes using massively parallel deep reinforcement learning." Conference on Robot Learning. PMLR, 2022.

Tobin, Josh, et al. "Domain randomization for transferring deep neural networks from simulation to the real world." 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS). IEEE, 2017.

Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
