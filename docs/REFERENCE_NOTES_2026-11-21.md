# Reading plan for the lightweight DDQN collaboration

Prepared 21 November 2026. This is a targeted reference review, not a systematic
literature search or clearance to claim novelty. Bibliographic metadata for
the first three DOI records below was checked live through Crossref. The two
local PDFs were inspected; Sensors sections on modeling, learning, experiments
and limitations were read, with pages 1, 12 and 14 also visually checked.

## Chosen recent full-text inspiration

**Xu, Shu; Liu, Qingjie; Gong, Chengye; Wen, Xupeng.** (2025).
“Energy-Efficient Multi-Agent Deep Reinforcement Learning Task Offloading and
Resource Allocation for UAV Edge Computing.” *Sensors*, 25(11), 3403.
Published 28 May 2025. https://doi.org/10.3390/s25113403

Local source: `October 31/sensors-25-03403.pdf`.

| Read | Extract for our design | Distinction to preserve |
| --- | --- | --- |
| Section 3, pp. 6–10 | Mobility, communication, computational model, objective and constraints; map units and decision variables. | Their mobile devices/UAVs and resource decisions do not automatically implement our vehicle-local/UAV/RSU choices. |
| Section 4, pp. 10–14 | State, actions, reward; per-agent actor and twin critics, global information used by critics. | MATD3-TORA is an actor-critic method; our selected learner remains discrete DDQN. |
| Section 5, pp. 14–22 | Comparators, learning-rate/exploration/weight sensitivity, task-volume and bandwidth experiments. | Reproduce compatible baselines locally if needed; published scores are not controlled comparison data. |
| Section 5.5, p. 22 | Authors identify scalability, centralized-critic information/processing and communication assumptions as limitations. | This motivates a measured compact-controller question; it does not prove DDQN will solve those limitations. |

The paper explicitly asserts exponential complexity growth in its limitations.
That statement is the authors’ claim; derive operation counts for the actual
implementations rather than importing it as an established theorem. The printed
target-update signs in equations 47–48 and the 320-second/40-one-second-slot
description warrant checking before reuse. Follow verified algorithms and
dimensionally consistent equations, and state our adaptations.

**Candidate research question:** can a structured compact DDQN reduce complete
online decision resources, while retaining coordinated task allocation and
deadline performance under the same observable information? Merely changing
MATD3 to DDQN or shrinking an MLP does not establish a new method. A new
mechanism, closest-method comparison, matched ablation and fresh evidence are
still required.

## Historical modeling anchor

**Hao, Hao; Xu, Changqiao; Zhang, Wei; Yang, Shujie; Muntean, Gabriel-Miro.**
(2024). “Joint Task Offloading, Resource Allocation, and Trajectory Design for
Multi-UAV Cooperative Edge Computing With Task Priority.” *IEEE Transactions
on Mobile Computing*, 23(9), 8649–8663.
https://doi.org/10.1109/TMC.2024.3350078

Local source:
`October 31/Joint_Task_Offloading_Resource_Allocation_and_Trajectory_Design_for_Multi-UAV_Cooperative_Edge_Computing_With_Task_Priority.pdf`.

The October 31 alignment audit identifies this as the primary modeling
reference. It considers binary offloading with hybrid discrete/continuous
control and latent-space DRL. Reuse the disciplined order of model definition,
objective, constraints and control formulation; declare the local adaptations
and provisional parameters. Do not describe our discretized DDQN as a
reproduction of its full resource/trajectory optimization.

## Newer domain work that still needs full-method review

**Sun, Haiyang; Chen, Honglong; Xue, Huansheng; Wu, Liantao; Li, Zhishuai;
Chen, Ning; Liu, Weifeng; Yu, Jiguo.** (2026). “Q-TOP: Queue-Aware Task
Offloading and Path Planning for Multi-UAV Assisted Vehicular Edge Computing
Networks.” *IEEE Transactions on Vehicular Technology*, 75(11), 28349–28361.
https://doi.org/10.1109/TVT.2026.3721145

Title/authors/venue/volume/issue/pages were reverified against Crossref. No full
paper was obtained/read in this handoff. Earlier project notes record an indexed
abstract, but they cannot establish all algorithmic differences. Obtain the full
method and compare coordination, queues, decisions and cost before making a
priority claim. This is why Sensors 2025 is called a recent **available full-text
inspiration**, not “the latest paper in the field.”

## Foundational and closest-method reading

| Work | Why it matters | Required check |
| --- | --- | --- |
| *Deep Reinforcement Learning with Double Q-learning*, https://arxiv.org/abs/1509.06461 | Defines the online-select/target-evaluate DDQN distinction. | Verify that the implemented target and terminal handling match the declared objective. |
| *Action Branching Architectures for Deep Reinforcement Learning*, https://arxiv.org/abs/1711.08946 | Shared representation, separate action branches and linear output growth already exist. | Distinguish our sharing, coordination and update equations from BDQ. |
| *Computation Offloading for Multi-Server Multi-Access Edge Vehicular Networks: A DDQN-Based Method*, https://arxiv.org/abs/2404.07215 | Vehicular congestion and server-side scheduling are already studied with DDQN. | Identify the difference between their request scheduling and our joint allocation. |
| *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*, https://arxiv.org/abs/2006.14171 | Feasibility masking is established. | It is not a DDQN convergence/deadline guarantee; include equally masked controls. |
| *Sustainable Task Offloading in Secure UAV-Assisted Smart Farm Networks: A Multi-Agent DRL With Action Mask Approach*, https://arxiv.org/abs/2407.19657 | UAV offloading with DDQN and masks already exists. | Avoid “first masked UAV DDQN”; its security scope does not enter contribution one. |
| *Hierarchical Task Offloading for UAV-Assisted Vehicular Edge Computing via Deep Reinforcement Learning*, https://arxiv.org/abs/2507.05722 | Learned control plus resource-aware scheduling already exists in this domain. | Do not label a generic learned/heuristic combination a new mechanism. |

These foundational/closest links are reading tasks supported by existing project
notes; they were not all independently reread in full in this handoff. Distinguish
preprints from published versions, and record version/date in the collaborator's
literature matrix.

Before implementation, make one row per closest method with exact state/actions,
decision location, model dimensions, sharing, decoder, DDQN update, deployment
boundary, measured cost, policy evidence, limitations, exact proposed difference,
and the experiment that could falsify that difference. “Not mentioned in the
abstract” must remain unresolved until the full method is checked.

## Writing and contribution statement

Follow the papers' research logic; use our own notation and prose. Every claimed
decision variable must exist in the implementation or be identified as a fixed
rule. Separate symbolic definitions from experimental numbers. Number/cite
equations consistently, define abbreviations once, and link each result table
to traceable raw data. Do not claim energy-efficient computation from reduced
neural parameter storage: UAV physical energy and controller compute are
different quantities, and both require measurements for their respective claims.

The sources motivate an investigation. They do not guarantee novelty, successful
training, thesis acceptance or lightweight deployment.
