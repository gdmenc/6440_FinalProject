**Project Title:** Commanding Humanoid Articulation: Inverse Kinematics with Natural Language Interface

**Team:** Ryan Quinn and Guillermo Mendoza

### Project Overview
Our group will design an Inverse Kinematics (IK) solver to realistically animate humanoid character limbs based on textual user input. The core objective is to compute the complex joint angles required to place the character’s end effector (e.g., a hand) at a target position while maintaining human-like constraints.

Unlike standard Forward Kinematics, where joints are rotated explicitly, our system will allow a user to define a goal state (a target point in 3D space), and the system will solve for the necessary configuration of the arm and torso joints. To achieve stable and realistic motion, we will utilize **Jacobian-based methods**, specifically implementing **Damped Least Squares (DLS)**. This approach optimizes response time while preventing mathematical instabilities (singularities) when limbs are fully extended [1].

### Technical Approach & Scope
To ensure the motion is not just mathematically correct but also "human-like," we will implement the following graphics concepts:

* **Constrained Motion:** We will implement joint limits (clamping) to prevent unrealistic poses (e.g., elbows bending backward).
* **Null Space Optimization:** We intend to utilize the "Null Space" of the Jacobian to handle redundancy. This allows the character to optimize for a secondary objective—such as keeping the elbow down in a natural resting state—while still reaching the target.
* **Kinematic Chain:** The character will be rooted at the waist/spine. The primary simulation will focus on the upper-body kinematic chain (spine, shoulders, elbows, wrists) to allow for high-fidelity interaction without the complexity of full-body balancing physics.

### Interaction & Interface
The system will be driven by a **Controlled Natural Language (CNL)** interface. Rather than parsing arbitrary sentence structures, we will design a robust interpreter that maps specific keywords to vector operations (e.g., parsing *"Raise Right Hand"* into a target coordinate relative to the character’s current state).

The user will interact via a terminal or text overlay. The system will support:
1.  **Action Commands:** (e.g., "Move left hand forward 10 units").
2.  **State Resets:** Returning the character to a neutral, standing T-pose or I-pose.
3.  **Stretch Goal:** If time permits, we will introduce simple interactive objects (e.g., a sphere) that the character can be commanded to "touch" or "look at."

### Final Project Demonstration
The final demonstration will take place in a minimalist 3D environment to prioritize the analysis of motion quality. The demo flow will be:
1.  **Input:** We will input a command such as "Reach for top shelf."
2.  **Visual Debugging:** The system will render a "ghost" target (wireframe indicator) at the calculated destination to visually confirm the parser's interpretation.
3.  **Interpolation:** The character will smoothly interpolate from its current pose to the solved IK configuration, demonstrating the effectiveness of the Damped Least Squares solver in avoiding jitter.

A successful project will demonstrate a robust mathematical solver that handles singularities without crashing and a parsing system that accurately translates English commands into 3D vector targets.

### References
[1] Aristidou, A., et al. “Inverse kinematics techniques in computer graphics: A survey.” *Computer Graphics Forum*, vol. 37, no. 6, 29 Nov. 2017, pp. 35–58, [https://doi.org/10.1111/cgf.13310](https://doi.org/10.1111/cgf.13310).