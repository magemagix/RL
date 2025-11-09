
<p align="center">
  <img src="./FrozenLake-MC-first-visit.gif" alt="Taxi Agent in Action" width="500"/><br>
  <em>The FrozenLake agent efficiently navigating the 4×4 grid!</em>
</p>


<br><br>

# 🧊 Frozen Lake with Monte Carlo First-Visit Control 🎯

Welcome to the **Frozen Lake** environment! Here, our goal is to teach an agent to navigate from the start to the goal safely, avoiding holes, using **Monte Carlo First-Visit Control**. 🌟

This algorithm is **model-free**, meaning it **does not require knowing the environment’s transition probabilities**. The agent learns optimal behavior **entirely from experience** by exploring and observing the rewards it collects.

---

## ❄️ About the FrozenLake-v1 Environment

**FrozenLake-v1** from **Gymnasium** is a **grid world** representing a frozen lake:

* Safe frozen tiles (`F`)
* Holes (`H`) that terminate the episode
* Start (`S`) and goal (`G`) tiles

The standard **4×4 grid**:

* `is_slippery=False` → deterministic movement
* `is_slippery=True` → stochastic/slippery movement

---

### 🏔️ States

Each **state** represents the agent’s position on the lake:

```markdown
Total states = 4 rows × 4 columns = 16 states
```

Each state is encoded as:

```markdown
state_index = row * ncols + col
```

For example, the **goal state** is:

```markdown
goal_state = 3 * 4 + 3 = 15
```

---

### 📊 Space Info

| Space Type               | Description                                |
| ------------------------ | ------------------------------------------ |
| **Observation space**    | 16 (4×4 grid positions)                    |
| **State representation** | Single integer 0–15                        |
| **Action space**         | 4 possible actions (left, down, right, up) |

---

### 🎮 Actions

| Action | Description   |
| ------ | ------------- |
| **0**  | Move Left ⬅️  |
| **1**  | Move Down ⬇️  |
| **2**  | Move Right ➡️ |
| **3**  | Move Up ⬆️    |

---

### 💡 Rewards

| Event               | Reward |
| ------------------- | ------ |
| Reach Goal          | **+1** |
| Fall into Hole      | 0      |
| Step on Frozen Tile | 0      |

A successful episode means reaching the goal **without falling into holes**. 🏅

---

## 🧠 Monte Carlo First-Visit Control

**Monte Carlo First-Visit Control** is a **model-free reinforcement learning algorithm**. The agent **learns from experience** by exploring the environment, collecting episodes, and updating the action-value function **only for the first time a state-action pair is visited in that episode**.

---

### 🔄 Exploration & Epsilon Decay

At the beginning, the agent knows nothing about the environment. To encourage learning:

* We use an **ε-greedy policy**, where the agent sometimes explores random actions.
* After **half of the episodes**, ε is gradually **reduced** to favor exploitation of the learned Q-values.

---

### 🧾 Algorithm: Monte Carlo First-Visit Control

```markdown
**Input:**
  - Environment with state set S and action set A
  - Discount factor γ ∈ [0, 1)
  - Number of episodes N
  - Exploration rate ε for ε-greedy policy

**Output:**
  - Optimal action-value function Q*(s, a)
  - Optimal policy π*(s)

**Steps:**
1. Initialize Q(s, a) and returns[(s,a)] arbitrarily (e.g., zeros)
2. For each episode 1..N:
   a. Generate an episode using ε-greedy policy
   b. Compute return Gt for each step in **reverse order**
   c. Update Q(s, a) **only for first-visit** in episode
   d. Optionally decay ε after half of episodes
3. Derive optimal policy π*(s) = argmax_a Q(s, a)
```
<br><br>
Here is another version of this algorithm, with explanations on how the **first-visit updates** are performed.



1. **Initialize:**

   * $Q(s, a)$ arbitrarily for all $s \in S, a \in A(s)$ (commonly zeros).
   * $N(s, a) \gets 0$ (a counter for the number of first visits to $(s, a)$).
   * $\text{Returns}(s, a) \gets []$ (to store all returns for each first visit).

2. **Loop forever (or for a fixed number of episodes):**

   1. Generate an **episode** following the current **ε-greedy policy** based on $Q$:

      $$
      S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T
      $$

      > **Note:** At the beginning, exploration is high (ε large). After half the episodes, decay ε to favor exploitation.

   2. Initialize return $G \gets 0$.

   3. **Traverse the episode backward** (from last step to first step):

      ```text
      for t = T-1, T-2, ..., 0:
      ```

      * Update the return:

        $$
        G \gets \gamma G + R_{t+1}
        $$

      * **First-visit check (forward order):**
        If the pair $(S_t, A_t)$ has **not been visited earlier in this episode**:

        $$
        (S_t, A_t) \notin { (S_0, A_0), \dots, (S_{t-1}, A_{t-1}) }
        $$

        do the following:

        1. Increment the visit counter:

           $$
           N(S_t, A_t) \gets N(S_t, A_t) + 1
           $$

        2. Append return $G$ to the list:

           $$
           \text{Returns}(S_t, A_t).append(G)
           $$

        3. Update the action-value estimate (average of returns):

           $$
           Q(S_t, A_t) \gets \text{mean}(\text{Returns}(S_t, A_t))
           $$





---

### 🧭 Intuition

1. Start with **no knowledge** of the environment.
2. Explore with **ε-greedy policy** → collect episodes and rewards.
3. Update Q-values **only for first visits** to avoid bias from repeated visits.
4. Gradually reduce exploration (decay ε) to exploit learned values.
5. Derive **optimal policy** from Q(s, a) → agent learns to safely reach the goal.

---

## 🛠️ Let’s Set Up FrozenLake-v1! ❄️

```bash
# Install uv package
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

### 📦 Initialize the Example

```bash
# Pin Python version
uv python pin 3.12

# Initialize new project
cd 004 && uv init
rm main.py

# Create virtual environment
uv venv --python 3.12
```

---

### 📥 Install Project Dependencies

```bash
# Inside examples/004/
uv add -r requirements.txt
```

---

### 🚀 Launch Jupyter Notebook

```bash
uv run jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password=''
```

Your Frozen Lake agent is now ready to **learn optimal navigation using Monte Carlo First-Visit Control**! 🧊✨

---


