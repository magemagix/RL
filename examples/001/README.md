# 🌙 LunaLander with PPO 🚀

Welcome to the **LunaLander** example using **Proximal Policy Optimization (PPO)**! 🚀✨

This code is inspired by **UNIT 1** of the **Deep RL Course** from **Hugging Face**. 🎓💡 
If you’re looking to learn more about the concepts and hands-on tutorials from this course, be sure to check out the full [UNIT 1: Hands-on](https://huggingface.co/learn/deep-rl-course/unit1/hands-on) guide! 📚


<br>



## 🛠️ Let's Set Up LunaLander! 🚀

Before we dive into the **LunaLander** action, let's get everything ready! 🙌 You’ll need to have the **uv** package and project manager installed to keep things running smoothly. If you don’t have it yet, no worries! 🧰

You can grab uv from its [GitHub repository](https://github.com/astral-sh/uv), or just run this super simple one-liner to install it:

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Now, why are we using **uv**? 🤔 It’s our secret weapon to set up the perfect environment for each project. Every project here has its own tailored setup, and **uv** makes sure you get exactly what you need for the **LunaLander** to run without a hitch. 🎯


### 📦 Initialize the Example:

```bash
# you must be in path examples/001/
uv python pin 3.12
uv init 
rm main.py
uv venv --python 3.12 # We want be sure that version is 3.12
```

We’re almost there! But before we get our virtual rocket flying, there are a couple of dependencies we need to take care of... 🚀

