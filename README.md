# flutter
Flutter: An LLM-powered simulated drone

Our project consists of three main components:
1. [AirSim](https://microsoft.github.io/AirSim/), a simulator for drones, cars and other vehicles, built on Unreal Engine.
- We develop a custom AirSim environment with a simulated drone that can be controlled using natural language.

2. LLM integration: GPT-4
- Using an LLM easily interprets natural language instructions, which are more intuitive than low-level action controllers, and introduces capabilities for multi-step reasoning and task planning.

3. Object Detection Model: Cloud Vision API
- Using a dedicated object detection model improves accuracy on understanding dynamic scenes and real-time mapping. This facilitates completing complex tasks by grounding the language model.