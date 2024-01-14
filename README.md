# Flutter: An LLM-powered simulated drone

Our project consists of three main components:
1. **[AirSim](https://microsoft.github.io/AirSim/), a simulator for drones, cars and other vehicles, built on Unreal Engine.**
    - We use a custom AirSim environment with a simulated drone that can be controlled using natural language.

2. **LLM integration: GPT-4**
    - Using an (outer) LLM easily interprets natural language instructions, which are more intuitive than low-level action controllers. This introduces capabilities for multi-step reasoning and task planning.
    - The inner LLM serves as a reasoning module to interpret JSON outputs from the object detection model (see below) for better scene understanding and longer horizon planning.

3. **Object Detection Model: Cloud Vision API**
    - Using a dedicated object detection model improves accuracy on understanding dynamic scenes and real-time mapping. This facilitates completing complex tasks by grounding the (outer) language model.