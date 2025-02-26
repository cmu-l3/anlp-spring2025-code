import gymnasium
import miniwob
import time
import numpy as np
import os
from miniwob.action import ActionTypes, ActionSpaceConfig

def vlm_agent(observation, history, env):
    # Pass the utterance and image to a language model
    screen_width = env.unwrapped.action_space_config.screen_width
    screen_height = env.unwrapped.action_space_config.screen_height
    PROMPT = """You are on a webpage. 

Here are your available actions:

NONE
Do nothing for the current step.

MOVE_COORDS x y
Move the cursor to the specified coordinates.
X Coordinates are between 0 and {}.
Y Coordinates are between 0 and {}.
Look at the history: if a coordinate didn't work previously, try a new one in a grid of size 10.

CLICK_COORDS x y
Click on the specified coordinates.
X Coordinates are between 0 and {}.
Y Coordinates are between 0 and {}.
Look at the history: if a coordinate didn't work previously, try a new one in a grid of size 10.

CLICK_ELEMENT <element_ref>
Click on the specified element using JavaScript.

----
Observation:
You see the image provided to you.

----
Task:
Here is your task:
{}

----
History:
{}
----

Output ONE action at a time using the following format:
ACTION_TYPE [ARGS]

Example 1:
CLICK_ELEMENT 1

Example 2:
NONE

Example 3:
MOVE_COORDS 25 25
----
Your action:
""".format(
   screen_width,
   screen_height,
   screen_width,
   screen_height,
   observation['utterance'],
   "\n".join(history)
)
    image = observation['screenshot']
    # image is current np arrays
    # convert to a PIL image
    from PIL import Image
    image = Image.fromarray(image)

    API_KEY = os.getenv("GEMINI_API_KEY")
    from google import genai
    
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=[PROMPT, image]
    )
    out = response.text

    # Parse the action type and args
    out = out.strip().split()
    if out[0] == "CLICK_ELEMENT":
        out = int(out[1])
        # Click on the element.
        action = env.unwrapped.create_action(
            ActionTypes.CLICK_ELEMENT, ref=out
        )
    elif out[0] == "MOVE_COORDS":
        out = [int(x) for x in out[1:]]
        action = env.unwrapped.create_action(
            ActionTypes.MOVE_COORDS, coords=np.array(out)
        )
    elif out[0] == "CLICK_COORDS":
        out = [int(x) for x in out[1:]]
        action = env.unwrapped.create_action(
            ActionTypes.CLICK_COORDS, coords=np.array(out)
        )
    elif out[0] == "NONE":
        action = env.unwrapped.create_action(ActionTypes.NONE)
    else:
        action = env.unwrapped.create_action(ActionTypes.NONE)

    print("LLM response: ", out)
    return action, response.text


gymnasium.register_envs(miniwob)

env = gymnasium.make('miniwob/click-test-2-v1', render_mode='human')
# env = gymnasium.make('miniwob/click-checkboxes', render_mode='human')

try:
  observation, info = env.reset(seed=41)
  history = []
  episode, step = 0, 0
  for _ in range(100):
    # time.sleep(1)
    action, response = vlm_agent(observation, history, env) 
    history.append(response)
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"Episode {episode} | Step {step}: {reward}, {terminated}")
    step += 1
    if terminated:
      time.sleep(2)
      print("Resetting env\n")
      observation, info = env.reset()
      history = []
      episode += 1  
      step = 0

finally:
  env.close()