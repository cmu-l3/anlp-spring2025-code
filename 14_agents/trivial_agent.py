import gymnasium
import miniwob
import time
from miniwob.action import ActionTypes

def agent(observation, env):
    # Find the HTML element with text "ONE"
    one_element = None
    for element in observation['dom_elements']:
      if element['text'] == "ONE":
        one_element = element
        break

    # Click on the element.
    action = env.unwrapped.create_action(
      ActionTypes.CLICK_ELEMENT, ref=one_element["ref"]
    )
    return action


gymnasium.register_envs(miniwob)
env = gymnasium.make('miniwob/click-test-2-v1', render_mode='human')
try:
  observation, info = env.reset(seed=41)
  episode = 0
  step = 0
  
  for _ in range(100):
    time.sleep(1) # For human watching purposes

    action = agent(observation, env) 
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"Episode {episode} | Step {step}: {reward}, {terminated}")
    step += 1
    if terminated:
      time.sleep(1) # For human watching purposes
      observation, info = env.reset()
      print("Resetting env\n")
      episode += 1
      step = 0

finally:
  env.close()