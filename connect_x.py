from kaggle_environments import evaluate, make, utils
import inspect
import sys
import os

# Create an Agent
def my_agent(observation, configuration):
    from random import choice

    def checker_position(a_list):
        x = []
        y = []
        for item in range(len(a_list)):
            x.append(a_list[item]%7)
            y.append(a_list[item]/7)
        return x, y

    def indexes_of_duplication(a_list, key):
        index = []
        for item in range(len(a_list)):
            if a_list[item] == key:
                index.append(item)
        return index

    def random_action():
        from random import choice
        return choice([c for c in range(configuration.columns)])

    take_action = None

    count = 1
    epsilon = 1.0
    learning_rate = 0.5

    vertical = False
    horizontal = False
    diagonal = False

    initial_state = observation.board

    if set(initial_state) == {0}:
        take_action = random_action()
    else:
        agent_position = indexes_of_duplication(initial_state, 1)
        agent_x, agent_y = checker_position(agent_position)
        rival_position = indexes_of_duplication(initial_state, 2)
        rival_x, rival_y = checker_position(rival_position)

        if not vertical or not horizontal or not diagonal:
            action_value = 0.0
            final_reward = 10
            transition_reward = -1

            behavior_policy = []
            target_policy = []


# Evaluate your Agent
def mean_reward(rewards):
    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

def main():
    # Run multiple episodes to estimate its performance.

    # Create Connect X Environment
    env = make("connectx", debug=True)
    env.render()

    # Test your Agent
    env.reset()
    env.run([my_agent, "random"])
    env.render(mode="ipython", width=500, height=450)

    # Debug/Train your Agent
    # Play as first position against random agent
    trainer = env.train([None, "random"])
    observation = trainer.reset()

    while not env.done:
        my_action = my_agent(observation, env.configuration)
        print("My Action", my_action)
        observation, reward, done, info = trainer.step(my_action)
    env.render()

    print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))
    print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))

    write_agent_to_file(my_agent, "submission.py")

    out = sys.stdout
    submission = utils.read_file("submission.py")
    agent = utils.get_last_callable(submission)
    sys.stdout = out

    env = make("connectx", debug=True)
    env.run([agent, agent])
    print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed")

if __name__ == '__main__':
    main()
