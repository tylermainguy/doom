import itertools as it
import random
import time

import vizdoom as vzd
from vizdoom import DoomGame, Mode


def start_game():
    """
    This is mostly just for VizDOOM, not for gym extension.
    """

    game = DoomGame()

    config_file_path = "scenarios/health_gathering.cfg"
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    print("Doom initialized.")

    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    episodes = 10000
    for episode in range(episodes):
        print("Episode #{}".format(episode))

        game.new_episode()

        while not game.is_episode_finished():
            state = game.get_state()

            n = state.number
            vars = state.game_variables
            screen_buf = state.screen_buffer
            depth_buf = state.depth_buffer
            labels_buf = state.labels_buffer
            automap_buf = state.automap_buffer
            labels = state.labels

            # Makes a random action and get remember reward.
            r = game.make_action(random.choice(actions))

            # Prints state's game variables and reward.
            print("State #" + str(n))
            print("Game variables:", vars)
            print("Reward:", r)
            print("=====================")

            if sleep_time > 0:
                time.sleep(sleep_time)

    game.close()

    return game


def main():
    start_game()


if __name__ == "__main__":
    main()
