from vizdoom import DoomGame
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution

import vizdoom as vzd


def configure_game():
    """
    This is mostly just for VizDOOM, not for gym extension.
    """

    game = DoomGame()

    game.set_doom_scenario_path("scenarios/basic.wad")
    game.set_doom_map("map01")

    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)

    game.add_available_button(Button.MOVE_LEFT)
    game.add_available_button(Button.MOVE_RIGHT)
    game.add_available_button(Button.ATTACK)

    game.add_available_game_variable(GameVariable.AMMO2)

    game.set_episode_timeout(10000)
    game.set_episode_start_time(10)
    game.set_window_visible(True)

    game.set_living_reward(-1)
    # episodes = 10
    # game = configure_game()

    # game.init()

    # shoot = [0, 0, 1]
    # left = [1, 0, 0]
    # right = [0, 1, 0]
    # actions = [shoot, left, right]

    # sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    # for episode in range(episodes):
    #    print("Episode #{}".format(episode))

    #    game.new_episode()

    #    while not game.is_episode_finished():
    #        state = game.get_state()

    #        n = state.number
    #        vars = state.game_variables
    #        screen_buf = state.screen_buffer
    #        depth_buf = state.depth_buffer
    #        labels_buf = state.labels_buffer
    #        automap_buf = state.automap_buffer
    #        labels = state.labels

    #        # Makes a random action and get remember reward.
    #        r = game.make_action(random.choice(actions))

    #        # Prints state's game variables and reward.
    #        print("State #" + str(n))
    #        print("Game variables:", vars)
    #        print("Reward:", r)
    #        print("=====================")

    #        if sleep_time > 0:
    #            time.sleep(sleep_time)

    # game.close()

    return game