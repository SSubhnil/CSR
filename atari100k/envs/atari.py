"""
Drop-in replacement for CSR/atari100k/envs/atari.py using ale-py instead of atari-py.

Deploy: copy or symlink to CSR/atari100k/envs/atari.py

Changes from original:
  - Removed: gym.envs.atari.AtariEnv (backed by deprecated atari-py)
  - Added:   ale_py.ALEInterface direct usage
  - Kept:    Same 4-tuple step API, observation processing, lives handling
  - Kept:    mode/difficulty constructor params
  - Kept:    gym.spaces for observation_space/action_space (CSR wrappers expect gym)
"""

import gym
import numpy as np


class Atari:
    LOCK = None
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=4,
        size=(84, 84),
        gray=True,
        noops=0,
        lives="unused",
        sticky=True,
        actions="all",
        length=108000,
        resize="opencv",
        mode=0,
        difficulty=0,
        seed=None,
    ):
        assert size[0] == size[1]
        assert lives in ("unused", "discount", "reset"), lives
        assert actions in ("all", "needed"), actions
        assert resize in ("opencv", "pillow"), resize
        if self.LOCK is None:
            import multiprocessing as mp
            mp = mp.get_context("spawn")
            self.LOCK = mp.Lock()
        self._resize = resize
        if self._resize == "opencv":
            import cv2
            self._cv2 = cv2
        if self._resize == "pillow":
            from PIL import Image
            self._image = Image

        # Normalize game name
        if name == "james_bond":
            name = "jamesbond"

        self._name = name
        self._repeat = action_repeat
        self._size = size
        self._gray = gray
        self._noops = noops
        self._lives = lives
        self._sticky = sticky
        self._length = length
        self._mode = mode
        self._difficulty = difficulty
        self._random = np.random.RandomState(seed)

        # Create ALE interface directly via ale-py
        with self.LOCK:
            self._ale = self._create_ale(name, mode, difficulty, sticky, seed)

        # Build action set
        if actions == "all":
            self._action_set = self._ale.getLegalActionSet()
        else:
            self._action_set = self._ale.getMinimalActionSet()
        self._num_actions = len(self._action_set)

        # Verify NOOP is action index 0
        assert self._ale.getMinimalActionSet()[0] == 0, "NOOP must be action 0"

        # Screen buffers (H, W, 3) for max-pooling between frames
        h, w = self._ale.getScreenDims()
        shape = (h, w, 3)
        self._buffer = [np.zeros(shape, np.uint8) for _ in range(2)]

        self._last_lives = None
        self._done = True
        self._step = 0
        self.reward_range = [-np.inf, np.inf]

    @staticmethod
    def _create_ale(name, mode, difficulty, sticky, seed):
        """Create and configure an ALEInterface instance."""
        import ale_py

        ale = ale_py.ALEInterface()
        ale.setInt("random_seed", seed if seed is not None else 0)
        ale.setFloat(
            "repeat_action_probability", 0.25 if sticky else 0.0
        )
        # We handle episode length ourselves
        ale.setInt("max_num_frames_per_episode", 0)
        ale.setBool("color_averaging", False)

        # Load ROM — ale-py ships ROMs via ale_py.roms
        rom_path = _find_rom(name)
        ale.loadROM(rom_path)

        # Set mode and difficulty after ROM load
        if mode is not None:
            ale.setMode(int(mode))
        if difficulty is not None:
            ale.setDifficulty(int(difficulty))
        ale.reset_game()
        return ale

    @property
    def observation_space(self):
        img_shape = self._size + ((1,) if self._gray else (3,))
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
            }
        )

    @property
    def action_space(self):
        space = gym.spaces.Discrete(self._num_actions)
        space.discrete = True
        return space

    def step(self, action):
        total = 0.0
        dead = False
        if len(action.shape) >= 1:
            action = np.argmax(action)
        ale_action = self._action_set[action]
        for repeat in range(self._repeat):
            reward = self._ale.act(ale_action)
            self._step += 1
            total += reward
            if repeat == self._repeat - 2:
                self._screen(self._buffer[1])
            if self._ale.game_over():
                break
            if self._lives != "unused":
                current = self._ale.lives()
                if current < self._last_lives:
                    dead = True
                    self._last_lives = current
                    break
        if not self._repeat:
            self._buffer[1][:] = self._buffer[0][:]
        self._screen(self._buffer[0])
        over = self._ale.game_over()
        self._done = over or (self._length and self._step >= self._length)
        return self._obs(
            total,
            is_last=self._done or (dead and self._lives == "reset"),
            is_terminal=dead or over,
        )

    def reset(self):
        self._ale.reset_game()
        if self._noops:
            for _ in range(self._random.randint(self._noops)):
                self._ale.act(0)  # NOOP
                if self._ale.game_over():
                    self._ale.reset_game()
        self._last_lives = self._ale.lives()
        self._screen(self._buffer[0])
        self._buffer[1].fill(0)

        self._done = False
        self._step = 0
        obs, reward, is_terminal, _ = self._obs(0.0, is_first=True)
        return obs

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        np.maximum(self._buffer[0], self._buffer[1], out=self._buffer[0])
        image = self._buffer[0]
        if image.shape[:2] != self._size:
            if self._resize == "opencv":
                image = self._cv2.resize(
                    image, self._size, interpolation=self._cv2.INTER_AREA
                )
            if self._resize == "pillow":
                image = self._image.fromarray(image)
                image = image.resize(self._size, self._image.NEAREST)
                image = np.array(image)
        if self._gray:
            weights = [0.299, 0.587, 1 - (0.299 + 0.587)]
            image = np.tensordot(image, weights, (-1, 0)).astype(image.dtype)
            image = image[:, :, None]
        return (
            {"image": image, "is_terminal": is_terminal, "is_first": is_first},
            reward,
            is_last,
            {},
        )

    def _screen(self, array):
        self._ale.getScreenRGB(array)

    def close(self):
        del self._ale


def _find_rom(name):
    """Find ROM path using ale-py's built-in ROM registry."""
    import ale_py.roms as roms

    # Normalize: "james_bond" → "jamesbond", keep lowercase with underscores
    normalized = name.lower().replace("-", "_")

    # Try direct lookup first
    try:
        return roms.get_rom_path(normalized)
    except Exception:
        pass

    # Try without underscores
    no_under = normalized.replace("_", "")
    try:
        return roms.get_rom_path(no_under)
    except Exception:
        pass

    # Search all available ROMs
    all_ids = roms.get_all_rom_ids()
    target = no_under
    for rom_id in all_ids:
        if rom_id.replace("_", "") == target:
            return roms.get_rom_path(rom_id)

    raise FileNotFoundError(
        f"ROM for game '{name}' not found in ale-py. "
        f"Available: {all_ids}"
    )
