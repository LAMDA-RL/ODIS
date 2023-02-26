from .episode_runner import EpisodeRunner as MultiTaskEpisodeRunner
from .parallel_runner import ParallelRunner as MultiTaskParallelRunner

REGISTRY = {}

REGISTRY["mt_episode"] = MultiTaskEpisodeRunner
REGISTRY["mt_parallel"] = MultiTaskParallelRunner