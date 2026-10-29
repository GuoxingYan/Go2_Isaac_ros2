from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
import env.terrain as terrain
from isaaclab.utils import configclass
from dataclasses import MISSING

@configclass
class HfUniformDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a discrete obstacles height field terrain."""
    # 高度场用二维数组描述表面高度；horizontal_scale/vertical_scale 继承自基类。
    # MISSING 标明调用方必须提供该配置项，尺寸范围使用米而不是网格下标。

    function = terrain.uniform_discrete_obstacles_terrain
    seed: float = 0
    """Env seed."""
    obstacle_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the obstacles (in m)."""
    obstacle_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the obstacles (in m)."""
    num_obstacles: int = MISSING
    """The number of obstacles to generate."""
    obstacles_distance: float = MISSING
    """The minimum distance between obstacles (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    avoid_positions: list[list[float, float]] = []
    add_stairs: bool = True
    # 当前生成函数未读取 add_stairs/add_slope，这两个开关暂不会生成台阶或坡面。
    """Whether to add stair features to the terrain."""
    add_slope: bool = True
    """Whether to add a sloped feature to the terrain."""


