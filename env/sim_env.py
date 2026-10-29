# USD 场景由 prim 组成；Xform 是可承载位姿的节点，引用 USD 会引入其场景内容。
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
try:
    import isaacsim.storage.native as nucleus_utils
except ModuleNotFoundError:
    import isaacsim.core.utils.nucleus as nucleus_utils
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter
from isaaclab.terrains import TerrainGeneratorCfg
from env.terrain_cfg import HfUniformDiscreteObstaclesTerrainCfg
import omni.replicator.core as rep

def add_semantic_label():
    """给指定地面 prim 附加 floor 类别，供渲染语义分割读取。"""
    # 此路径为 /World/GroundPlane，而机器人配置创建的是 /World/ground；应在场景树核对。
    ground_plane = rep.get.prims("/World/GroundPlane")
    with ground_plane:
    # Add a semantic label
        rep.modify.semantics([("class", "floor")])

def create_obstacle_env():
    """将程序生成的离散障碍高度场导入场景，作为共享障碍地形。"""
    # TerrainGenerator 负责生成网格，TerrainImporter 负责把网格及碰撞配置放入仿真。
    add_semantic_label()
    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/obstacleTerrain",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50, 50),
            color_scheme="height",
            sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
                seed=0,
                size=(50, 50),
                obstacle_width_range=(0.5, 1.0),
                obstacle_height_range=(1.0, 2.0),
                num_obstacles=100 ,                   #use 200,400 to add more obstacles or 0 for plane env--------------------
                obstacles_distance=2.0,
                border_width=5,
                avoid_positions=[[0, 0]]
            )},
        ),
        visual_material=None,     
    )
    TerrainImporter(terrain) 


def create_warehouse_env():
    """从 Isaac 资产根目录引用仓库 USD；资产获取可能涉及网络或本地缓存。"""
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Simple_Warehouse/warehouse.usd"     #also--> full_warehouse.usd,, warehouse_with_forklifts.usd ,, warehouse_multiple_shelves.usd
    prim.GetReferences().AddReference(asset_path)

def create_hospital_env():
    """在 World 下创建变换节点并引用医院 USD 场景。"""
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Hospital")
    prim = define_prim("/World/Hospital", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Hospital/hospital.usd"
    prim.GetReferences().AddReference(asset_path)

def create_office_env():
    """引用办公室 USD 场景，机器人本身仍由 quadruped 环境配置创建。"""
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Office")
    prim = define_prim("/World/Office", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Office/office.usd"
    prim.GetReferences().AddReference(asset_path)

def create_rivermark_env():
    """引用 Rivermark 户外场景；不改变已加载的机器人控制策略。"""
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Outdoor")
    prim = define_prim("/World/Outdoor", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Outdoor/Rivermark/rivermark.usd"
    prim.GetReferences().AddReference(asset_path)

def create_terrain_env():
    """引用预制 slope.usd 坡地；此入口不同于程序化障碍高度场生成器。"""
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Terrains")
    prim = define_prim("/World/Terrains", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Terrains/slope.usd"      #also--> rough_plane.usd,, stairs.usd,, slope.usd
    prim.GetReferences().AddReference(asset_path)
