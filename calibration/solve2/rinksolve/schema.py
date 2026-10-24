"""RinkSolve 2 data contracts; independent of every legacy solver.

A raw pixel is evidence, a semantic landmark is an entity identity, and a free
station is an observation-local nuisance variable. They are deliberately different
records. A landmark may have an unknown *shared* position without becoming a
per-frame sliding point. Polyline vertices are representation samples only.

Artwork is independent of its installation. Template coordinates describe the
printed design. Installation coordinates live on the continuous unwrapped board;
they do not assert that the installed print is a planar rectangle in 3-D.

All records are frozen, tuples are used instead of mutable lists/dicts, JSON is
strictly tagged, and source evidence is never rewritten by parameter updates.
This module validates data meaning and references. It does not qualify historical
packages, infer physical facts, choose residual weights, or implement a solver.
"""
from __future__ import annotations

import dataclasses
import json
import math
import types
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import ClassVar, Union, get_args, get_origin, get_type_hints

SCHEMA_VERSION = "rinksolve.data.v1"
_RECORDS: dict[str, type] = {}


def doc(description: str, *, unit=None, ref=(), **kwargs):
    return field(metadata={"description": description, "unit": unit, "ref": ref}, **kwargs)


def _check_type(value, expected, path: str) -> None:
    origin, args = get_origin(expected), get_args(expected)
    if origin in (Union, types.UnionType):
        for choice in args:
            try:
                _check_type(value, choice, path)
                return
            except (TypeError, ValueError):
                pass
        raise TypeError(f"{path}: value does not match {expected}")
    if origin is tuple:
        if not isinstance(value, tuple):
            raise TypeError(f"{path}: immutable tuple required")
        if len(args) == 2 and args[1] is Ellipsis:
            for i, item in enumerate(value):
                _check_type(item, args[0], f"{path}[{i}]")
        else:
            if len(value) != len(args):
                raise TypeError(f"{path}: expected {len(args)} entries, got {len(value)}")
            for i, (item, kind) in enumerate(zip(value, args)):
                _check_type(item, kind, f"{path}[{i}]")
        return
    if expected is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise TypeError(f"{path}: finite number required")
    elif expected in (str, int, bool, type(None)):
        if type(value) is not expected:
            raise TypeError(f"{path}: expected {expected.__name__}")
    elif not isinstance(value, expected):
        raise TypeError(f"{path}: expected {expected.__name__}")


class Record:
    """Frozen subclasses have runtime type checks as well as static annotations."""
    __slots__ = ()
    schema: ClassVar[str] = SCHEMA_VERSION

    def __post_init__(self):
        hints = get_type_hints(type(self))
        for f in fields(self):
            value = getattr(self, f.name)
            _check_type(value, hints[f.name], f"{type(self).__name__}.{f.name}")
            if f.name == "id" and not value.strip():
                raise ValueError(f"{type(self).__name__}.id: nonempty identity required")


def record(cls):
    cls.__contract_meaning__ = cls.__doc__
    cls = dataclass(frozen=True, slots=True)(cls)
    _RECORDS[cls.__name__] = cls
    return cls


class Unit(str, Enum):
    PIXEL = "pixel"
    NATIVE_PTS = "native_pts"
    SECOND = "second"
    SCENE_UNIT = "scene_unit"
    VERIFIED_METRE = "verified_m"
    NORMALIZED_2D = "normalized_2d"
    RADIAN = "radian"
    DIMENSIONLESS = "dimensionless"


class MaterialScope(str, Enum):
    ICE_BODY = "ice_body"
    BOARD_BODY = "board_body"
    UNCONFIRMED = "unconfirmed"
    OUT_OF_MODEL = "out_of_model"


class VerificationStatus(str, Enum):
    CONFIRMED = "confirmed"
    UNCONFIRMED = "unconfirmed"
    REJECTED = "rejected"


class EvidenceKind(str, Enum):
    NATIVE_VIDEO = "native_video"
    HUMAN_ANNOTATION = "human_annotation"
    SOURCE_IMAGE = "source_image"
    OFFICIAL_ARTWORK = "official_artwork"
    WEB_ARTWORK = "web_artwork"
    MEASUREMENT = "measurement"
    USER_STATEMENT = "user_statement"
    DERIVATION = "derivation"
    COMPUTATION = "computation"


class CoordinateKind(str, Enum):
    IMAGE = "image"
    WORLD = "world_cartesian"
    SURFACE = "surface_coordinate"
    BOARD_UNWRAPPED = "board_unwrapped"
    ARTWORK = "artwork_normalized"


class CurveMeaning(str, Enum):
    MATERIAL_CENTERLINE = "material_width_centerline"
    MATERIAL_BOUNDARY = "material_boundary"
    ARTWORK_CONTOUR = "artwork_contour"


class RepresentationKind(str, Enum):
    ANALYTIC = "analytic"
    POLYLINE_SAMPLES = "polyline_representation_samples"
    MESH = "surface_mesh"
    MODEL = "registered_model"


class OwnerKind(str, Enum):
    SCENE = "scene"
    ENTITY = "entity"
    FEATURE = "semantic_feature"
    ARTWORK = "artwork_template"
    INSTALLATION = "board_installation"
    RIG = "physical_camera_rig"
    SHOT = "shot_motion"
    OBSERVATION_AUXILIARY = "observation_auxiliary"


class VariableMode(str, Enum):
    FREE = "free"
    FIXED = "fixed_by_definition"
    DERIVED = "derived"


class AssociationUse(str, Enum):
    FIT = "fit"
    HOLDOUT = "holdout"
    INACTIVE = "inactive"
    REJECTED = "rejected"


class ConstraintKind(str, Enum):
    PHYSICAL_EQUALITY = "physical_equality"
    PHYSICAL_INEQUALITY = "physical_inequality"
    STATISTICAL_PRIOR = "statistical_prior"
    COORDINATE_GAUGE = "coordinate_gauge"
    ESTIMATION_CONVENTION = "estimation_convention"
    MODEL_DOMAIN = "model_domain"


class Authority(str, Enum):
    USER_CONFIRMED = "user_confirmed"
    USER_WORKING_CHOICE = "user_working_choice"
    MEASURED = "measured"
    MATHEMATICAL_DEFINITION = "mathematical_definition"
    ENGINEERING_PROPOSAL = "engineering_proposal"
    UNAPPROVED = "unapproved"


class DependenceMode(str, Enum):
    JOINT_RAW_SUPPORT = "joint_raw_support"
    DERIVED_REPLACEMENT = "derived_replaces_parents"
    INDEPENDENT = "independent_verified"


@record
class UnitRef(Record):
    unit: Unit = doc("显式单位；scene_unit 不等于已验证米制。")
    reference_id: str | None = doc("像素/PTS的源、场景坐标系、模板或已验证米制坐标系。", default=None)


@record
class Quantity(Record):
    value: float = doc("有限数值；不附带默认噪声或物理范围。", unit="unit")
    unit: UnitRef = doc("数值单位及其基准。")


@record
class RecordRevision(Record):
    id: str = doc("稳定对象身份。")
    revision: int = doc("该对象当前修订；修订不改变身份。", unit="dimensionless")


@record
class Evidence(Record):
    id: str = doc("不可变证据身份；修正产生新记录。")
    kind: EvidenceKind = doc("原片、人工记录、图稿、测量、声明或计算。")
    uri: str = doc("保留实际本地地址或来源网址，不伪造缺失资产。")
    description: str = doc("证据能支持什么，不能支持什么。")
    sha256: str | None = doc("已有文件的字节哈希；远程尚未取得可为空。", default=None)
    parent_evidence_ids: tuple[str, ...] = doc("派生证据的直接来源。", default=(), ref="Evidence")


@record
class SourceClock(Record):
    id: str = doc("源时钟身份。")
    time_base_num: int = doc("原生 time_base 分子。")
    time_base_den: int = doc("原生 time_base 分母。")
    source_evidence_id: str = doc("对应单节目原视频证据。", ref="Evidence")

    def seconds(self, pts: int) -> float:
        if type(pts) is not int:
            raise TypeError("native PTS must be integer")
        return pts * self.time_base_num / self.time_base_den


@record
class SourceFrame(Record):
    id: str = doc("原生曝光身份，不以导出序号替代。")
    clock_id: str = doc("原视频时钟。", ref="SourceClock")
    native_pts: int = doc("原片 native PTS，保持整数。", unit="native_pts (clock_id)")
    width: int = doc("原生像素宽度。", unit="pixel")
    height: int = doc("原生像素高度。", unit="pixel")
    evidence_id: str = doc("该帧的原片来源。", ref="Evidence")


@record
class PixelObservation(Record):
    id: str = doc("不可变原始像素观测ID。")
    frame_id: str = doc("精确原生曝光。", ref="SourceFrame")
    uv: tuple[float, float] = doc("原始像素坐标；优化不得回写。", unit="pixel")
    evidence_ids: tuple[str, ...] = doc("人工点、材料边缘或原帧证据。", ref="Evidence")
    capture_method: str = doc("实际取点方法，不把自动检测冒充人工确认。")
    covariance_px2: tuple[float, float, float] | None = doc("可证实的二维协方差 xx,xy,yy；未知保持空。", default=None, unit="pixel^2")


@record
class DerivedPixelObservation(Record):
    id: str = doc("派生图像位置ID；不能改写为独立原始点。")
    frame_id: str = doc("其父观测所属的同一原生曝光。", ref="SourceFrame")
    uv: tuple[float, float] = doc("派生图像位置，像素。", unit="pixel")
    method_id: str = doc("派生算法与版本的稳定身份。")
    parent_observation_ids: tuple[str, ...] = doc("完整直接依赖；保留原线拟合样本。", ref=("PixelObservation", "DerivedPixelObservation",))
    evidence_ids: tuple[str, ...] = doc("计算记录与审核证据。", ref="Evidence")
    covariance_px2: tuple[float, float, float] | None = doc("传播所得协方差；未知保持空。", default=None, unit="pixel^2")


@record
class CoordinateSystem(Record):
    id: str = doc("场景/图像/展开面/模板的坐标系。")
    kind: CoordinateKind = doc("坐标意义，不混用透视像素与欧氏/米制。")
    dimension: int = doc("坐标维数。")
    axis_names: tuple[str, ...] = doc("每个坐标轴的明确名称及顺序。")
    units: tuple[UnitRef, ...] = doc("逐轴单位。")
    definition: str = doc("原点、方向和尺度定义；不暗中设标准尺寸。")
    metric_evidence_ids: tuple[str, ...] = doc("使用 verified_m 必须有独立米制依据。", default=(), ref="Evidence")


@record
class Scene(Record):
    id: str = doc("赛事共同场景身份。")
    coordinate_system_id: str = doc("共同世界坐标系；尺度状态显式。", ref="CoordinateSystem")


@record
class CameraRig(Record):
    id: str = doc("物理机位身份；不等于一个shot。")
    scene_id: str = doc("机位安装所属场景。", ref="Scene")
    definition: str = doc("实际采用的安装/光学定义，不写默认硬件假设。")
    evidence_ids: tuple[str, ...] = doc("机位身份依据。", ref="Evidence")


@record
class Shot(Record):
    id: str = doc("镜头身份。")
    rig_id: str = doc("所属物理机位。", ref="CameraRig")
    clock_id: str = doc("源时钟。", ref="SourceClock")
    frame_ids: tuple[str, ...] = doc("当前镜头原生曝光；不以重采样时钟替代。", ref="SourceFrame")


@record
class GeometryEntity(Record):
    id: str = doc("静止场地材料实体的稳定身份。")
    scene_id: str = doc("赛事共同场景。", ref="Scene")
    label: str = doc("人可读实体名；身份由ID决定。")
    claimed_scope: MaterialScope = doc("待证实的物理归属，不等于已确认。")


@record
class EntityVerification(Record):
    id: str = doc("实体归属的有据审核记录。")
    entity_id: str = doc("被审核实体。", ref="GeometryEntity")
    status: VerificationStatus = doc("未确认/拒绝保留，但不能进入拟合。")
    verified_scope: MaterialScope = doc("核实属于冰面或围挡本体才可入场地拟合。")
    static_material_confirmed: bool = doc("确认为静止实体材料；反射/人物/字幕禁止。")
    evidence_ids: tuple[str, ...] = doc("原片与人工确认依据。", ref="Evidence")
    notes: str = doc("核对结论与剩余不确定性。")


@record
class GeometryPrimitive(Record):
    id: str = doc("几何计算表示ID；采样点不是语义landmark。")
    entity_id: str = doc("所属物理材料实体。", ref="GeometryEntity")
    coordinate_system_id: str = doc("计算表示的坐标。", ref="CoordinateSystem")
    representation: RepresentationKind = doc("解析模型、曲线样本或网格；不带定位语义。")
    model_id: str = doc("新实现注册的几何函数身份，不是旧代码路径。")
    parameter_block_ids: tuple[str, ...] = doc("该表示依赖的显式变量。", ref="VariableBlock")
    asset_evidence_id: str | None = doc("外部样本/网格的证据资产，可为空。", default=None)


@record
class Interval(Record):
    lower: float | None = doc("显式允许区间下界；空表示该方向不设界。")
    upper: float | None = doc("显式允许区间上界。")
    unit: UnitRef = doc("曲线参数的实际单位；不默认0到1。")
    lower_closed: bool = doc("下界是否包含。", default=True)
    upper_closed: bool = doc("上界是否包含。", default=True)


@record
class CurveFeature(Record):
    id: str = doc("命名的物理曲线特征身份。")
    entity_id: str = doc("曲线属于哪个实体材料。", ref="GeometryEntity")
    primitive_id: str = doc("曲线几何表示。", ref="GeometryPrimitive")
    support_surface_id: str = doc("曲线所在同一材料表面身份。", ref=("GeometryPrimitive", "BoardSurface",))
    meaning: CurveMeaning = doc("宽线中线、实体边界或图案轮廓，不能混用。")
    verification_id: str = doc("曲线归属已核实的记录。", ref="EntityVerification")
    parameter_domain: Interval = doc("曲线参数域；与样本索引不同。")
    definition_evidence_ids: tuple[str, ...] = doc("该曲线测量定义的明确依据。", ref="Evidence")


@record
class SharedPointPosition(Record):
    variable_block_id: str = doc("同一实体点的共享未知坐标，绝非每帧station。", ref="VariableBlock")
    coordinate_system_id: str = doc("共享位置所在坐标系。", ref="CoordinateSystem")



@record
class SharedCurvePointPosition(Record):
    curve_feature_id: str = doc("该固定实体点所在已确认曲线。", ref="CurveFeature")
    station_variable_block_id: str = doc("同一实体点全帧共享的未知station；owner是feature，不是观测。", ref="VariableBlock")
    allowed_interval: Interval = doc("固定点位置可估计的明确范围，不因逐帧观测而独立滑动。")


@record
class KnownPointPosition(Record):
    coordinates: tuple[float, ...] = doc("已知位置的数值；初值不能写进此字段。", unit="coordinate_system_id units")
    coordinate_system_id: str = doc("位置单位与坐标定义。", ref="CoordinateSystem")
    definition_constraint_id: str = doc("该数值为何被固定：测量/用户确认/坐标规范。", ref="Constraint")


@record
class IntersectionPosition(Record):
    curve_feature_ids: tuple[str, str] = doc("两条已确认且测量定义明确的曲线。", ref="CurveFeature")
    branch_id: str = doc("确定的唯一交点分支身份；不是最近初值选择。")
    branch_definition: str = doc("分支的实体/拓扑定义，准备阶段须验证唯一存在。")
    branch_evidence_ids: tuple[str, ...] = doc("分支识别依据；数学唯一性证明亦需留存。", ref="Evidence")


@record
class StructuralPointPosition(Record):
    definition_id: str = doc("明确的实体角点/端点等定义，新注册几何函数。")
    geometry_feature_ids: tuple[str, ...] = doc("定义所依赖的实体几何。")
    definition_evidence_ids: tuple[str, ...] = doc("真实角点/端点身份依据，不许用采样vertex替代。", ref="Evidence")


@record
class LandmarkFeature(Record):
    id: str = doc("一个固定实体位置的身份；位置可以尚未求出。")
    entity_id: str = doc("固定于哪个材料实体。", ref="GeometryEntity")
    verification_id: str = doc("归属核实。", ref="EntityVerification")
    label: str = doc("交点/角点等人可读名称。")
    position: SharedPointPosition | SharedCurvePointPosition | KnownPointPosition | IntersectionPosition | StructuralPointPosition = doc("共享未知、已知、交点或实体结构定义；无per-frame滑动位置。")


@record
class CurveLocusFeature(Record):
    id: str = doc("只要求在某条曲线范围内的特征定义。")
    curve_feature_id: str = doc("所允许滑动的命名曲线。", ref="CurveFeature")
    allowed_intervals: tuple[Interval, ...] = doc("允许范围；无隐式0到1或默认整线。")


@record
class RegionLocusFeature(Record):
    id: str = doc("只要求在某表面区域内，不指定固定位置。")
    entity_id: str = doc("静止材料实体。", ref="GeometryEntity")
    verification_id: str = doc("区域归属核实。", ref="EntityVerification")
    support_surface_id: str = doc("允许区域所在材料面。", ref=("GeometryPrimitive", "BoardSurface",))
    region_model_id: str = doc("明确的新区域定义注册ID。")
    definition_feature_ids: tuple[str, ...] = doc("区域边界/结构定义依赖。")
    coordinate_system_id: str = doc("观测局部区域坐标的单位。", ref="CoordinateSystem")



class TemplateExtent(str, Enum):
    WHOLE_BANNER = "whole_banner"
    COMPONENT_LOGO = "component_logo"
    DESIGN_SUBREGION = "design_subregion"


class ContourRole(str, Enum):
    OUTER = "outer"
    INNER_HOLE = "inner_hole"
    OPEN_STROKE = "open_stroke"


@record
class ArtworkSourceFrame(Record):
    evidence_id: str = doc("实际高清图像/SVG来源。", ref="Evidence")
    pixel_dimensions: tuple[int, int] | None = doc("原始栅格宽高；矢量源可为空。")
    view_box: tuple[float, float, float, float] = doc("源图稿viewBox的x,y,width,height，保留原始比例与裁切范围。")
    aspect_definition: str = doc("来源viewBox比例如何解释；logo比例不是现场整幅横幅比例。")


@record
class ArtworkStyle(Record):
    id: str = doc("模板材料原色/绘制样式证据，不是求解硬物理约束。")
    color_space: str = doc("原始颜色空间。")
    fill_rgba: tuple[float, float, float, float] | None = doc("来源填色，未知或非纯色可为空。")
    stroke_rgba: tuple[float, float, float, float] | None = doc("来源描边色，未知可为空。")
    evidence_ids: tuple[str, ...] = doc("样式的来源图稿依据。", ref="Evidence")


@record
class ArtworkShape(Record):
    id: str = doc("单个字形/图案轮廓的模板内身份。")
    role: str = doc("glyph、mark、outline等实际语义。")
    path_evidence_id: str = doc("保留高清源及矢量化的可追溯路径资产。")
    closed: bool = doc("轮廓实际是否闭合。")
    topology_role: ContourRole = doc("外轮廓、内洞或开放笔画的明确拓扑。")
    group_id: str = doc("同一字形/图案组的稳定ID，多个contour不是无关片段。")
    parent_shape_id: str | None = doc("内洞/嵌套轮廓所属外轮廓；外层可空。", default=None, ref="ArtworkShape")
    style_id: str | None = doc("来源填色/描边样式，仅作模板证据。", default=None, ref="ArtworkStyle")
    text: str | None = doc("确实识别的文字；未知不补造。", default=None)


@record
class ArtworkLandmark(Record):
    id: str = doc("模板内固定位置，如字形尖角，区别于轮廓采样。")
    uv: tuple[float, float] = doc("模板规范坐标，normalized_2d。", unit="normalized_2d")
    definition: str = doc("实体字形/图案位置定义；不以任意采样顶点冒充。")
    evidence_ids: tuple[str, ...] = doc("模板位置依据。", ref="Evidence")


@record
class ArtworkTemplate(Record):
    id: str = doc("广告设计对象；不等于围挡上某一次安装。")
    revision: int = doc("设计/轮廓修订。", unit="dimensionless")
    brand: str = doc("品牌身份；同品牌不同版式不能自动等同。")
    variant: str = doc("具体版本、颜色、语言、裁切版式。")
    coordinate_system_id: str = doc("统一模板2D坐标、原点和轴向。", ref="CoordinateSystem")
    extent: TemplateExtent = doc("整幅横幅、组成logo或局部设计；不强给彼此相同比例。")
    source_frames: tuple[ArtworkSourceFrame, ...] = doc("源尺寸/viewBox与比例依据；与现场物理宽高分离。")
    source_evidence_ids: tuple[str, ...] = doc("高清图可来自官网/网络，视频只是安装核对证据。", ref="Evidence")
    shapes: tuple[ArtworkShape, ...] = doc("文字字形、图案和外轮廓的完整对象层。")
    landmarks: tuple[ArtworkLandmark, ...] = doc("可重复识别的固定模板位置。")
    completeness_notes: str = doc("已拆清/缺失/裁切/尚待确认的部分。")
    styles: tuple[ArtworkStyle, ...] = doc("材料原色/绘制样式证据，不自动变成物理约束。", default=())
    parent_template_id: str | None = doc("组件所属整幅设计若已知则明确引用。", default=None, ref="ArtworkTemplate")


@record
class ArtworkLayoutGroup(Record):
    """A source logo/text group in one shared static installed design layout."""
    id: str = doc("整组文字/图案的现场拼版身份；不是单帧自由字形。")
    source_template_id: str = doc("真实源组件图稿，保留官网组合与现场重排的区别。", ref="ArtworkTemplate")
    source_feature_ids: tuple[str, ...] = doc("该组所属源shape/landmark；组内原形状不任意变形。", ref=("ArtworkShape", "ArtworkLandmark"))
    transform_model_id: str = doc("已注册的跨帧共享静态相似变换，或源坐标恒等。")
    parameter_block_ids: tuple[str, ...] = doc("拼版位置/统一比例/转角的共享变量，归该组而非曝光。", ref="VariableBlock")
    match_status: VerificationStatus = doc("该组件版本与现场材料的核对状态。")
    match_evidence_ids: tuple[str, ...] = doc("现场拼版与源组件核对证据。", ref="Evidence")
    notes: str = doc("源组合比例是否适用，缺失组件和未知留白。")


@record
class ArtworkLayout(Record):
    """Source components are composed before the whole banner is installed."""
    id: str = doc("一次真实印刷设计的共享拼版对象；区别源图稿与安装。")
    revision: int = doc("现场拼版修订。", unit="dimensionless")
    coordinate_system_id: str = doc("拼版的normalized_2d坐标与外矩形定义；不继承源画布白边。", ref="CoordinateSystem")
    group_ids: tuple[str, ...] = doc("完整组件组；组之间位置可共享估计，不逐帧重排。", ref="ArtworkLayoutGroup")
    evidence_ids: tuple[str, ...] = doc("现场原片和真实印刷设计的拼版依据。", ref="Evidence")
    banner_extent_definition: str = doc("真实广告外边界如何定义；未见边界明确未知，源画布不自动等于边界。")
    notes: str = doc("整体拼版已知/待估/缺失说明，不冒充整幅母版。")


@record
class BoardSurface(Record):
    id: str = doc("连续整圈围挡表面，不以直线/圆弧切成互不连续广告面。")
    entity_id: str = doc("围挡本体实体。", ref="GeometryEntity")
    primitive_id: str = doc("连续三维材料面的新模型表示。", ref="GeometryPrimitive")
    unwrapped_coordinate_system_id: str = doc("整圈剪开摊平后的坐标系，含接缝与方向定义。", ref="CoordinateSystem")
    seam_definition: str = doc("展开接缝的位置和跨接缝处理定义。")
    governing_constraint_ids: tuple[str, ...] = doc("围挡物理关系与连续性依据，不附带3D平面矩形假设。", ref="Constraint")


@record
class BoardInstallation(Record):
    id: str = doc("一个具体实体广告安装对象。")
    board_surface_id: str = doc("位于哪条连续围挡表面。", ref="BoardSurface")
    template_id: str | None = doc("已确定匹配的图稿对象；无高清图也可先登记现场安装。", ref="ArtworkTemplate")
    template_match_status: VerificationStatus = doc("同品牌不构成版式已匹配。")
    template_match_evidence_ids: tuple[str, ...] = doc("原片中具体文字/颜色/布局与模板的核对证据。", ref="Evidence")
    mapping_model_id: str = doc("模板2D到围挡展开面的显式新映射定义。")
    parameter_block_ids: tuple[str, ...] = doc("安装位置/尺寸/朝向等共享变量，不是每帧广告自由形变。", ref="VariableBlock")
    governing_constraint_ids: tuple[str, ...] = doc("展开矩形等已确认关系；不能暗中增加轴对齐或3D平面。", ref="Constraint")
    notes: str = doc("真实安装边界、遮挡、跨弧与接缝情况。")
    candidate_template_ids: tuple[str, ...] = doc("明确的待核对图稿候选，不自动成为实体身份。", default=(), ref="ArtworkTemplate")
    layout_id: str | None = doc("已确定现场拼版；源组件先拼版再整体映射连续围挡。", default=None, ref="ArtworkLayout")


@record
class InstalledArtworkFeature(Record):
    id: str = doc("某安装上的图案特征；不会只凭模板直接产生视频观测。")
    installation_id: str = doc("实际物理安装。", ref="BoardInstallation")
    template_feature_id: str = doc("模板landmark或shape的ID。", ref=("ArtworkShape", "ArtworkLandmark",))
    verification_id: str = doc("该安装对应围挡静止材料的归属确认。", ref="EntityVerification")
    allowed_intervals: tuple[Interval, ...] = doc("shape轮廓允许的station域；landmark必须为空。", default=())
    layout_group_id: str | None = doc("若来自现场重排拼版，明确使用哪一源组件组。", default=None, ref="ArtworkLayoutGroup")


@record
class OwnerRef(Record):
    kind: OwnerKind = doc("变量物理共享范围；与数组地址无关。")
    id: str = doc("所属对象ID。")


@record
class ParameterComponent(Record):
    name: str = doc("变量分量的稳定名称。")
    unit: UnitRef = doc("显式逐分量单位。")


@record
class VariableBlock(Record):
    id: str = doc("稳定参数块ID。")
    owner: OwnerRef = doc("场景/实体/点/机位/shot/安装/观测辅助量所有权。")
    components: tuple[ParameterComponent, ...] = doc("分量名称、维数、单位；数值另存。")
    mode: VariableMode = doc("自由、由明确依据固定、或由模型推导。")
    definition_constraint_id: str | None = doc("固定或推导的依据；不能靠初值冻结。", default=None, ref="Constraint")


@record
class ParameterValue(Record):
    variable_block_id: str = doc("引用变量布局；不重复定义owner。", ref="VariableBlock")
    values: tuple[float, ...] = doc("有限实际值，维数必须和块一致。", unit="VariableBlock.components units")


@record
class ObservationAssociation(Record):
    id: str = doc("像素与语义特征的可修订关联，不修改原像素。")
    observation_id: str = doc("原始或派生图像观测。", ref=("PixelObservation", "DerivedPixelObservation",))
    feature_id: str = doc("固定位置、curve/region locus或安装图案特征。")
    use: AssociationUse = doc("拟合/独立验证/停用/拒绝分开保留。")
    evidence_ids: tuple[str, ...] = doc("关联判定依据。", ref="Evidence")
    local_variable_block_id: str | None = doc("仅locus有观测局部station/区域坐标；landmark禁止。", default=None, ref="VariableBlock")
    reason: str = doc("关联或排除理由。", default="")


@record
class ExpressionReference(Record):
    record_id: str = doc("约束显式依赖对象。")
    field_path: str = doc("已注册模型字段；不执行任意代码字符串。")


@record
class NamedQuantity(Record):
    name: str = doc("表达式常量名称。")
    quantity: Quantity = doc("常量数值及单位。")


@record
class Constraint(Record):
    id: str = doc("物理事实/工作选择/坐标规范的稳定身份。")
    revision: int = doc("该声明当前修订。", unit="dimensionless")
    kind: ConstraintKind = doc("物理等式/不等式/统计先验与数学规范分离。")
    authority: Authority = doc("用户确认、用户暂用、测量、数学定义、提案或未批准。")
    enabled: bool = doc("只有当前已授权的定义能被准备阶段启用。")
    operator_id: str = doc("新计算核注册的表达式，不从描述字符串eval。")
    references: tuple[ExpressionReference, ...] = doc("显式作用对象与字段。")
    constants: tuple[NamedQuantity, ...] = doc("逐常量单位，不默认米制/噪声/阈值。")
    evidence_ids: tuple[str, ...] = doc("来源原句、测量或数学定义依据。", ref="Evidence")
    description: str = doc("准确含义、作用范围和限制。")


@record
class RevisionStamp(Record):
    data: int = doc("原始证据与观测关联修订。")
    model: int = doc("实体/模型/变量布局修订。")
    constraints: int = doc("约束台账修订。")
    initialization: int = doc("初值修订；不等于物理事实修订。")


@record
class DependencyNode(Record):
    id: str = doc("输入、模型、约束、准备缓存或状态节点。")
    revision: int = doc("仅本节点修订。", unit="dimensionless")
    kind: str = doc("模块/数据类型。")


@record
class DependencyEdge(Record):
    input_id: str = doc("发生变化的上游对象。")
    consumer_id: str = doc("确实消费该上游的下游对象。")
    reason: str = doc("为何依赖；不把历史生产资格链接成普通求解前置。")


@record
class RevisionGraph(Record):
    nodes: tuple[DependencyNode, ...] = doc("当前影响图节点。")
    edges: tuple[DependencyEdge, ...] = doc("从输入到消费者的有向依赖。")

    def affected(self, changed_ids: tuple[str, ...]) -> tuple[str, ...]:
        known = {n.id for n in self.nodes}
        missing = set(changed_ids) - known
        if missing:
            raise ValueError(f"unknown changed nodes: {sorted(missing)}")
        affected, pending = set(changed_ids), list(changed_ids)
        while pending:
            input_id = pending.pop()
            for edge in self.edges:
                if edge.input_id == input_id and edge.consumer_id not in affected:
                    affected.add(edge.consumer_id)
                    pending.append(edge.consumer_id)
        return tuple(sorted(affected))


@record
class ProjectionContract(Record):
    id: str = doc("投影数学定义的明确身份。")
    implementation_id: str = doc("新投影注册实现。")
    distortion_definition: str = doc("实际畸变定义，包括明确禁用时的说明。")
    exposure_time_definition: str = doc("global/逐行时钟等明确说明，不从相机型号猜。")
    coordinate_convention: str = doc("R、C、轴向和内参定义。")
    evidence_ids: tuple[str, ...] = doc("定义依据。", ref="Evidence")


@record
class ProjectionProof(Record):
    id: str = doc("投影/真实画面验证报告身份。")
    source_contract_id: str = doc("对照的真实显示或物理投影契约。", ref="ProjectionContract")
    target_contract_id: str = doc("新初始化/求解投影契约。", ref="ProjectionContract")
    frame_ids: tuple[str, ...] = doc("已实际核验的原生曝光覆盖，不虚报完整。", ref="SourceFrame")
    stage: str = doc("数学等价、原片叠图或独立整镜头验证，不能互代。")
    status: str = doc("pending/passed/failed；报告保存真实判断依据。")
    evidence_ids: tuple[str, ...] = doc("真实输出、误差及审阅证据。", ref="Evidence")
    acceptance_constraint_ids: tuple[str, ...] = doc("若宣布通过，采用什么明确验收定义。", ref="Constraint")
    limitations: str = doc("例如平面demo等价不等于Brown/RS等价。")


@record
class InitialState(Record):
    id: str = doc("新的参数初值包，没有旧优化器lambda/rho。")
    stamp: RevisionStamp = doc("导入所对应的当前数据/模型/约束/初值修订。")
    parameter_values: tuple[ParameterValue, ...] = doc("只给数值，不自动固定实体位置。")
    source_evidence_ids: tuple[str, ...] = doc("成功demo/测量/人工提供的数据。", ref="Evidence")
    frame_ids: tuple[str, ...] = doc("实际导入覆盖的源PTS。", ref="SourceFrame")
    projection_contract_id: str = doc("实际采用的投影，不偷换物理模型。", ref="ProjectionContract")
    proof_ids: tuple[str, ...] = doc("独立转换等价/不等价报告。", ref="ProjectionProof")
    batched_parameter_table_ids: tuple[str, ...] = doc("批量station等初值的外部完整参数表。", ref="ColumnTableRef", default=())


@record
class LayoutEntry(Record):
    variable_block_id: str = doc("稳定参数块ID。", ref="VariableBlock")
    offset: int = doc("PreparedProblem本次明确数值布局的起点。", unit="dimensionless")
    size: int = doc("布局维数，必须与定义一致。", unit="dimensionless")



class SelectionMode(str, Enum):
    SMALL_IDS = "small_ids"
    ROW_INDEX = "external_row_index"
    ROW_MASK = "external_row_mask"


@record
class SelectionRef(Record):
    id: str = doc("稳定批选择身份，不将百万个字符串写进JSON。")
    table_id: str = doc("被选择的外部列式表。", ref="ColumnTableRef")
    generation: int = doc("精确表布局generation。", unit="dimensionless")
    mode: SelectionMode = doc("小型ID列表、外部row索引或外部mask，三者只能选一。")
    ids: tuple[str, ...] = doc("仅小集合使用稳定ID，不是当前row号。", default=())
    row_index_array_id: str | None = doc("大量选择使用外部整数索引。", ref="TypedArrayRef", default=None)
    row_mask_array_id: str | None = doc("大量选择使用外部布尔/uint8 mask。", ref="TypedArrayRef", default=None)
    selected_count: int | None = doc("实际读取后已知的有效选择数；未读为None。", unit="dimensionless", default=None)


@record
class NoiseFamily(Record):
    id: str = doc("共享原始证据支持的噪声/依赖族。")
    observation_ids: tuple[str, ...] = doc("原始或派生观测，必须覆盖factor实际使用。")
    mode: DependenceMode = doc("联合原始支持、派生替代父样本、或已证实独立。")
    observation_selection: SelectionRef | None = doc("生产批观测选择；与小型observation_ids二选一。", default=None)
    support_relation_table_id: str | None = doc("派生点到原始支持的外部关系表；不能把交点冒充独立信息。", ref="ColumnTableRef", default=None)
    joint_model_id: str | None = doc("共享支持联合计数采用的新统计模型；未实现不能假独立。", default=None)
    evidence_ids: tuple[str, ...] = doc("统计/替代定义依据；未知噪声不能补成确定值。", default=(), ref="Evidence")


@record
class FactorBlock(Record):
    id: str = doc("已组装的新因素身份。")
    kernel_id: str = doc("实现残差/Jacobian的新注册核。")
    association_ids: tuple[str, ...] = doc("图像因素实际消费的关联。", ref="ObservationAssociation")
    constraint_ids: tuple[str, ...] = doc("物理因素实际消费的声明。", ref="Constraint")
    variable_block_ids: tuple[str, ...] = doc("稀疏依赖，不隐式抓取全局变量。", ref="VariableBlock")
    residual_units: tuple[UnitRef, ...] = doc("逐残差单位；像素和物理等式不混。")
    noise_family_id: str | None = doc("有图像观测必须声明证据依赖族。", default=None, ref="NoiseFamily")
    association_selection: SelectionRef | None = doc("生产批使用外部关联表选择；与小型association_ids二选一。", default=None)
    batched_variable_block_ids: tuple[str, ...] = doc("批量station/其他同构变量依赖；不展开为百万参数对象。", ref="BatchedVariableBlock", default=())


@record
class PreparedProblem(Record):
    id: str = doc("当前完整问题实例，不是历史生产资格。")
    stamp: RevisionStamp = doc("精确当前修订；变化只影响必要依赖。")
    active_shot_id: str = doc("当前允许拟合的单镜头。")
    frame_ids: tuple[str, ...] = doc("当前镜头准备覆盖的原生曝光。", ref="SourceFrame")
    projection_contract_id: str = doc("求解与renderer共同投影。", ref="ProjectionContract")
    initialization_id: str = doc("新初值包。", ref="InitialState")
    layout: tuple[LayoutEntry, ...] = doc("同一check/fit数值布局。")
    factors: tuple[FactorBlock, ...] = doc("实际完整因素集合，不是统计摘要。")
    noise_families: tuple[NoiseFamily, ...] = doc("相关原始/派生观测的显式计数模型。")
    preparation_evidence_ids: tuple[str, ...] = doc("当前输入一致性、全量前向/导数实际检查。", ref="Evidence")
    fingerprint: str = doc("当前问题的内容身份；不能只用旧包/旧回执版本。")
    batched_variable_block_ids: tuple[str, ...] = doc("本次实际采用的列式批变量；全局小变量仍在layout。", ref="BatchedVariableBlock", default=())


@record
class ArrayArtifact(Record):
    path: str = doc("Data下新generation的新数组资产；不得覆盖仍被memmap/求解器消费的文件。")
    sha256: str = doc("数组实际字节哈希。")
    dtype: str = doc("明确dtype。")
    shape: tuple[int, ...] = doc("明确shape。", unit="dimensionless")


@record
class OptimizerState(Record):
    implementation_id: str = doc("新优化器与状态格式身份。")
    algorithm_revision: int = doc("新实现版本，不导入旧失败状态。")
    scalars: tuple[NamedQuantity, ...] = doc("实际算法标量，不能冒充物理先验。")
    arrays: tuple[ArrayArtifact, ...] = doc("完整可恢复算法数组。")


@record
class RunState(Record):
    id: str = doc("实际运行身份；未开始不制造迭代数/目标值。")
    prepared_problem_id: str = doc("实际运行的同一PreparedProblem。", ref="PreparedProblem")
    problem_fingerprint: str = doc("防止恢复到不同数据或布局。")
    stamp: RevisionStamp = doc("运行所用修订。")
    active_shot_id: str = doc("当前只有一个镜头。")
    status: str = doc("实际状态，prepared/running/completed/failed。")
    iteration: int = doc("真实迭代序号。", unit="dimensionless")
    accepted_steps: int = doc("实际接受步数。", unit="dimensionless")
    rejected_steps: int = doc("实际拒绝步数。", unit="dimensionless")
    parameter_values: tuple[ParameterValue, ...] = doc("完整参数及辅助量；先完成新数组文件，再原子替换state manifest，不覆盖运行中映射。")
    optimizer_state: OptimizerState | None = doc("真正续算所需的新完整状态；未运行可空。")
    main_log_path: str = doc("每次运行的主日志绝对地址。")
    batched_parameter_table_ids: tuple[str, ...] = doc("原子checkpoint保存的完整外部批参数/辅助表。", ref="ColumnTableRef", default=())



@record
class RigMount(Record):
    id: str = doc("物理机位的共享安装定义，区别于逐曝光光心。")
    rig_id: str = doc("真实物理机位。", ref="CameraRig")
    pivot_variable_block_id: str = doc("固定云台轴心的共享位置变量；owner必须是该rig。", ref="VariableBlock")
    lever_variable_block_id: str = doc("相机本地lever的共享变量；不等于逐帧自由光心。", ref="VariableBlock")
    coordinate_system_id: str = doc("pivot与世界位移采用的共同坐标。", ref="CoordinateSystem")
    camera_local_axis_definition: str = doc("lever相机本地坐标轴与正方向。")
    center_mapping_definition: str = doc("例如 C=P+R^T lever；必须是当前批准定义。")
    governing_constraint_ids: tuple[str, ...] = doc("固定pivot/lever及高度等实际启用依据。", ref="Constraint")


@record
class ModelRevision(Record):
    id: str = doc("完整场景、安装、相机投影的同一模型身份。")
    revision: int = doc("完整模型修订。", unit="dimensionless")
    stamp: RevisionStamp = doc("该完整模型所对应的分离修订。")
    scene_id: str = doc("唯一共同场景。", ref="Scene")
    coordinate_system_id: str = doc("世界单位与尺度身份。", ref="CoordinateSystem")
    projection_contract_id: str = doc("总览、叠图和求解共同投影定义。", ref="ProjectionContract")
    primitive_ids: tuple[str, ...] = doc("完整场地几何对象集合，不只本帧可见部分。", ref="GeometryPrimitive")
    rig_mount_ids: tuple[str, ...] = doc("三机位等所有实际安装；不是只显示当前机位。", ref="RigMount")
    installation_ids: tuple[str, ...] = doc("完整广告安装集合，不只当前可见广告。", ref="BoardInstallation")
    parameter_state_id: str = doc("同一InitialState或RunState数值；禁止混demo几何/d78相机。", ref=("InitialState", "RunState",))


@record
class ExposurePose(Record):
    id: str = doc("当前模型某次曝光的派生只读相机状态。")
    model_revision: RecordRevision = doc("完整模型的精确版本。")
    frame_id: str = doc("源PTS对应曝光。", ref="SourceFrame")
    shot_id: str = doc("该曝光所属镜头。", ref="Shot")
    rig_mount_id: str = doc("共享pivot/lever安装来源。", ref="RigMount")
    coordinate_system_id: str = doc("与完整场景相同世界单位，不能偷偷换米制。", ref="CoordinateSystem")
    rotation_world_to_camera: tuple[float, float, float, float, float, float, float, float, float] = doc("R_cw逐行9分量，来自同一参数状态。", unit="dimensionless")
    optical_center_world: tuple[float, float, float] = doc("派生光心C，不能当作独立每帧平移自由度。", unit="coordinate_system_id units")
    intrinsics_px: tuple[float, float, float, float, float, float, float, float, float] = doc("像素内参K逐行9分量；畸变另由投影契约指定。", unit="pixel intrinsics; homogeneous row dimensionless")


@record
class InspectionSnapshot(Record):
    id: str = doc("可视化总览/叠图使用的同一真实模型快照。")
    model_revision: RecordRevision = doc("禁止跨修订拼图的精确typed link。")
    parameter_state_id: str = doc("与ModelRevision同一参数状态。", ref=("InitialState", "RunState",))
    coordinate_system_id: str = doc("共同世界坐标、单位与米制验证状态。", ref="CoordinateSystem")
    scene_id: str = doc("完整场地。", ref="Scene")
    all_primitive_ids: tuple[str, ...] = doc("完整冰场全部几何表示；缺失必须报告。", ref="GeometryPrimitive")
    all_installation_ids: tuple[str, ...] = doc("全部广告对象。", ref="BoardInstallation")
    all_rig_mount_ids: tuple[str, ...] = doc("全部物理机位相对位置。", ref="RigMount")
    exposure_pose_ids: tuple[str, ...] = doc("选择源帧时可叠加的派生光心/R/K。", ref="ExposurePose")
    missing_items: tuple[str, ...] = doc("真正未恢复/未定义对象，绝不补假值。")
    evidence_ids: tuple[str, ...] = doc("该快照来源与派生记录。", ref="Evidence")



class ArrayRole(str, Enum):
    IMMUTABLE_EVIDENCE = "immutable_evidence"
    DERIVED_EVIDENCE = "derived_evidence"
    SEMANTIC_FOREIGN_KEYS = "semantic_foreign_keys"
    PARAMETER_STATE = "parameter_state"
    AUXILIARY_STATE = "auxiliary_state"
    COMPUTE_CACHE = "compute_cache"


class ColumnMeaning(str, Enum):
    STABLE_ID = "stable_id"
    RAW_U = "raw_pixel_u"
    RAW_V = "raw_pixel_v"
    NATIVE_PTS = "native_pts"
    SOURCE_FRAME_ID = "source_frame_id"
    ENTITY_ID = "entity_id"
    FEATURE_ID = "feature_id"
    INSTALLATION_ID = "installation_id"
    PARENT_OBSERVATION_ID = "parent_observation_id"
    OBSERVATION_ID = "observation_id"
    ASSOCIATION_USE = "association_use"
    PARAMETER_VALUE = "parameter_value"
    AUXILIARY_VALUE = "auxiliary_value"
    OTHER = "explicit_other"


@record
class TypedArrayRef(Record):
    id: str = doc("Data下数组的稳定资产ID；不在JSON中内嵌百万行。")
    path: str = doc("实际外部数组绝对地址；新generation用新文件，禁止覆盖仍被映射的输入。")
    storage: str = doc("npy/npz/raw等明确读取格式；不得pickle对象数组。")
    dtype: str = doc("含字节序的明确dtype，例如<i8/<f8；不能靠指针重解释类型。")
    shape: tuple[int, ...] = doc("数组的实际维数和shape。", unit="dimensionless")
    unit: UnitRef = doc("整列单位；不同单位的字段必须分列。")
    role: ArrayRole = doc("不可变证据、关联外键、参数、辅助状态或缓存分离。")
    read_only: bool = doc("证据必须只读；工作参数是否可写须显式。")
    generation: int = doc("当前数组布局generation，与语义版本分离。", unit="dimensionless")
    clock_id: str | None = doc("带PTS的列明确源时钟。", default=None, ref="SourceClock")
    coordinate_system_id: str | None = doc("坐标/长度列明确共同坐标系。", default=None, ref="CoordinateSystem")
    member: str | None = doc("npz等容器的精确成员名。", default=None)
    sha256: str | None = doc("资产交付时的当前摘要；普通启动不重算历史链。", default=None)


@record
class ColumnRef(Record):
    name: str = doc("稳定字段名，不以列号当身份。")
    array_id: str = doc("该列外部typed array。", ref="TypedArrayRef")
    meaning: ColumnMeaning = doc("该列数据语义；other必须有明确说明。")
    description: str = doc("实际内容与缺失值约定，不补造缺失。")


@record
class ColumnForeignKey(Record):
    column_name: str = doc("当前表的稳定外键列。")
    target_table_id: str = doc("目标实体/feature/安装/观测表。", ref="ColumnTableRef")
    target_column_name: str = doc("目标稳定ID列，不依赖当前行顺序。")


@record
class IdIndexRef(Record):
    id: str = doc("稳定ID到当前行的外部索引，不做巨型JSON字典。")
    table_id: str = doc("被索引的表。", ref="ColumnTableRef")
    key_array_id: str = doc("稳定ID数组。", ref="TypedArrayRef")
    row_array_id: str = doc("对应行号整数数组。", ref="TypedArrayRef")
    generation: int = doc("必须匹配当前表布局generation。", unit="dimensionless")
    lookup_definition: str = doc("sorted/search/hash等实际索引算法与重复处理定义。")


@record
class ColumnTableRef(Record):
    id: str = doc("向量化SoA数据表，独立于少量语义对象。")
    row_count: int = doc("实际行数；不能用原始点数虚报有效点。", unit="dimensionless")
    generation: int = doc("列与ID索引的共同布局版本。", unit="dimensionless")
    role: ArrayRole = doc("本表证据/关联/参数角色，不混为通用dict。")
    columns: tuple[ColumnRef, ...] = doc("每列明确dtype/shape/unit/role的数据引用。")
    primary_key_column: str = doc("稳定ID主键列。")
    id_index_id: str = doc("外部稳定ID到row索引。", ref="IdIndexRef")
    foreign_keys: tuple[ColumnForeignKey, ...] = doc("稳定实体、feature、安装与帧的外键。")
    evidence_ids: tuple[str, ...] = doc("本表当前生成/确认依据，不做历史资格启动依赖。", ref="Evidence")



@record
class BatchedVariableBlock(Record):
    id: str = doc("大规模同构变量块，例如每有效观测station；不是百万个VariableBlock。")
    owner_kind: OwnerKind = doc("整批一致的物理共享层；固定landmark仍归feature，不归观测。")
    owner_selection: SelectionRef = doc("当前有效owner行集合，稳定ID到row索引。")
    owner_id_column: str = doc("owner表中实际稳定身份列，禁止用行序当实体身份。")
    components: tuple[ParameterComponent, ...] = doc("全批共享一次定义的分量名称/单位。")
    parameter_table_id: str = doc("外部SoA参数表。", ref="ColumnTableRef")
    component_columns: tuple[str, ...] = doc("与components逐一对应的参数列名。")
    mode: VariableMode = doc("自由、明确固定或剖面派生。")
    definition_constraint_id: str | None = doc("固定/派生的实际定义依据。", ref="Constraint", default=None)


@record
class ComputeBufferBinding(Record):
    array_id: str = doc("当前输入/参数数组。", ref="TypedArrayRef")
    device: str = doc("cpu或明确GPU设备ID；不把监控当加速完成。")
    dtype: str = doc("执行层实际dtype，必须与数组一致。")
    shape: tuple[int, ...] = doc("执行层实际shape。", unit="dimensionless")
    byte_strides: tuple[int, ...] = doc("明确字节stride，禁止类型混用。", unit="byte")
    generation: int = doc("当前CPU/GPU驻留布局generation。", unit="dimensionless")


@record
class ComputeLayout(Record):
    id: str = doc("CPU/GPU共同数学问题的明确驻留布局。")
    problem_fingerprint: str = doc("当前PreparedProblem内容身份。")
    generation: int = doc("当前layout generation，改变后必须使旧buffer失效。", unit="dimensionless")
    bindings: tuple[ComputeBufferBinding, ...] = doc("SoA批次的CPU/GPU显式数组绑定。")


@record
class DataBundle(Record):
    id: str = doc("当前自足数据模型；普通准备不追溯旧生产链。")
    stamp: RevisionStamp = doc("分开数据/模型/约束/初值修订。")
    active_shot_id: str = doc("当前唯一拟合镜头。")
    batched_variables: tuple[BatchedVariableBlock, ...] = doc("生产规模参数按owner-row表及共享components结构表达。", default=())
    array_refs: tuple[TypedArrayRef, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    column_tables: tuple[ColumnTableRef, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    id_indices: tuple[IdIndexRef, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    compute_layouts: tuple[ComputeLayout, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    run_states: tuple[RunState, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    evidence: tuple[Evidence, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    clocks: tuple[SourceClock, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    frames: tuple[SourceFrame, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    raw_observations: tuple[PixelObservation, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    derived_observations: tuple[DerivedPixelObservation, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    coordinate_systems: tuple[CoordinateSystem, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    scenes: tuple[Scene, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    rigs: tuple[CameraRig, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    shots: tuple[Shot, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    entities: tuple[GeometryEntity, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    verifications: tuple[EntityVerification, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    primitives: tuple[GeometryPrimitive, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    curves: tuple[CurveFeature, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    landmarks: tuple[LandmarkFeature, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    curve_loci: tuple[CurveLocusFeature, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    region_loci: tuple[RegionLocusFeature, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    artworks: tuple[ArtworkTemplate, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    artwork_layout_groups: tuple[ArtworkLayoutGroup, ...] = doc("源组件在现场共享拼版中的语义组。", default=())
    artwork_layouts: tuple[ArtworkLayout, ...] = doc("源组件→静态拼版→整体安装的中间设计层。", default=())
    board_surfaces: tuple[BoardSurface, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    installations: tuple[BoardInstallation, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    installed_artwork_features: tuple[InstalledArtworkFeature, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    rig_mounts: tuple[RigMount, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    model_revisions: tuple[ModelRevision, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    exposure_poses: tuple[ExposurePose, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    inspection_snapshots: tuple[InspectionSnapshot, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    variables: tuple[VariableBlock, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    associations: tuple[ObservationAssociation, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    constraints: tuple[Constraint, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    projection_contracts: tuple[ProjectionContract, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    projection_proofs: tuple[ProjectionProof, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())
    initial_states: tuple[InitialState, ...] = doc("当前自足模型中的类型化记录；批量数据使用外部列式数组引用，不代表历史数据已迁移。", default=())


@record
class Issue(Record):
    path: str = doc("准确字段路径/对象身份。")
    code: str = doc("机器可处理的错误原因。")
    message: str = doc("人可读修正说明。")


class ContractError(ValueError):
    def __init__(self, issues: tuple[Issue, ...]):
        self.issues = issues
        super().__init__("; ".join(f"{x.path}: {x.message}" for x in issues))


def to_json_value(value):
    """Tagged JSON; dataclass identity is preserved, arrays are not guessed."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Record):
        return {"$type": type(value).__name__, **{f.name: to_json_value(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, tuple):
        return [to_json_value(x) for x in value]
    if value is None or type(value) in (str, int, float, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value
    raise TypeError(f"not a schema value: {type(value).__name__}")


def _decode(value, expected, path):
    origin, args = get_origin(expected), get_args(expected)
    if origin in (Union, types.UnionType):
        errors = []
        for choice in args:
            try:
                return _decode(value, choice, path)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        raise ValueError(f"{path}: no valid union branch: {' | '.join(errors)}")
    if origin is tuple:
        if not isinstance(value, list):
            raise TypeError(f"{path}: JSON array required")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(x, args[0], f"{path}[{i}]") for i, x in enumerate(value))
        if len(value) != len(args):
            raise ValueError(f"{path}: expected {len(args)} entries")
        return tuple(_decode(x, kind, f"{path}[{i}]") for i, (x, kind) in enumerate(zip(value, args)))
    if isinstance(expected, type) and issubclass(expected, Enum):
        return expected(value)
    if isinstance(expected, type) and issubclass(expected, Record):
        if not isinstance(value, dict) or value.get("$type") != expected.__name__:
            raise ValueError(f"{path}: expected tagged {expected.__name__}")
        valid = {f.name for f in fields(expected)}
        extra = set(value) - valid - {"$type"}
        if extra:
            raise ValueError(f"{path}: unexpected fields {sorted(extra)}")
        hints = get_type_hints(expected)
        kwargs = {k: _decode(v, hints[k], f"{path}.{k}") for k, v in value.items() if k != "$type"}
        return expected(**kwargs)
    _check_type(value, expected, path)
    return float(value) if expected is float else value


def from_json_value(value, expected=None):
    if expected is None:
        if not isinstance(value, dict) or value.get("$type") not in _RECORDS:
            raise ValueError("unknown or missing schema record tag")
        expected = _RECORDS[value["$type"]]
    return _decode(value, expected, "$root")


def dumps(value: Record, **kwargs) -> str:
    return json.dumps(to_json_value(value), ensure_ascii=False, allow_nan=False, **kwargs)


def loads(text: str, expected=None):
    def no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return from_json_value(json.loads(text, object_pairs_hook=no_duplicate_keys,
                                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"non-finite number: {x}"))), expected)


def schema_catalog() -> dict:
    """UI-facing type/field catalogue; no sample data is presented as real data."""
    def type_name(t):
        if get_origin(t) is tuple:
            args = get_args(t)
            return {"type": "array", "immutable": True, "items": [type_name(x) for x in args if x is not Ellipsis], "variadic": bool(args and args[-1] is Ellipsis)}
        if get_origin(t) in (Union, types.UnionType):
            return {"one_of": [type_name(x) for x in get_args(t)]}
        if isinstance(t, type) and issubclass(t, Enum):
            return {"enum": t.__name__, "values": [x.value for x in t]}
        return {"type": getattr(t, "__name__", str(t))}
    return {"schema": SCHEMA_VERSION, "records": [{"name": name, "description": cls.__doc__ or "", "immutable": True,
        "fields": [{"name": f.name, **type_name(get_type_hints(cls)[f.name]), "description": f.metadata.get("description", ""),
                    "required": f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING}
                   for f in fields(cls)]} for name, cls in _RECORDS.items()]}


def _walk_records(value):
    if isinstance(value, Record):
        yield value
        for f in fields(value):
            yield from _walk_records(getattr(value, f.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from _walk_records(item)


def validate_bundle(bundle: DataBundle, *, raise_on_error: bool = True) -> tuple[Issue, ...]:
    """Check current self-contained meaning/references, without replaying history.

    Geometry branch uniqueness, numerical domains and registered kernels must be
    checked by the problem preparer. This function never claims those checks ran.
    """
    issues: list[Issue] = []
    def bad(path, code, message):
        issues.append(Issue(path, code, message))
    index: dict[str, Record] = {}
    for f in fields(bundle):
        value = getattr(bundle, f.name)
        if isinstance(value, tuple):
            for obj in value:
                if isinstance(obj, Record) and hasattr(obj, "id"):
                    if obj.id in index:
                        bad(f"{f.name}/{obj.id}", "duplicate_identity", "身份必须全局唯一，不得用名称或数组地址合并对象")
                    else:
                        index[obj.id] = obj
    for artwork in bundle.artworks:
        for child in (*artwork.shapes, *artwork.landmarks, *artwork.styles):
            if child.id in index:
                bad(f"artworks/{artwork.id}/{child.id}", "duplicate_identity", "模板内部特征ID也必须唯一")
            else:
                index[child.id] = child
    def need(ref, kind, path):
        obj = index.get(ref)
        if not isinstance(obj, kind):
            bad(path, "reference_type", f"{ref!r} 不是有效的 {getattr(kind, '__name__', str(kind))} 引用")
            return None
        return obj
    def evidence(ids, path, required=False):
        if required and not ids:
            bad(path, "evidence_missing", "该定义需要真实来源；不能用默认假设代替")
        for ref in ids:
            need(ref, Evidence, path)
    def verified(ref, entity_id, path):
        v = need(ref, EntityVerification, path)
        if v and (v.entity_id != entity_id or v.status is not VerificationStatus.CONFIRMED or
                  not v.static_material_confirmed or v.verified_scope not in (MaterialScope.ICE_BODY, MaterialScope.BOARD_BODY)):
            bad(path, "material_not_confirmed", "拟合对象必须有对应实体的静止冰面/围挡本体确认")
        return v
    def interval(x, path):
        if x.lower is not None and x.upper is not None:
            if x.lower > x.upper or (x.lower == x.upper and not (x.lower_closed and x.upper_closed)):
                bad(path, "empty_interval", "参数范围为空")
    def covariance(c, path):
        if c is not None and (c[0] < 0 or c[2] < 0 or c[0] * c[2] < c[1] * c[1]):
            bad(path, "covariance_not_psd", "协方差必须半正定；未知应为空")
    def no_cycles(edges, path):
        state: dict[str, int] = {}
        def visit(node):
            if state.get(node) == 1:
                bad(path + "/" + node, "dependency_cycle", "来源或定义依赖形成循环")
                return
            if state.get(node) == 2:
                return
            state[node] = 1
            for parent in edges.get(node, ()):
                if parent in edges:
                    visit(parent)
            state[node] = 2
        for node in edges:
            visit(node)
    for obj in _walk_records(bundle):
        path = f"{type(obj).__name__}/{getattr(obj, 'id', '')}"
        if hasattr(obj, "revision") and obj.revision < 0:
            bad(path, "negative_revision", "修订号必须非负")
        if isinstance(obj, RevisionStamp) and min(obj.data, obj.model, obj.constraints, obj.initialization) < 0:
            bad(path, "negative_revision", "各类修订号必须非负")
        if isinstance(obj, UnitRef):
            if obj.unit is Unit.NATIVE_PTS:
                need(obj.reference_id, SourceClock, path)
            elif obj.unit in (Unit.SCENE_UNIT, Unit.VERIFIED_METRE, Unit.NORMALIZED_2D):
                coordinate = need(obj.reference_id, CoordinateSystem, path)
                if coordinate and obj.unit is Unit.VERIFIED_METRE and not coordinate.metric_evidence_ids:
                    bad(path, "metric_unverified", "verified_m必须链接真实米制验证依据")
                if coordinate and obj.unit is Unit.NORMALIZED_2D and coordinate.kind is not CoordinateKind.ARTWORK:
                    bad(path, "unit_reference", "normalized_2d必须指向图稿坐标系")
        for f in fields(obj):
            value = getattr(obj, f.name)
            if f.name.endswith("evidence_ids"):
                evidence(value, path + "/" + f.name)
            elif f.name.endswith("evidence_id") and value is not None:
                need(value, Evidence, path + "/" + f.name)
    for e in bundle.evidence:
        if e.sha256 is not None and (len(e.sha256) != 64 or any(c not in "0123456789abcdef" for c in e.sha256)):
            bad("evidence/" + e.id, "hash_format", "SHA256必须为实际64位小写十六进制，未知应为空")
    no_cycles({e.id: e.parent_evidence_ids for e in bundle.evidence}, "evidence")
    for c in bundle.clocks:
        if c.time_base_num <= 0 or c.time_base_den <= 0:
            bad("clocks/" + c.id, "clock_time_base", "源time_base必须为正有理数")
        e = need(c.source_evidence_id, Evidence, "clocks/" + c.id)
        if e and e.kind is not EvidenceKind.NATIVE_VIDEO:
            bad("clocks/" + c.id, "native_video_required", "节目处理时钟必须来自原生单节目视频")
    exposure_keys = set()
    for frame in bundle.frames:
        need(frame.clock_id, SourceClock, "frames/" + frame.id)
        if frame.width <= 0 or frame.height <= 0:
            bad("frames/" + frame.id, "image_size", "原生图像尺寸必须为正")
        key = (frame.clock_id, frame.native_pts)
        if key in exposure_keys:
            bad("frames/" + frame.id, "duplicate_exposure", "同一源时钟的同一PTS不可伪装为不同曝光")
        exposure_keys.add(key)
    for o in (*bundle.raw_observations, *bundle.derived_observations):
        need(o.frame_id, SourceFrame, "observations/" + o.id)
        evidence(o.evidence_ids, "observations/" + o.id, True)
        covariance(o.covariance_px2, "observations/" + o.id)
        if isinstance(o, DerivedPixelObservation):
            if not o.parent_observation_ids or len(set(o.parent_observation_ids)) != len(o.parent_observation_ids):
                bad("observations/" + o.id, "derived_parent_support", "派生点必须保留非空且不重复的原始依赖")
            for parent_id in o.parent_observation_ids:
                parent = need(parent_id, (PixelObservation, DerivedPixelObservation), "observations/" + o.id)
                if parent and parent.frame_id != o.frame_id:
                    bad("observations/" + o.id, "cross_exposure_derivation", "图像交点不能混合不同曝光的像素")
    no_cycles({o.id: o.parent_observation_ids for o in bundle.derived_observations}, "derived_observations")
    for c in bundle.coordinate_systems:
        if c.dimension <= 0 or len(c.axis_names) != c.dimension or len(c.units) != c.dimension or len(set(c.axis_names)) != c.dimension:
            bad("coordinate_systems/" + c.id, "coordinate_dimension", "坐标维数、不同轴名和单位必须一致")
        if any(u.unit is Unit.VERIFIED_METRE for u in c.units) and not c.metric_evidence_ids:
            bad("coordinate_systems/" + c.id, "metric_unverified", "工程标准尺寸与条件尺度不能冒充已验证米制")
    for scene in bundle.scenes:
        c = need(scene.coordinate_system_id, CoordinateSystem, "scenes/" + scene.id)
        if c and c.kind is not CoordinateKind.WORLD:
            bad("scenes/" + scene.id, "world_coordinate", "场景必须指向共同世界坐标")
    for rig in bundle.rigs:
        need(rig.scene_id, Scene, "rigs/" + rig.id)
    active_shot = need(bundle.active_shot_id, Shot, "active_shot_id")
    for shot in bundle.shots:
        need(shot.rig_id, CameraRig, "shots/" + shot.id)
        need(shot.clock_id, SourceClock, "shots/" + shot.id)
        pts = []
        for fid in shot.frame_ids:
            frame = need(fid, SourceFrame, "shots/" + shot.id)
            if frame:
                if frame.clock_id != shot.clock_id:
                    bad("shots/" + shot.id, "shot_clock", "镜头只能使用其原生时钟")
                pts.append(frame.native_pts)
        if not pts or any(a >= b for a, b in zip(pts, pts[1:])):
            bad("shots/" + shot.id, "shot_exposure_order", "镜头原生PTS必须非空、无重复、严格递增")
    for entity in bundle.entities:
        need(entity.scene_id, Scene, "entities/" + entity.id)
    for v in bundle.verifications:
        need(v.entity_id, GeometryEntity, "verifications/" + v.id)
        if v.status is VerificationStatus.CONFIRMED:
            evidence(v.evidence_ids, "verifications/" + v.id, True)
            if not v.static_material_confirmed or v.verified_scope not in (MaterialScope.ICE_BODY, MaterialScope.BOARD_BODY):
                bad("verifications/" + v.id, "invalid_confirmation", "confirmed仅用于静止冰面/围挡本体，其他归属保留为未确认/拒绝")
    for primitive in bundle.primitives:
        need(primitive.entity_id, GeometryEntity, "primitives/" + primitive.id)
        need(primitive.coordinate_system_id, CoordinateSystem, "primitives/" + primitive.id)
        for var_id in primitive.parameter_block_ids:
            var = need(var_id, VariableBlock, "primitives/" + primitive.id)
            if var and var.owner.kind in (OwnerKind.OBSERVATION_AUXILIARY, OwnerKind.SHOT):
                bad("primitives/" + primitive.id, "static_geometry_owner", "静止场地几何不能由每帧/每shot局部变量独立改变")
    def check_support(surface_id, entity_id, path):
        support = need(surface_id, (GeometryPrimitive, BoardSurface), path)
        if support and support.entity_id != entity_id:
            bad(path, "surface_entity", "表面和特征必须属于同一静止材料实体")
    for curve in bundle.curves:
        need(curve.entity_id, GeometryEntity, "curves/" + curve.id)
        primitive = need(curve.primitive_id, GeometryPrimitive, "curves/" + curve.id)
        if primitive and primitive.entity_id != curve.entity_id:
            bad("curves/" + curve.id, "curve_entity", "几何表示不属于该命名曲线实体")
        check_support(curve.support_surface_id, curve.entity_id, "curves/" + curve.id)
        verified(curve.verification_id, curve.entity_id, "curves/" + curve.id)
        evidence(curve.definition_evidence_ids, "curves/" + curve.id, True)
        interval(curve.parameter_domain, "curves/" + curve.id)
    for landmark in bundle.landmarks:
        need(landmark.entity_id, GeometryEntity, "landmarks/" + landmark.id)
        verified(landmark.verification_id, landmark.entity_id, "landmarks/" + landmark.id)
        position = landmark.position
        if isinstance(position, SharedPointPosition):
            var = need(position.variable_block_id, VariableBlock, "landmarks/" + landmark.id)
            c = need(position.coordinate_system_id, CoordinateSystem, "landmarks/" + landmark.id)
            if var:
                if not ((var.owner.kind is OwnerKind.FEATURE and var.owner.id == landmark.id) or
                        (var.owner.kind is OwnerKind.ENTITY and var.owner.id == landmark.entity_id)):
                    bad("landmarks/" + landmark.id, "landmark_shared_owner", "固定实体点位置可共同估计，但只能归该feature或实体共享")
                if c and (len(var.components) != c.dimension or tuple(x.unit for x in var.components) != c.units):
                    bad("landmarks/" + landmark.id, "landmark_coordinate", "共享位置变量的维数与单位必须匹配坐标系")
        elif isinstance(position, SharedCurvePointPosition):
            curve = need(position.curve_feature_id, CurveFeature, "landmarks/" + landmark.id)
            var = need(position.station_variable_block_id, VariableBlock, "landmarks/" + landmark.id)
            interval(position.allowed_interval, "landmarks/" + landmark.id)
            if curve and (curve.entity_id != landmark.entity_id or position.allowed_interval.unit != curve.parameter_domain.unit):
                bad("landmarks/" + landmark.id, "shared_station_curve", "固定点与所属曲线实体/参数单位必须一致")
            if var and (var.owner != OwnerRef(OwnerKind.FEATURE, landmark.id) or len(var.components) != 1 or var.components[0].unit != position.allowed_interval.unit):
                bad("landmarks/" + landmark.id, "shared_station_owner", "固定在线上的点只有一个归该feature全帧共享station，不能归每帧观测")
        elif isinstance(position, KnownPointPosition):
            c = need(position.coordinate_system_id, CoordinateSystem, "landmarks/" + landmark.id)
            constraint = need(position.definition_constraint_id, Constraint, "landmarks/" + landmark.id)
            if c and len(position.coordinates) != c.dimension:
                bad("landmarks/" + landmark.id, "known_point_dimension", "已知位置维数不符")
            if constraint and (not constraint.enabled or landmark.id not in {r.record_id for r in constraint.references}):
                bad("landmarks/" + landmark.id, "known_point_authority", "已知坐标必须有当前启用且明确引用该点的固定依据，不能锁初值")
        elif isinstance(position, IntersectionPosition):
            a, b = [need(x, CurveFeature, "landmarks/" + landmark.id) for x in position.curve_feature_ids]
            if position.curve_feature_ids[0] == position.curve_feature_ids[1]:
                bad("landmarks/" + landmark.id, "intersection_same_curve", "交点必须由两条不同的明确曲线定义")
            if a and b and (a.support_surface_id != b.support_surface_id or a.entity_id != landmark.entity_id or b.entity_id != landmark.entity_id):
                bad("landmarks/" + landmark.id, "intersection_surface", "交点的两条曲线必须在同一已确认材料面，且定义一致")
            if not position.branch_id.strip() or not position.branch_definition.strip():
                bad("landmarks/" + landmark.id, "intersection_branch", "交点必须明确唯一实体分支；不能按初值最近点自动选")
            evidence(position.branch_evidence_ids, "landmarks/" + landmark.id, True)
        elif isinstance(position, StructuralPointPosition):
            if not position.geometry_feature_ids:
                bad("landmarks/" + landmark.id, "structural_definition", "实体端点/角点须有明确结构依赖，不得用polyline采样vertex冒充")
            for ref in position.geometry_feature_ids:
                need(ref, (CurveFeature, LandmarkFeature, RegionLocusFeature), "landmarks/" + landmark.id)
            evidence(position.definition_evidence_ids, "landmarks/" + landmark.id, True)
    no_cycles({x.id: x.position.geometry_feature_ids for x in bundle.landmarks if isinstance(x.position, StructuralPointPosition)}, "landmarks")
    for locus in bundle.curve_loci:
        curve = need(locus.curve_feature_id, CurveFeature, "curve_loci/" + locus.id)
        if not locus.allowed_intervals:
            bad("curve_loci/" + locus.id, "locus_range_missing", "范围约束必须显式给出范围，不默认整个线段或0到1")
        for allowed in locus.allowed_intervals:
            interval(allowed, "curve_loci/" + locus.id)
            if curve:
                domain = curve.parameter_domain
                if allowed.unit != domain.unit:
                    bad("curve_loci/" + locus.id, "station_unit", "范围参数必须和曲线参数化同单位同基准")
                if (domain.lower is not None and (allowed.lower is None or allowed.lower < domain.lower)) or (domain.upper is not None and (allowed.upper is None or allowed.upper > domain.upper)):
                    bad("curve_loci/" + locus.id, "locus_out_of_domain", "允许范围超出实体曲线域")
    for locus in bundle.region_loci:
        need(locus.entity_id, GeometryEntity, "region_loci/" + locus.id)
        verified(locus.verification_id, locus.entity_id, "region_loci/" + locus.id)
        check_support(locus.support_surface_id, locus.entity_id, "region_loci/" + locus.id)
        need(locus.coordinate_system_id, CoordinateSystem, "region_loci/" + locus.id)
        for ref in locus.definition_feature_ids:
            need(ref, (CurveFeature, LandmarkFeature), "region_loci/" + locus.id)
    for artwork in bundle.artworks:
        c = need(artwork.coordinate_system_id, CoordinateSystem, "artworks/" + artwork.id)
        if c and (c.kind is not CoordinateKind.ARTWORK or c.dimension != 2 or any(u.unit is not Unit.NORMALIZED_2D for u in c.units)):
            bad("artworks/" + artwork.id, "artwork_coordinate", "高清图稿须有独立明确的normalized_2d坐标")
        evidence(artwork.source_evidence_ids, "artworks/" + artwork.id, True)
        if not artwork.source_frames:
            bad("artworks/" + artwork.id, "artwork_source_frame", "图稿须保留原始源尺寸/viewBox与比例定义")
        if artwork.parent_template_id is not None:
            need(artwork.parent_template_id, ArtworkTemplate, "artworks/" + artwork.id)
        shape_ids = {shape.id for shape in artwork.shapes}
        shape_map = {shape.id: shape for shape in artwork.shapes}
        for source in artwork.source_frames:
            if source.pixel_dimensions is not None and min(source.pixel_dimensions) <= 0 or source.view_box[2] <= 0 or source.view_box[3] <= 0 or not source.aspect_definition.strip():
                bad("artworks/" + artwork.id, "artwork_source_extent", "图稿原尺寸/viewBox须有效，比例解释不能暗改成现场横幅比例")
        for shape in artwork.shapes:
            if not shape.group_id.strip():
                bad("artworks/" + artwork.id, "artwork_group", "每个字形/图案轮廓须归属明确对象组")
            if shape.topology_role is ContourRole.INNER_HOLE and (shape.parent_shape_id is None or not shape.closed):
                bad("artworks/" + artwork.id, "artwork_hole_parent", "内洞须是闭合轮廓并明确所属外轮廓")
            if shape.parent_shape_id is not None:
                parent = shape_map.get(shape.parent_shape_id)
                if parent is None or parent.group_id != shape.group_id:
                    bad("artworks/" + artwork.id, "artwork_topology_parent", "轮廓父对象须属于同一模板和同一字形/图案组")
            if shape.style_id is not None:
                need(shape.style_id, ArtworkStyle, "artworks/" + artwork.id)
        no_cycles({shape.id: ((shape.parent_shape_id,) if shape.parent_shape_id else ()) for shape in artwork.shapes}, "artworks/" + artwork.id)

    no_cycles({a.id: ((a.parent_template_id,) if a.parent_template_id else ()) for a in bundle.artworks}, "artwork_template_hierarchy")
    for group in bundle.artwork_layout_groups:
        path = "artwork_layout_groups/" + group.id
        template = need(group.source_template_id, ArtworkTemplate, path)
        if not group.source_feature_ids:
            bad(path, "layout_empty_group", "拼版组件组不能没有实际源特征")
        source_ids = {x.id for x in (*template.shapes, *template.landmarks)} if template else set()
        for ref in group.source_feature_ids:
            need(ref, (ArtworkShape, ArtworkLandmark), path)
            if template and ref not in source_ids:
                bad(path, "layout_source_variant", "拼版组件须属于明确的源图稿版本")
        if group.transform_model_id not in ("shared_static_similarity_2d.v1", "source_coordinates_identity.v1"):
            bad(path, "layout_transform_family", "不允许未知或逐帧自由拼版变换")
        if group.transform_model_id == "source_coordinates_identity.v1" and group.parameter_block_ids:
            bad(path, "layout_identity_parameters", "恒等拼版不能带隐藏自由变量")
        for ref in group.parameter_block_ids:
            var = need(ref, VariableBlock, path)
            if var and var.owner != OwnerRef(OwnerKind.ARTWORK, group.id):
                bad(path, "layout_owner", "拼版变量须归共享组件组，不得归曝光或单个观测")
        if group.match_status is VerificationStatus.CONFIRMED:
            evidence(group.match_evidence_ids, path, True)
    for layout in bundle.artwork_layouts:
        path = "artwork_layouts/" + layout.id
        coordinate = need(layout.coordinate_system_id, CoordinateSystem, path)
        if coordinate and (coordinate.kind is not CoordinateKind.ARTWORK or coordinate.dimension != 2 or any(u.unit is not Unit.NORMALIZED_2D for u in coordinate.units)):
            bad(path, "layout_coordinate", "拼版须有明确normalized_2d坐标")
        if not layout.group_ids or len(set(layout.group_ids)) != len(layout.group_ids):
            bad(path, "layout_groups", "拼版组件组须非空且不重复")
        for ref in layout.group_ids:
            need(ref, ArtworkLayoutGroup, path)
        if not layout.banner_extent_definition.strip():
            bad(path, "layout_extent", "须声明真实横幅边界或明确未知")
        evidence(layout.evidence_ids, path)
    for surface in bundle.board_surfaces:
        entity = need(surface.entity_id, GeometryEntity, "board_surfaces/" + surface.id)
        primitive = need(surface.primitive_id, GeometryPrimitive, "board_surfaces/" + surface.id)
        c = need(surface.unwrapped_coordinate_system_id, CoordinateSystem, "board_surfaces/" + surface.id)
        if entity and entity.claimed_scope is not MaterialScope.BOARD_BODY:
            bad("board_surfaces/" + surface.id, "board_scope", "广告安装面必须是围挡本体")
        if primitive and primitive.entity_id != surface.entity_id:
            bad("board_surfaces/" + surface.id, "board_primitive", "连续围挡几何属于其他实体")
        if c and (c.kind is not CoordinateKind.BOARD_UNWRAPPED or c.dimension != 2 or len(c.units) == 2 and c.units[0] != c.units[1]):
            bad("board_surfaces/" + surface.id, "board_unwrapped_units", "围挡展开面必须用同一长度单位的二维坐标")
        for ref in surface.governing_constraint_ids:
            need(ref, Constraint, "board_surfaces/" + surface.id)
    for installation in bundle.installations:
        need(installation.board_surface_id, BoardSurface, "installations/" + installation.id)
        if installation.template_id is not None:
            need(installation.template_id, ArtworkTemplate, "installations/" + installation.id)
        if installation.layout_id is not None:
            need(installation.layout_id, ArtworkLayout, "installations/" + installation.id)
        for candidate in installation.candidate_template_ids:
            need(candidate, ArtworkTemplate, "installations/" + installation.id)
        if installation.template_match_status is VerificationStatus.CONFIRMED:
            if installation.template_id is None and installation.layout_id is None:
                bad("installations/" + installation.id, "confirmed_template_missing", "宣布匹配必须有具体图稿版本或现场拼版；无图稿仍可登记独立安装几何")
            evidence(installation.template_match_evidence_ids, "installations/" + installation.id, True)
        for ref in installation.parameter_block_ids:
            var = need(ref, VariableBlock, "installations/" + installation.id)
            if var and var.owner != OwnerRef(OwnerKind.INSTALLATION, installation.id):
                bad("installations/" + installation.id, "installation_owner", "广告安装变量必须属于该安装对象，不得每帧自由变形")
        for ref in installation.governing_constraint_ids:
            need(ref, Constraint, "installations/" + installation.id)
    for f in bundle.installed_artwork_features:
        installation = need(f.installation_id, BoardInstallation, "installed_artwork_features/" + f.id)
        template_feature = need(f.template_feature_id, (ArtworkShape, ArtworkLandmark), "installed_artwork_features/" + f.id)
        if installation:
            artwork = None
            if f.layout_group_id is not None:
                group = need(f.layout_group_id, ArtworkLayoutGroup, "installed_artwork_features/" + f.id)
                layout = need(installation.layout_id, ArtworkLayout, "installed_artwork_features/" + f.id)
                if layout and f.layout_group_id not in layout.group_ids:
                    bad("installed_artwork_features/" + f.id, "installation_layout_group", "特征组件组不属于该安装的现场拼版")
                if group:
                    artwork = need(group.source_template_id, ArtworkTemplate, "installed_artwork_features/" + f.id)
                    if f.template_feature_id not in group.source_feature_ids:
                        bad("installed_artwork_features/" + f.id, "layout_feature_group", "特征不属于明确组件组")
            elif installation.template_id is not None:
                artwork = need(installation.template_id, ArtworkTemplate, "installed_artwork_features/" + f.id)
            else:
                bad("installed_artwork_features/" + f.id, "installed_feature_source", "安装特征须绑定具体源图稿或拼版组件组")
            surface = need(installation.board_surface_id, BoardSurface, "installed_artwork_features/" + f.id)
            if artwork and f.template_feature_id not in {x.id for x in (*artwork.shapes, *artwork.landmarks)}:
                bad("installed_artwork_features/" + f.id, "template_variant_mismatch", "特征不属于该安装的具体图稿版本")
            if surface:
                verified(f.verification_id, surface.entity_id, "installed_artwork_features/" + f.id)
        if isinstance(template_feature, ArtworkLandmark) and f.allowed_intervals:
            bad("installed_artwork_features/" + f.id, "artwork_landmark_sliding", "固定字形/图案位置不允许沿轮廓滑动")
        for allowed in f.allowed_intervals:
            interval(allowed, "installed_artwork_features/" + f.id)
    owner_types = {OwnerKind.SCENE: Scene, OwnerKind.ENTITY: GeometryEntity,
        OwnerKind.FEATURE: (CurveFeature, LandmarkFeature, CurveLocusFeature, RegionLocusFeature, InstalledArtworkFeature),
        OwnerKind.ARTWORK: (ArtworkTemplate, ArtworkLayout, ArtworkLayoutGroup), OwnerKind.INSTALLATION: BoardInstallation,
        OwnerKind.RIG: CameraRig, OwnerKind.SHOT: Shot, OwnerKind.OBSERVATION_AUXILIARY: (PixelObservation, DerivedPixelObservation)}
    for var in bundle.variables:
        need(var.owner.id, owner_types[var.owner.kind], "variables/" + var.id)
        if not var.components or len({x.name for x in var.components}) != len(var.components):
            bad("variables/" + var.id, "parameter_components", "参数分量须非空且名称唯一")
        if var.mode in (VariableMode.FIXED, VariableMode.DERIVED):
            constraint = need(var.definition_constraint_id, Constraint, "variables/" + var.id)
            if constraint and not constraint.enabled:
                bad("variables/" + var.id, "inactive_variable_definition", "固定/派生变量不能依赖停用声明")
    for constraint in bundle.constraints:
        physical = constraint.kind in (ConstraintKind.PHYSICAL_EQUALITY, ConstraintKind.PHYSICAL_INEQUALITY, ConstraintKind.STATISTICAL_PRIOR)
        if constraint.enabled:
            if constraint.authority in (Authority.UNAPPROVED, Authority.ENGINEERING_PROPOSAL):
                bad("constraints/" + constraint.id, "unapproved_enabled", "未批准物理假设/提案不能靠enabled绕过授权")
            if constraint.authority is Authority.MEASURED and not any(isinstance(index.get(eid), Evidence) and index[eid].kind is EvidenceKind.MEASUREMENT for eid in constraint.evidence_ids):
                bad("constraints/" + constraint.id, "measurement_evidence_missing", "measured需要测量记录，不能把工程数值或声明换名成已测量")
            if physical and constraint.authority is Authority.MATHEMATICAL_DEFINITION:
                bad("constraints/" + constraint.id, "physical_fact_as_gauge", "物理数值或先验不能冒充数学坐标规范")
            evidence(constraint.evidence_ids, "constraints/" + constraint.id, True)
        for ref in constraint.references:
            if ref.record_id not in index:
                bad("constraints/" + constraint.id, "constraint_reference", f"未知引用对象 {ref.record_id}")
    active_frame_ids = set(active_shot.frame_ids) if active_shot else set()
    for association in bundle.associations:
        obs = need(association.observation_id, (PixelObservation, DerivedPixelObservation), "associations/" + association.id)
        feature = need(association.feature_id, (LandmarkFeature, CurveLocusFeature, RegionLocusFeature, InstalledArtworkFeature), "associations/" + association.id)
        is_locus, local_dimension, local_units = False, None, None
        if isinstance(feature, CurveLocusFeature):
            curve = index.get(feature.curve_feature_id)
            is_locus, local_dimension = True, 1
            if isinstance(curve, CurveFeature):
                local_units = (curve.parameter_domain.unit,)
        elif isinstance(feature, RegionLocusFeature):
            c = index.get(feature.coordinate_system_id)
            is_locus = True
            if isinstance(c, CoordinateSystem):
                local_dimension, local_units = c.dimension, c.units
        elif isinstance(feature, InstalledArtworkFeature):
            template_feature = index.get(feature.template_feature_id)
            if isinstance(template_feature, ArtworkShape):
                is_locus, local_dimension = True, 1
                if association.use is AssociationUse.FIT and not feature.allowed_intervals:
                    bad("associations/" + association.id, "artwork_curve_range", "图稿轮廓观测需要明确station域")
                if feature.allowed_intervals:
                    local_units = (feature.allowed_intervals[0].unit,)
            installation = index.get(feature.installation_id)
            if association.use is AssociationUse.FIT and isinstance(installation, BoardInstallation) and installation.template_match_status is not VerificationStatus.CONFIRMED:
                bad("associations/" + association.id, "template_match_unconfirmed", "高清图稿未与该原片安装版本核实，不得进入拟合")
        if not is_locus and association.local_variable_block_id is not None:
            bad("associations/" + association.id, "landmark_local_slide", "固定交点/角点/字形位置禁止每观测station，未知位置应共同估计")
        if is_locus and association.local_variable_block_id is not None:
            var = need(association.local_variable_block_id, VariableBlock, "associations/" + association.id)
            if var and (var.owner != OwnerRef(OwnerKind.OBSERVATION_AUXILIARY, association.observation_id) or
                        local_dimension is not None and len(var.components) != local_dimension or
                        local_units is not None and tuple(x.unit for x in var.components) != local_units):
                bad("associations/" + association.id, "locus_local_owner", "station/区域位置只归本观测，维数与参数域单位须匹配")
        if association.use is AssociationUse.FIT:
            evidence(association.evidence_ids, "associations/" + association.id, True)
            if obs and obs.frame_id not in active_frame_ids:
                bad("associations/" + association.id, "shot_locked", "当前只允许active_shot，后续镜头观测不能混入拟合")
            if is_locus and association.local_variable_block_id is None:
                bad("associations/" + association.id, "locus_local_missing", "范围观测的局部位置/剖面辅助量必须显式存储")
        elif association.use in (AssociationUse.INACTIVE, AssociationUse.REJECTED) and not association.reason.strip():
            bad("associations/" + association.id, "exclusion_reason_missing", "撤点保留原记录，并说明为何退出优化")
    for initial in bundle.initial_states:
        for value in initial.parameter_values:
            var = need(value.variable_block_id, VariableBlock, "initial_states/" + initial.id)
            if var and len(value.values) != len(var.components):
                bad("initial_states/" + initial.id, "parameter_value_dimension", "初值维数与变量定义不符")
        if len({v.variable_block_id for v in initial.parameter_values}) != len(initial.parameter_values):
            bad("initial_states/" + initial.id, "duplicate_parameter_value", "同一变量不能给两组初值")
        need(initial.projection_contract_id, ProjectionContract, "initial_states/" + initial.id)
        for ref in initial.frame_ids:
            need(ref, SourceFrame, "initial_states/" + initial.id)
        for ref in initial.proof_ids:
            need(ref, ProjectionProof, "initial_states/" + initial.id)
        for ref in initial.batched_parameter_table_ids:
            table = need(ref, ColumnTableRef, "initial_states/" + initial.id)
            if table and table.role not in (ArrayRole.PARAMETER_STATE, ArrayRole.AUXILIARY_STATE):
                bad("initial_states/" + initial.id, "initial_batch_role", "批初值须引用参数/辅助表，不能回写证据表")
    for proof in bundle.projection_proofs:
        need(proof.source_contract_id, ProjectionContract, "projection_proofs/" + proof.id)
        need(proof.target_contract_id, ProjectionContract, "projection_proofs/" + proof.id)
        for ref in proof.frame_ids:
            need(ref, SourceFrame, "projection_proofs/" + proof.id)
        for ref in proof.acceptance_constraint_ids:
            need(ref, Constraint, "projection_proofs/" + proof.id)
        if proof.status == "passed" and (not proof.frame_ids or not proof.evidence_ids or not proof.acceptance_constraint_ids):
            bad("projection_proofs/" + proof.id, "unsupported_pass", "通过必须有实际覆盖、真实报告及明确验收依据")
    for mount in bundle.rig_mounts:
        need(mount.rig_id, CameraRig, "rig_mounts/" + mount.id)
        c = need(mount.coordinate_system_id, CoordinateSystem, "rig_mounts/" + mount.id)
        for ref in (mount.pivot_variable_block_id, mount.lever_variable_block_id):
            var = need(ref, VariableBlock, "rig_mounts/" + mount.id)
            if var and (var.owner != OwnerRef(OwnerKind.RIG, mount.rig_id) or len(var.components) != 3):
                bad("rig_mounts/" + mount.id, "mount_owner", "pivot/lever为该物理rig共享三维变量，不是逐帧光心")
            if var and c and tuple(x.unit for x in var.components) != c.units:
                bad("rig_mounts/" + mount.id, "mount_units", "pivot/lever与世界位移必须同一单位基准")
        for ref in mount.governing_constraint_ids:
            need(ref, Constraint, "rig_mounts/" + mount.id)
    for model in bundle.model_revisions:
        scene = need(model.scene_id, Scene, "model_revisions/" + model.id)
        need(model.coordinate_system_id, CoordinateSystem, "model_revisions/" + model.id)
        need(model.projection_contract_id, ProjectionContract, "model_revisions/" + model.id)
        state = need(model.parameter_state_id, (InitialState, RunState), "model_revisions/" + model.id)
        if scene and model.coordinate_system_id != scene.coordinate_system_id:
            bad("model_revisions/" + model.id, "model_coordinate_mixed", "完整模型场景坐标必须一致")
        if state and (state.stamp.model != model.stamp.model or isinstance(state, InitialState) and state.projection_contract_id != model.projection_contract_id):
            bad("model_revisions/" + model.id, "model_state_mixed", "几何、相机与投影必须来自同一模型修订和数值状态")
        expected_primitives = {p.id for p in bundle.primitives if isinstance(index.get(p.entity_id), GeometryEntity) and index[p.entity_id].scene_id == model.scene_id}
        if set(model.primitive_ids) != expected_primitives:
            bad("model_revisions/" + model.id, "model_geometry_incomplete", "完整模型必须声明全部场地几何，不能只取当前可见部分")
        expected_mounts = {m.id for m in bundle.rig_mounts if isinstance(index.get(m.rig_id), CameraRig) and index[m.rig_id].scene_id == model.scene_id}
        expected_rigs = {r.id for r in bundle.rigs if r.scene_id == model.scene_id}
        actual_rigs = {index[m].rig_id for m in model.rig_mount_ids if isinstance(index.get(m), RigMount)}
        if set(model.rig_mount_ids) != expected_mounts or actual_rigs != expected_rigs:
            bad("model_revisions/" + model.id, "model_rigs_incomplete", "完整模型须登记全部真实机位安装；未知位置可标缺失，不能只显示当前机位")
        expected_installations = {a.id for a in bundle.installations if isinstance(index.get(a.board_surface_id), BoardSurface) and isinstance(index.get(index[a.board_surface_id].entity_id), GeometryEntity) and index[index[a.board_surface_id].entity_id].scene_id == model.scene_id}
        if set(model.installation_ids) != expected_installations:
            bad("model_revisions/" + model.id, "model_ads_incomplete", "完整总览必须登记场景全部广告安装，不只本帧可见广告")
        for ref in model.primitive_ids:
            need(ref, GeometryPrimitive, "model_revisions/" + model.id)
        for ref in model.rig_mount_ids:
            mount = need(ref, RigMount, "model_revisions/" + model.id)
            if mount and mount.coordinate_system_id != model.coordinate_system_id:
                bad("model_revisions/" + model.id, "model_rig_coordinate_mixed", "三机位相对安装必须使用完整场景同一单位坐标")
        for ref in model.installation_ids:
            need(ref, BoardInstallation, "model_revisions/" + model.id)
    for pose in bundle.exposure_poses:
        model = need(pose.model_revision.id, ModelRevision, "exposure_poses/" + pose.id)
        frame = need(pose.frame_id, SourceFrame, "exposure_poses/" + pose.id)
        shot = need(pose.shot_id, Shot, "exposure_poses/" + pose.id)
        mount = need(pose.rig_mount_id, RigMount, "exposure_poses/" + pose.id)
        if model and (pose.model_revision.revision != model.revision or pose.coordinate_system_id != model.coordinate_system_id or pose.rig_mount_id not in model.rig_mount_ids):
            bad("exposure_poses/" + pose.id, "pose_model_mixed", "派生光心/R/K必须链接同一完整模型修订和单位")
        if frame and shot and pose.frame_id not in shot.frame_ids or shot and mount and shot.rig_id != mount.rig_id:
            bad("exposure_poses/" + pose.id, "pose_rig_shot", "源曝光、shot和物理安装不匹配")
    for snapshot in bundle.inspection_snapshots:
        model = need(snapshot.model_revision.id, ModelRevision, "inspection_snapshots/" + snapshot.id)
        if model and (snapshot.model_revision.revision != model.revision or snapshot.parameter_state_id != model.parameter_state_id or
                      snapshot.coordinate_system_id != model.coordinate_system_id or snapshot.scene_id != model.scene_id or
                      set(snapshot.all_primitive_ids) != set(model.primitive_ids) or set(snapshot.all_installation_ids) != set(model.installation_ids) or
                      set(snapshot.all_rig_mount_ids) != set(model.rig_mount_ids)):
            bad("inspection_snapshots/" + snapshot.id, "snapshot_model_mixed", "总览必须完整使用同一模型/参数状态/世界单位，不能混demo几何与另一版相机")
        for ref in snapshot.exposure_pose_ids:
            pose = need(ref, ExposurePose, "inspection_snapshots/" + snapshot.id)
            if pose and pose.model_revision != snapshot.model_revision:
                bad("inspection_snapshots/" + snapshot.id, "snapshot_pose_mixed", "叠图和场景总览的曝光相机必须是同一修订")
    issues.extend(validate_array_contracts(bundle, raise_on_error=False))
    result = tuple(issues)
    if result and raise_on_error:
        raise ContractError(result)
    return result


def validate_revision_graph(graph: RevisionGraph) -> None:
    ids = {x.id for x in graph.nodes}
    issues = []
    if len(ids) != len(graph.nodes):
        issues.append(Issue("revision_graph", "duplicate_identity", "修订影响图节点ID重复"))
    for edge in graph.edges:
        if edge.input_id not in ids or edge.consumer_id not in ids:
            issues.append(Issue("revision_graph", "reference_type", "影响边引用未知节点"))
    if issues:
        raise ContractError(tuple(issues))


def raw_support(bundle: DataBundle, observation_id: str) -> frozenset[str]:
    """Original measurement support of a derived point, never a new sample count."""
    raw = {o.id for o in bundle.raw_observations}
    derived = {o.id: o for o in bundle.derived_observations}
    visiting: set[str] = set()
    def visit(ref):
        if ref in raw:
            return {ref}
        if ref in visiting:
            raise ValueError(f"cyclic derived observation {ref}")
        if ref not in derived:
            raise ValueError(f"unknown observation {ref}")
        visiting.add(ref)
        result: set[str] = set()
        for parent in derived[ref].parent_observation_ids:
            result.update(visit(parent))
        visiting.remove(ref)
        return result
    return frozenset(visit(observation_id))


def validate_prepared(problem: PreparedProblem, bundle: DataBundle, *, batch_validator=None) -> None:
    """Validate a complete current graph; check and fit consume this same object."""
    validate_bundle(bundle)
    issues: list[Issue] = []
    def bad(path, code, message):
        issues.append(Issue(path, code, message))
    variables = {x.id: x for x in bundle.variables}
    associations = {x.id: x for x in bundle.associations}
    constraints = {x.id: x for x in bundle.constraints}
    observations = {x.id: x for x in (*bundle.raw_observations, *bundle.derived_observations)}
    initial = next((x for x in bundle.initial_states if x.id == problem.initialization_id), None)
    shot = next((x for x in bundle.shots if x.id == bundle.active_shot_id), None)
    if problem.stamp != bundle.stamp:
        bad("prepared/stamp", "stale_problem", "PreparedProblem不是当前数据/模型/约束/初值修订")
    if problem.active_shot_id != bundle.active_shot_id:
        bad("prepared/active_shot", "shot_locked", "不能准备后续未获准镜头")
    if shot and problem.frame_ids != shot.frame_ids:
        bad("prepared/frame_ids", "shot_coverage", "完整问题须覆盖当前镜头全部原生曝光；批量切片不是新问题")
    if initial is None or initial.projection_contract_id != problem.projection_contract_id or initial.stamp.model != problem.stamp.model:
        bad("prepared/initialization", "initialization_mismatch", "初值必须对应当前模型及同一投影契约")
    if not any(x.id == problem.projection_contract_id for x in bundle.projection_contracts):
        bad("prepared/projector", "reference_type", "投影契约不存在")
    if not problem.preparation_evidence_ids or not set(problem.preparation_evidence_ids) <= {e.id for e in bundle.evidence}:
        bad("prepared/report", "preparation_report_missing", "须有当前完整前向/导数检查真实记录，不能用历史回执替代")
    if len(problem.fingerprint) != 64 or any(c not in "0123456789abcdef" for c in problem.fingerprint):
        bad("prepared/fingerprint", "fingerprint_format", "当前问题内容身份须为实际SHA256")
    if len({f.id for f in problem.factors}) != len(problem.factors):
        bad("prepared/factors", "duplicate_identity", "因素ID重复")
    families = {n.id: n for n in problem.noise_families}
    if len(families) != len(problem.noise_families):
        bad("prepared/noise_families", "duplicate_identity", "噪声依赖族ID重复")
    used_variables, used_associations = set(), set()
    family_observations: dict[str, set[str]] = {}
    for factor in problem.factors:
        if not factor.association_ids and factor.association_selection is None and not factor.constraint_ids:
            bad("factors/" + factor.id, "factor_no_input", "因素必须明确消费观测或约束")
        if not factor.residual_units:
            bad("factors/" + factor.id, "residual_units_missing", "因素残差单位必须明确")
        for ref in factor.variable_block_ids:
            if ref not in variables:
                bad("factors/" + factor.id, "variable_missing", f"未知参数块 {ref}")
            used_variables.add(ref)
        for ref in factor.constraint_ids:
            if ref not in constraints or not constraints[ref].enabled:
                bad("factors/" + factor.id, "constraint_inactive", "因素引用了缺失或未启用约束")
        for ref in factor.association_ids:
            association = associations.get(ref)
            if association is None or association.use is not AssociationUse.FIT:
                bad("factors/" + factor.id, "association_not_fit", "holdout/撤点/未知关联不能进入拟合因素")
                continue
            used_associations.add(ref)
            if association.local_variable_block_id is not None and association.local_variable_block_id not in factor.variable_block_ids:
                bad("factors/" + factor.id, "station_dependency_missing", "局部station/剖面量依赖必须显式列入因素")
            if factor.noise_family_id not in families:
                bad("factors/" + factor.id, "noise_family_missing", "图像因素必须明确原始/派生证据的联合计数方式")
            else:
                family_observations.setdefault(factor.noise_family_id, set()).add(association.observation_id)
    if used_associations != {x.id for x in bundle.associations if x.use is AssociationUse.FIT}:
        bad("prepared/factors", "fit_association_coverage", "完整问题必须消费当前全部有效拟合关联，不能偷偷漏点")
    offset, seen_layout = 0, set()
    for entry in problem.layout:
        var = variables.get(entry.variable_block_id)
        if entry.variable_block_id in seen_layout or entry.offset != offset or entry.size <= 0 or var is None or len(var.components) != entry.size or var.mode is not VariableMode.FREE:
            bad("prepared/layout", "parameter_layout", "自由参数布局须连续、唯一、维数匹配，固定/派生块不能当自由量")
        seen_layout.add(entry.variable_block_id)
        offset += entry.size
    expected_free = {ref for ref in used_variables if ref in variables and variables[ref].mode is VariableMode.FREE}
    if seen_layout != expected_free:
        bad("prepared/layout", "layout_dependency_coverage", "布局必须恰好包含实际因素依赖的自由变量")
    support_owner: dict[str, str] = {}
    for family in problem.noise_families:
        selected = family_observations.get(family.id, set())
        if family.observation_selection is None and (set(family.observation_ids) != selected or len(family.observation_ids) != len(set(family.observation_ids))):
            bad("noise/" + family.id, "noise_selection", "依赖族必须恰好列出实际因素使用的观测，不能重复")
        if family.mode is DependenceMode.JOINT_RAW_SUPPORT and not family.joint_model_id:
            bad("noise/" + family.id, "joint_model_missing", "共享原始支持的联合模型必须明确，不能把派生交点再算独立信息")
        if family.mode is DependenceMode.INDEPENDENT and not family.evidence_ids:
            bad("noise/" + family.id, "independence_unproven", "独立性要有依据，不能从观测ID不同推出")
        within: set[str] = set()
        for ref in family.observation_ids:
            if ref not in observations:
                bad("noise/" + family.id, "reference_type", f"未知观测 {ref}")
                continue
            support = raw_support(bundle, ref)
            if family.mode in (DependenceMode.INDEPENDENT, DependenceMode.DERIVED_REPLACEMENT) and within & support:
                bad("noise/" + family.id, "derived_double_count", "派生交点与其原始线样本共享信息，不能同时作为独立测量计数")
            within.update(support)
            for raw_id in support:
                previous = support_owner.get(raw_id)
                if previous is not None and previous != family.id:
                    bad("noise/" + family.id, "cross_family_double_count", "同一原始测量支持不能在多个噪声族重复计数")
                support_owner[raw_id] = family.id
        if not set(family.evidence_ids) <= {e.id for e in bundle.evidence}:
            bad("noise/" + family.id, "reference_type", "依赖族证据不存在")
    issues.extend(validate_batch_plan(problem, bundle))
    has_batches = bool(problem.batched_variable_block_ids or any(f.association_selection is not None for f in problem.factors) or any(n.observation_selection is not None for n in problem.noise_families))
    if has_batches:
        if batch_validator is None:
            bad("prepared/batches", "batch_values_unchecked", "metadata验证不等于实际数组验证；当前批读入器必须向量化检查选择、外键、单镜头范围与原始支持重复计数")
        else:
            issues.extend(batch_validator.validate_batches(problem, bundle))
    if issues:
        raise ContractError(tuple(issues))


def validate_run_state(state: RunState, problem: PreparedProblem, bundle: DataBundle, *, batch_validator=None) -> None:
    validate_prepared(problem, bundle, batch_validator=batch_validator)
    issues: list[Issue] = []
    if state.prepared_problem_id != problem.id or state.problem_fingerprint != problem.fingerprint or state.stamp != problem.stamp or state.active_shot_id != problem.active_shot_id:
        issues.append(Issue("run_state", "checkpoint_problem_mismatch", "状态必须恢复到相同完整问题、投影、修订和单镜头"))
    if min(state.iteration, state.accepted_steps, state.rejected_steps) < 0 or state.accepted_steps + state.rejected_steps > state.iteration:
        issues.append(Issue("run_state", "iteration_accounting", "迭代与接受/拒绝步必须是真实一致的非负计数"))
    if state.status not in ("prepared", "running", "completed", "failed"):
        issues.append(Issue("run_state/status", "unknown_run_status", "未注册真实运行状态"))
    values = {x.variable_block_id: x for x in state.parameter_values}
    variables = {x.id: x for x in bundle.variables}
    if len(values) != len(state.parameter_values) or set(values) != set(variables):
        issues.append(Issue("run_state/values", "incomplete_state", "完整状态必须恰好保存全部当前变量及辅助量，不能只存增量"))
    for ref, value in values.items():
        if ref in variables and len(value.values) != len(variables[ref].components):
            issues.append(Issue("run_state/values/" + ref, "parameter_value_dimension", "完整状态数组维数不匹配"))
    batched = {v.id: v for v in bundle.batched_variables}
    expected_tables = {batched[ref].parameter_table_id for ref in problem.batched_variable_block_ids if ref in batched}
    if set(state.batched_parameter_table_ids) != expected_tables or len(state.batched_parameter_table_ids) != len(expected_tables):
        issues.append(Issue("run_state/batched_values", "incomplete_batch_state", "完整checkpoint必须保存实际批变量的全部SoA参数/辅助表，不能只存小型变量"))
    if state.status == "running" and state.optimizer_state is None:
        issues.append(Issue("run_state/optimizer", "optimizer_state_missing", "运行中必须有可续算的新优化器完整状态"))
    from pathlib import PurePosixPath, PureWindowsPath
    if not (PureWindowsPath(state.main_log_path).is_absolute() or PurePosixPath(state.main_log_path).is_absolute()):
        issues.append(Issue("run_state/main_log_path", "absolute_log_required", "每次启动主日志必须为绝对地址"))
    if issues:
        raise ContractError(tuple(issues))


def validate_array_contracts(bundle: DataBundle, *, raise_on_error: bool = True) -> tuple[Issue, ...]:
    """Validate references/layout metadata, without loading or rehashing archives.

    A column reader must additionally validate the actual current array header,
    stable-ID uniqueness and foreign-key values when the affected table changes.
    Those value checks must not materialize millions of Python records.
    """
    import re
    from pathlib import PurePosixPath, PureWindowsPath
    issues: list[Issue] = []
    def bad(path, code, message):
        issues.append(Issue(path, code, message))
    arrays = {a.id: a for a in bundle.array_refs}
    tables = {t.id: t for t in bundle.column_tables}
    indices = {i.id: i for i in bundle.id_indices}
    clocks = {c.id: c for c in bundle.clocks}
    coordinates = {c.id: c for c in bundle.coordinate_systems}
    def array(ref, path):
        a = arrays.get(ref)
        if a is None:
            bad(path, "array_missing", f"未知typed array {ref}")
        return a
    asset_generations = {}
    for a in bundle.array_refs:
        path = "arrays/" + a.id
        asset_key = (a.path.casefold(), a.member)
        previous_generation = asset_generations.get(asset_key)
        if previous_generation is not None and previous_generation != a.generation:
            bad(path, "generation_path_reuse", "不同generation须写新数组文件，禁止覆盖运行中memmap的旧路径")
        asset_generations[asset_key] = a.generation
        if not (PureWindowsPath(a.path).is_absolute() or PurePosixPath(a.path).is_absolute()):
            bad(path, "absolute_array_path", "大数组必须有明确外部绝对地址")
        if a.storage not in ("npy", "npz", "raw", "zarr") or a.storage == "npz" and not a.member:
            bad(path, "array_storage", "读取格式/npz成员名必须明确")
        if not re.fullmatch(r"[<>=|][biufcSU][0-9]+", a.dtype):
            bad(path, "array_dtype", "dtype必须明确字节序与宽度；禁止pickle/object隐式数组")
        if any(x < 0 for x in a.shape) or a.generation < 0:
            bad(path, "array_shape_generation", "shape与generation须非负")
        if a.role in (ArrayRole.IMMUTABLE_EVIDENCE, ArrayRole.DERIVED_EVIDENCE) and not a.read_only:
            bad(path, "evidence_array_mutable", "原UV/PTS/ID及派生证据是只读记录，优化不能回写")
        if a.clock_id is not None and a.clock_id not in clocks:
            bad(path, "array_clock", "数组引用未知源时钟")
        if a.coordinate_system_id is not None and a.coordinate_system_id not in coordinates:
            bad(path, "array_coordinate", "数组引用未知坐标系")
        if a.unit.unit is Unit.NATIVE_PTS and (a.clock_id is None or a.unit.reference_id != a.clock_id or a.dtype[1:2] not in ("i", "u")):
            bad(path, "pts_array_contract", "原生PTS必须整数列，并绑定同一源时钟")
        if a.sha256 is not None and (len(a.sha256) != 64 or any(c not in "0123456789abcdef" for c in a.sha256)):
            bad(path, "hash_format", "已有摘要必须为实际SHA256；未知保持空")
    for table in bundle.column_tables:
        path = "tables/" + table.id
        if table.row_count < 0 or table.generation < 0:
            bad(path, "table_shape_generation", "行数/generation须非负")
        columns = {c.name: c for c in table.columns}
        if not columns or len(columns) != len(table.columns) or table.primary_key_column not in columns:
            bad(path, "table_primary_key", "表须有唯一列名及显式稳定ID主键")
        elif columns[table.primary_key_column].meaning is not ColumnMeaning.STABLE_ID:
            bad(path, "table_primary_key", "主键必须为稳定身份列，不能用当前row当实体身份")
        for col in table.columns:
            a = array(col.array_id, path + "/" + col.name)
            if a:
                if not a.shape or a.shape[0] != table.row_count or a.generation != table.generation:
                    bad(path + "/" + col.name, "column_layout_generation", "列首维行数和generation必须匹配表")
                if table.role in (ArrayRole.IMMUTABLE_EVIDENCE, ArrayRole.DERIVED_EVIDENCE) and (not a.read_only or a.role is not table.role):
                    bad(path + "/" + col.name, "table_evidence_role", "不可变证据表不能混入可变参数/资格列")
                if col.meaning in (ColumnMeaning.RAW_U, ColumnMeaning.RAW_V) and (a.unit.unit is not Unit.PIXEL or not a.read_only):
                    bad(path + "/" + col.name, "raw_uv_contract", "原UV必须是只读pixel列")
                if col.meaning is ColumnMeaning.NATIVE_PTS and a.unit.unit is not Unit.NATIVE_PTS:
                    bad(path + "/" + col.name, "pts_unit", "原PTS列不得转成浮点秒或导出帧号")
        ix = indices.get(table.id_index_id)
        if ix is None or ix.table_id != table.id or ix.generation != table.generation:
            bad(path, "id_index_generation", "稳定ID索引必须属于当前表和当前generation")
        for fk in table.foreign_keys:
            target = tables.get(fk.target_table_id)
            if fk.column_name not in columns or target is None or fk.target_column_name != target.primary_key_column:
                bad(path, "foreign_key_definition", "外键须指向已声明目标表的稳定主键，不能靠row对齐")
    for ix in bundle.id_indices:
        path = "id_indices/" + ix.id
        table = tables.get(ix.table_id)
        keys, rows = array(ix.key_array_id, path), array(ix.row_array_id, path)
        if table is None:
            bad(path, "id_index_table", "索引引用未知表")
        for a in (keys, rows):
            if a and (table is not None and a.shape != (table.row_count,) or a.generation != ix.generation):
                bad(path, "id_index_shape", "ID索引keys/rows须为同长度的一维当前generation数组")
        if rows and rows.dtype[1:2] not in ("i", "u"):
            bad(path, "id_index_row_dtype", "行号索引必须整数")
    for layout in bundle.compute_layouts:
        path = "compute_layouts/" + layout.id
        if layout.generation < 0:
            bad(path, "layout_generation", "CPU/GPU布局generation不能为负")
        if len(layout.problem_fingerprint) != 64 or any(c not in "0123456789abcdef" for c in layout.problem_fingerprint):
            bad(path, "layout_fingerprint", "计算布局须绑定当前完整问题身份")
        seen = set()
        for binding in layout.bindings:
            a = array(binding.array_id, path)
            if a and (binding.dtype != a.dtype or binding.shape != a.shape or binding.generation != layout.generation or a.generation != layout.generation or len(binding.byte_strides) != len(binding.shape)):
                bad(path, "buffer_layout_mismatch", "CPU/GPU绑定必须与当前数组dtype/shape/generation一致")
            if (binding.array_id, binding.device) in seen:
                bad(path, "duplicate_buffer_binding", "同数组同设备重复绑定")
            seen.add((binding.array_id, binding.device))
            if binding.device != "cpu" and not re.fullmatch(r"gpu:[0-9]+", binding.device):
                bad(path, "device_id", "设备必须明确cpu或gpu:序号")
    result = tuple(issues)
    if result and raise_on_error:
        raise ContractError(result)
    return result


def contract_catalog() -> dict:
    """Single reflected contract for API/UI, never a second handwritten schema."""
    def hint_name(t):
        origin, args = get_origin(t), get_args(t)
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return "tuple[" + hint_name(args[0]) + ", ...]"
            return "tuple[" + ", ".join(hint_name(x) for x in args) + "]"
        if origin in (Union, types.UnionType):
            return " | ".join(hint_name(x) for x in args)
        return getattr(t, "__name__", str(t))
    def linked_types(t):
        if isinstance(t, type) and issubclass(t, Record):
            return (t.__name__,)
        return tuple(x for arg in get_args(t) if arg is not Ellipsis for x in linked_types(arg))
    types_out, relations = [], []
    for name, cls in _RECORDS.items():
        hints, field_out = get_type_hints(cls), []
        owner = "record identity"
        for candidate in ("owner", "installation_id", "entity_id", "rig_id", "shot_id", "scene_id", "frame_id", "table_id", "model_revision"):
            if candidate in hints:
                owner = candidate
                break
        for f in fields(cls):
            targets = f.metadata.get("ref", ())
            if isinstance(targets, str):
                targets = (targets,)
            targets = tuple(dict.fromkeys((*targets, *linked_types(hints[f.name]))))
            field_out.append({"name": f.name, "type": hint_name(hints[f.name]), "unit": f.metadata.get("unit"),
                              "required": f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING,
                              "meaning": f.metadata.get("description", ""), "references": list(targets)})
            for target in targets:
                relations.append({"source_type": name, "field": f.name, "target_type": target, "stable_identity": True})
        types_out.append({"name": name, "meaning": getattr(cls, "__contract_meaning__", None) or next((x["meaning"] for x in field_out if x["meaning"]), name),
                          "owner": owner, "immutable": True, "fields": field_out})
    return {"schema": SCHEMA_VERSION, "types": types_out, "relations": relations,
        "policies": {"evidence": "原始UV/native PTS/ID只读；更正关联与撤点另留记录，不回写原证据。",
                     "identity": "Entity/Feature ID独立于label、variant、chart、path和数值布局；当前row不是实体身份。",
                     "position": "固定实体位置可共享未知坐标或共享station；仅locus允许观测局部自由位置。",
                     "artwork": "高清源组件/字形轮廓→跨帧共享现场拼版→连续围挡展开面安装→原片观测分别建模；不隐含三维平面。",
                     "derived_evidence": "图像交点保留原线样本支持；联合噪声或替代，禁止假作独立新增信息。",
                     "arrays": "百万行数据以Data下只读SoA列和外部ID索引引用；语义对象不内嵌巨型观测dict。",
                     "units": "像素、原生PTS、scene_unit和verified_m显式区分；米制必须链接验证依据。",
                     "assumptions": "无默认物理尺寸、噪声、阈值或机位高度；未批准声明不能启用。",
                     "versions": "ModelRevision/InspectionSnapshot精确链接同一参数状态、投影契约与世界坐标；禁止跨版本拼接。",
                     "invalidation": "数据/模型/约束/初值修订分开；RevisionGraph只使受影响消费者失效。",
                     "startup": "只检查当前输入/状态必要一致性；不重放历史生产/资格链，不重哈希全历史。",
                     "atomic_state": "先写新generation新路径的完整数组，再原子替换状态manifest；禁止覆盖在用memmap输入。",
                     "numerics": "本模块不执行拟合；分支唯一性、实际数组头/外键值和投影/Jacobian由当前准备器验证。"}}


def validate_batch_plan(problem: PreparedProblem, bundle: DataBundle) -> tuple[Issue, ...]:
    """Close external metadata references; never claim unresolved values passed."""
    issues: list[Issue] = []
    def bad(path, code, message):
        issues.append(Issue(path, code, message))
    arrays = {a.id: a for a in bundle.array_refs}
    tables = {t.id: t for t in bundle.column_tables}
    variables = {v.id: v for v in bundle.batched_variables}
    families = {n.id: n for n in problem.noise_families}
    constraints = {c.id: c for c in bundle.constraints}
    def selection(s, path):
        table = tables.get(s.table_id)
        if table is None or s.generation != table.generation:
            bad(path, "selection_table_generation", "批选择须绑定当前外部表与当前generation")
            return None
        if s.selected_count is not None and (s.selected_count < 0 or s.selected_count > table.row_count):
            bad(path, "selection_count", "有效选择数必须符合当前表行数，未读保持None")
        if s.mode is SelectionMode.SMALL_IDS:
            if s.row_index_array_id is not None or s.row_mask_array_id is not None or len(s.ids) != len(set(s.ids)):
                bad(path, "selection_union", "小型稳定IDs不可重复且不能同时带index/mask")
        else:
            ref = s.row_index_array_id if s.mode is SelectionMode.ROW_INDEX else s.row_mask_array_id
            other = s.row_mask_array_id if s.mode is SelectionMode.ROW_INDEX else s.row_index_array_id
            a = arrays.get(ref)
            if s.ids or other is not None or a is None or len(a.shape) != 1 or a.generation != s.generation or not a.read_only:
                bad(path, "selection_union", "大量选择必须用一个只读当前generation的外部index或mask")
            elif s.mode is SelectionMode.ROW_INDEX:
                if a.dtype[1:2] not in ("i", "u") or s.selected_count is not None and a.shape != (s.selected_count,):
                    bad(path, "selection_index_dtype", "row索引必须一维整数且与已知选择数一致")
            elif a.dtype not in ("|b1", "|u1") or a.shape != (table.row_count,):
                bad(path, "selection_mask_dtype", "mask须为与表等长的布尔/uint8列，值合法性由当前读入器验证")
        return table
    for var in bundle.batched_variables:
        path = "batched_variables/" + var.id
        owner_table = selection(var.owner_selection, path)
        table = tables.get(var.parameter_table_id)
        if owner_table and var.owner_id_column not in {c.name for c in owner_table.columns}:
            bad(path, "batch_owner_column", "owner身份列不存在")
        if not var.components or len(var.components) != len(var.component_columns) or len(set(var.component_columns)) != len(var.component_columns):
            bad(path, "batch_parameter_components", "共享分量schema必须和SoA列逐一对应")
        if table is None or table.role not in (ArrayRole.PARAMETER_STATE, ArrayRole.AUXILIARY_STATE):
            bad(path, "batch_parameter_table", "批参数必须引用独立参数/辅助状态表，不能写进原UV表")
        else:
            columns = {c.name: c for c in table.columns}
            for component, col_name in zip(var.components, var.component_columns):
                col = columns.get(col_name)
                a = arrays.get(col.array_id) if col else None
                if a is None or a.unit != component.unit or a.shape != (table.row_count,):
                    bad(path, "batch_parameter_units", "SoA参数列须为一维且匹配共享分量的明确单位")
            if owner_table and not any(fk.target_table_id == owner_table.id and fk.target_column_name == var.owner_id_column for fk in table.foreign_keys):
                bad(path, "batch_owner_foreign_key", "参数表必须显式稳定外键连接owner表，不按row顺序猜对应")
        if var.mode in (VariableMode.FIXED, VariableMode.DERIVED) and (var.definition_constraint_id not in constraints or not constraints[var.definition_constraint_id].enabled):
            bad(path, "batch_variable_authority", "批固定/派生状态需启用的明确依据，不得靠种子数值固定")
    used_batch_ids = set()
    for factor in problem.factors:
        path = "factors/" + factor.id
        if factor.association_selection is not None:
            if factor.association_ids:
                bad(path, "association_selection_union", "小集合association_ids与外部批选择只能选一")
            table = selection(factor.association_selection, path)
            required = {ColumnMeaning.OBSERVATION_ID, ColumnMeaning.FEATURE_ID, ColumnMeaning.ASSOCIATION_USE}
            if table and not required <= {c.meaning for c in table.columns}:
                bad(path, "association_table_semantics", "批关联表必须有observation/feature/use稳定列，不能靠原始点row推断")
            family = families.get(factor.noise_family_id)
            if family is None or family.observation_selection is None:
                bad(path, "batch_noise_family", "批因素必须对应有外部观测选择的噪声依赖族")
            elif table:
                obs_columns = {c.name for c in table.columns if c.meaning is ColumnMeaning.OBSERVATION_ID}
                if not any(fk.column_name in obs_columns and fk.target_table_id == family.observation_selection.table_id for fk in table.foreign_keys):
                    bad(path, "batch_observation_foreign_key", "批关联观测外键与噪声族观测表不一致")
        for ref in factor.batched_variable_block_ids:
            if ref not in variables:
                bad(path, "batch_variable_missing", "因素引用未知批变量")
            used_batch_ids.add(ref)
    if set(problem.batched_variable_block_ids) != used_batch_ids or len(problem.batched_variable_block_ids) != len(set(problem.batched_variable_block_ids)):
        bad("prepared/batched_variables", "batch_dependency_coverage", "问题必须恰好声明实际因素依赖的批变量")
    for family in problem.noise_families:
        path = "noise/" + family.id
        if family.observation_selection is not None:
            if family.observation_ids:
                bad(path, "observation_selection_union", "小观测ID列表与外部选择只能选一")
            table = selection(family.observation_selection, path)
            if table and table.role is ArrayRole.DERIVED_EVIDENCE and family.support_relation_table_id is None:
                bad(path, "batch_derived_support", "派生交点批须带完整原始支持关系表，不能成为新的独立样本")
            if family.support_relation_table_id is not None:
                support = tables.get(family.support_relation_table_id)
                if support is None or not {ColumnMeaning.OBSERVATION_ID, ColumnMeaning.PARENT_OBSERVATION_ID} <= {c.meaning for c in support.columns}:
                    bad(path, "batch_support_relation", "派生支持关系表必须明确child observation与parent observation外键")
    return tuple(issues)
