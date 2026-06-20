#!/usr/bin/env python3
"""
数据库配置守卫模块 —— 系统级强制约束
========================================

必须在所有回测/数据访问脚本的最开头导入（import db_guard）。
功能：
1. 记录初始数据库配置
2. 锁定 database.name，禁止运行时修改
3. 禁止在 ClickHouse 驱动缺失时自动回退到 SQLite
4. 拦截 get_database() 中的隐式数据库回退

触发方式：
    import db_guard          # 模块级自动执行
    # 或显式调用
    from factor_system.db_guard import enforce_db_guard
    enforce_db_guard()

环境变量：
    QUANT_DB_LOCK=1          # 启用数据库配置锁定（run.sh 自动设置）
"""
import os
import sys
from pathlib import Path

# =============================================================================
# 全局状态
# =============================================================================
_DB_GUARD_INIT_DB_NAME: str = ""
_DB_GUARD_LOCKED: bool = False
_original_get_database = None


# =============================================================================
# 异常工厂
# =============================================================================
def _raise_db_switch_blocked(original: str, attempted: str, context: str = ""):
    """统一的数据库切换拦截异常。"""
    ctx = f"\n  上下文: {context}" if context else ""
    raise RuntimeError(
        f"\n{'='*60}\n"
        f"【系统级禁止】擅自修改数据库配置被拦截\n"
        f"{'='*60}\n"
        f"  操作: 试图将 database.name 从 '{original}' 改为 '{attempted}'{ctx}\n"
        f"  铁律: 禁止在回测运行时擅自切换数据库（ClickHouse ↔ SQLite 等）\n"
        f"  如需变更数据源，必须经用户明确授权，且需用户亲自操作。\n"
        f"{'='*60}\n"
    )


def _raise_clickhouse_fallback():
    """ClickHouse 驱动缺失时的拦截异常。"""
    raise RuntimeError(
        f"\n{'='*60}\n"
        f"【系统级禁止】数据库配置违规检测\n"
        f"{'='*60}\n"
        f"  当前配置: database.name = 'clickhouse'\n"
        f"  问题: vnpy_clickhouse 驱动未安装，vnpy 会自动回退到 SQLite\n"
        f"  铁律: 禁止利用此回退机制擅自使用 SQLite 进行回测。\n"
        f"  正确做法: 安装驱动 `pip install vnpy-clickhouse`\n"
        f"  或在用户明确授权下修改数据库配置。\n"
        f"{'='*60}\n"
    )


# =============================================================================
# SETTINGS 代理字典
# =============================================================================
class SettingsProxy(dict):
    """
    代理 vnpy SETTINGS 字典，拦截对 database.name 的修改。
    继承 dict 以保持 isinstance 兼容性。
    """

    def __setitem__(self, key, value):
        if key == "database.name" and _DB_GUARD_LOCKED:
            _raise_db_switch_blocked(
                self.get("_db_guard_init_name", _DB_GUARD_INIT_DB_NAME),
                value,
                "SETTINGS['database.name'] = ..."
            )
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        if _DB_GUARD_LOCKED:
            for arg in args:
                if isinstance(arg, dict) and "database.name" in arg:
                    _raise_db_switch_blocked(
                        self.get("_db_guard_init_name", _DB_GUARD_INIT_DB_NAME),
                        arg["database.name"],
                        "SETTINGS.update({'database.name': ...})"
                    )
            if "database.name" in kwargs:
                _raise_db_switch_blocked(
                    self.get("_db_guard_init_name", _DB_GUARD_INIT_DB_NAME),
                    kwargs["database.name"],
                    "SETTINGS.update(database.name=...)"
                )
        super().update(*args, **kwargs)

    def setdefault(self, key, default=None):
        if key == "database.name" and _DB_GUARD_LOCKED:
            _raise_db_switch_blocked(
                self.get("_db_guard_init_name", _DB_GUARD_INIT_DB_NAME),
                default,
                "SETTINGS.setdefault('database.name', ...)"
            )
        return super().setdefault(key, default)


# =============================================================================
# get_database 守卫
# =============================================================================
def _patch_get_database():
    """
    拦截 vnpy get_database() 中的隐式 SQLite 回退逻辑。
    """
    global _original_get_database

    try:
        from vnpy.trader import database as db_module
    except Exception:
        return

    if _original_get_database is not None:
        return  # 已经 patch 过

    _original_get_database = db_module.get_database

    def _guarded_get_database():
        try:
            from vnpy.trader.setting import SETTINGS
            db_name = SETTINGS.get("database.name", "sqlite")
        except Exception:
            db_name = "sqlite"

        if db_name == "clickhouse":
            try:
                import vnpy_clickhouse  # noqa: F401
            except ImportError:
                _raise_clickhouse_fallback()

        return _original_get_database()

    db_module.get_database = _guarded_get_database

    # 也替换已导入模块中的引用
    for mod_name in ("vnpy_ctastrategy.backtesting", "vnpy_ctastrategy.engine"):
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            if hasattr(mod, "get_database"):
                mod.get_database = _guarded_get_database


# =============================================================================
# 核心守卫逻辑
# =============================================================================
def enforce_db_guard():
    """
    强制执行数据库守卫检查。
    应在回测脚本开头显式调用，或依赖模块级自动导入。
    """
    global _DB_GUARD_INIT_DB_NAME, _DB_GUARD_LOCKED

    # 如果已经执行过，跳过（避免重复 patch）
    if _DB_GUARD_INIT_DB_NAME:
        return

    try:
        from vnpy.trader.setting import SETTINGS
    except Exception as e:
        print(f"[db_guard] 警告: 无法导入 vnpy SETTINGS: {e}")
        return

    _DB_GUARD_INIT_DB_NAME = SETTINGS.get("database.name", "unknown")

    # 1) 锁定 database.name（当 QUANT_DB_LOCK=1 时）
    if os.environ.get("QUANT_DB_LOCK") == "1":
        _DB_GUARD_LOCKED = True

        import vnpy.trader.setting as _setting_mod

        proxy = SettingsProxy()
        proxy._db_guard_init_name = _DB_GUARD_INIT_DB_NAME

        # 使用 dict.__setitem__ 绕过代理的 __setitem__，复制现有配置
        for k, v in SETTINGS.items():
            dict.__setitem__(proxy, k, v)

        _setting_mod.SETTINGS = proxy

        # 同步替换已加载模块中的 SETTINGS 引用
        for mod_name in ("vnpy.trader.database",):
            if mod_name in sys.modules:
                sys.modules[mod_name].SETTINGS = proxy

        print(f"[db_guard] 数据库配置已锁定: database.name = {_DB_GUARD_INIT_DB_NAME}")

    # 2) 检测 ClickHouse 驱动缺失（无论是否锁定）
    if _DB_GUARD_INIT_DB_NAME == "clickhouse":
        try:
            import vnpy_clickhouse  # noqa: F401
        except ImportError:
            _raise_clickhouse_fallback()

    # 3) Patch get_database 以拦截隐式回退
    _patch_get_database()


# =============================================================================
# 模块导入时自动执行
# =============================================================================
enforce_db_guard()
