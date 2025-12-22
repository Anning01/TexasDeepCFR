# DeepCFR Poker AI项目的共享设置。

# 全局状态验证设置
STRICT_CHECKING = False


def set_strict_checking(strict_mode):
    """设置全局严格检查模式"""
    global STRICT_CHECKING
    STRICT_CHECKING = strict_mode
