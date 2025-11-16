"""
Вспомогательные функции для HRNet-конфигурации в проекте 2_Landmarking_v1.0.

Задачи:
- определить корень проекта;
- прочитать число ландмарок из LM_number.txt;
- прочитать параметры HRNet из config/hrnet_config.yaml.
"""

from pathlib import Path

try:
    import yaml  # PyYAML должен быть установлен установочным скриптом среды
except ImportError as e:
    raise ImportError(
        "PyYAML не установлен. Установи среду через INSTALL_LM_ENV/аналогичный скрипт "
        "перед запуском обучения HRNet."
    ) from e


# Корень проекта: D:\\GM\\tools\\2_Landmarking_v1.0 (один уровень выше папки scripts)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LM_NUMBER_FILE = PROJECT_ROOT / "LM_number.txt"
HRNET_CONFIG_FILE = PROJECT_ROOT / "config" / "hrnet_config.yaml"


class HRNetConfigError(RuntimeError):
    """Ошибки, связанные с конфигом HRNet или LM_number.txt."""
    pass


def read_num_keypoints() -> int:
    """
    Читать число ландмарок из LM_number.txt.
    Берём первое непустое число в файле. Комментарии (# ...) игнорируются.
    """
    if not LM_NUMBER_FILE.exists():
        raise HRNetConfigError(
            f"LM_number.txt не найден: {LM_NUMBER_FILE}\n"
            f"Проверь, что файл существует в корне модуля 2_Landmarking_v1.0."
        )

    text = LM_NUMBER_FILE.read_text(encoding="utf-8", errors="ignore")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        try:
            value = int(line)
        except ValueError:
            # если строка не число (например, комментарий) — пропускаем
            continue
        if value <= 0:
            raise HRNetConfigError(
                f"Некорректное количество ландмарок в LM_number.txt: {value}. "
                f"Должно быть положительное целое число."
            )
        return value

    raise HRNetConfigError(
        "Не удалось прочитать количество ландмарок из LM_number.txt.\n"
        "Убедись, что в файле есть строка с целым числом (например: 18)."
    )


def read_hrnet_config() -> dict:
    """
    Читать словарь конфигурации HRNet из config/hrnet_config.yaml.

    Ожидается структура, описанная в ТЗ-V2:
    - resize_long_side
    - model_type
    - train_val_split, max_epochs, batch_size, learning_rate, weight_decay, heatmap_sigma_px
    - блок augmentation
    - блок infer
    """
    if not HRNET_CONFIG_FILE.exists():
        raise HRNetConfigError(
            f"Файл конфигурации HRNet не найден: {HRNET_CONFIG_FILE}\n"
            f"Сначала создай config/hrnet_config.yaml (мы уже сделали это на предыдущем шаге)."
        )

    with HRNET_CONFIG_FILE.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise HRNetConfigError(
            f"Ожидался словарь в hrnet_config.yaml, но получено: {type(cfg).__name__}"
        )

    return cfg


def get_resize_long_side(cfg: dict) -> int:
    """
    Достать параметр resize_long_side из словаря конфига.
    Если по какой-то причине его нет, вернуть дефолт 1280 (как в ТЗ-V2).
    """
    value = cfg.get("resize_long_side", 1280)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise HRNetConfigError(
            f"Некорректное значение resize_long_side в hrnet_config.yaml: {value!r}"
        )


if __name__ == "__main__":
    # Простой режим диагностики при ручном запуске:
    #   python scripts/hrnet_config_utils.py
    print("=== HRNet config utils: diagnostic mode ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")

    try:
        n_kpts = read_num_keypoints()
        print(f"LM_number.txt -> num_keypoints = {n_kpts}")
    except HRNetConfigError as e:
        print("\n[ERR] Проблема с LM_number.txt:")
        print(e)
    except Exception as e:
        print("\n[ERR] Неожиданная ошибка при чтении LM_number.txt:")
        print(e)

    try:
        cfg = read_hrnet_config()
        resize = get_resize_long_side(cfg)
        print(f"hrnet_config.yaml -> resize_long_side = {resize}")
        print("Ключи конфига:", ", ".join(sorted(cfg.keys())))
    except HRNetConfigError as e:
        print("\n[ERR] Проблема с hrnet_config.yaml:")
        print(e)
    except Exception as e:
        print("\n[ERR] Неожиданная ошибка при чтении hrnet_config.yaml:")
        print(e)
