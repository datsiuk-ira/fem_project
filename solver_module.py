# solver_module.py
import numpy as np


def solve_system(MG: np.ndarray, F: np.ndarray) -> np.ndarray | None:
    """
    Розв'язує систему лінійних алгебраїчних рівнянь MG * U = F.

    Args:
        MG (np.ndarray): Глобальна матриця жорсткості.
        F (np.ndarray): Глобальний вектор навантаження/правих частин.

    Returns:
        np.ndarray | None: Вектор розв'язків U, або None у випадку помилки.
    """
    try:
        U = np.linalg.solve(MG, F)
        return U
    except np.linalg.LinAlgError as e:
        print(f"Помилка розв'язання системи: {e}")
        # Можливо, матриця сингулярна навіть після застосування ГУ
        return None
    except Exception as e_gen:
        print(f"Неочікувана помилка при розв'язанні системи: {e_gen}")
        return None