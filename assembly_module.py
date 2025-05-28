# assembly_module.py
import numpy as np

from constants import BOUNDARY_CONDITION_MAGNITUDE
from mesh_module import Mesh  # Для доступу до NT та ng


class GlobalSystem:
    """Клас для збирання глобальної матриці жорсткості та вектора навантаження."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.num_dof = mesh.nqp * 3  # Загальна кількість ступенів свободи
        self.is_banded = True  # Прапорець для позначення використання стрічкового сховища

        if self.is_banded:
            # Ініціалізація глобальної матриці MG у стрічковому форматі
            # mesh.ng - це півширина стрічки, включаючи головну діагональ
            # Зберігаємо головну діагональ та ng-1 наддіагоналей
            # Розмір: (mesh.ng, num_dof)
            self.MG = np.zeros((mesh.ng, self.num_dof))
        else:
            # Ініціалізація повної матриці (для відладки або якщо стрічковий формат не потрібен)
            self.MG = np.zeros((self.num_dof, self.num_dof))

        self.F = np.zeros(self.num_dof)
        self.assembled_elements_mge = {}
        self.assembled_elements_fe = {}

    def assemble_element(self, element_idx: int, MGE: np.ndarray, FE: np.ndarray):
        """
        Додає матрицю жорсткості (MGE) та вектор навантаження (FE) одного елемента
        до глобальної системи. Працює зі стрічковим або повним форматом MG.
        """
        self.assembled_elements_mge[element_idx] = MGE
        self.assembled_elements_fe[element_idx] = FE

        element_global_nodes = self.mesh.NT[element_idx, :]

        dof_map = np.zeros(60, dtype=int)
        for i_local_node in range(20):
            global_node_idx = element_global_nodes[i_local_node]
            # Глобальні індекси вузлів в NT вже 0-індексовані
            dof_map[i_local_node] = global_node_idx * 3  # Ux
            dof_map[i_local_node + 20] = global_node_idx * 3 + 1  # Uy
            dof_map[i_local_node + 40] = global_node_idx * 3 + 2  # Uz

        # Розміщення MGE у MG
        for i_local_dof in range(60):
            global_row_idx = dof_map[i_local_dof]
            self.F[global_row_idx] += FE[i_local_dof]

            for j_local_dof in range(60):
                global_col_idx = dof_map[j_local_dof]
                mge_value = MGE[i_local_dof, j_local_dof]

                if self.is_banded:
                    # MGE симетрична, тому MGE[i,j] == MGE[j,i]
                    # Збираємо тільки верхню стрічку (включаючи головну діагональ)
                    # MG_banded[global_col_idx - global_row_idx, global_col_idx] = MG_full[global_row_idx, global_col_idx]
                    if global_col_idx >= global_row_idx:  # Елемент на головній діагоналі або над нею
                        band_row = global_col_idx - global_row_idx
                        if band_row < self.mesh.ng:  # Перевірка, чи елемент в межах стрічки
                            self.MG[band_row, global_col_idx] += mge_value
                        # else: # Елемент поза стрічкою (малоймовірно при правильному ng)
                        #    pass
                    # else: # Елемент під головною діагоналлю - ігноруємо, бо MGE симетрична і ми заповнюємо верхню стрічку
                    #    pass
                else:  # Повна матриця
                    self.MG[global_row_idx, global_col_idx] += mge_value

    def _get_mg_value_banded(self, r_idx: int, c_idx: int) -> float:
        """Допоміжна функція для отримання значення з стрічкової MG, враховуючи симетрію."""
        if c_idx >= r_idx:  # Елемент на головній діагоналі або в верхньому трикутнику
            band_row = c_idx - r_idx
            if band_row < self.mesh.ng:
                return self.MG[band_row, c_idx]
        else:  # Елемент в нижньому трикутнику, використовуємо симетрію MG[r,c] == MG[c,r]
            band_row = r_idx - c_idx  # Тепер r_idx - "стовпець", c_idx - "рядок" для верхньої стрічки
            if band_row < self.mesh.ng:
                return self.MG[band_row, r_idx]  # Доступ до симетричного елемента A[c,r]
        return 0.0  # Поза стрічкою

    def apply_boundary_conditions(self, fixed_nodes_info: list):
        """
        Застосовує граничні умови фіксованих переміщень.
        Працює зі стрічковим або повним форматом MG.
        """
        for global_node_idx, dof_idx_in_node, value in fixed_nodes_info:
            global_dof_idx = global_node_idx * 3 + dof_idx_in_node

            # Модифікація F для ненульових заданих переміщень
            if abs(value) > 1e-9:
                for i in range(self.num_dof):
                    if i != global_dof_idx:
                        mg_i_global_dof_val = 0.0
                        if self.is_banded:
                            mg_i_global_dof_val = self._get_mg_value_banded(i, global_dof_idx)
                        else:  # Повна матриця
                            mg_i_global_dof_val = self.MG[i, global_dof_idx]
                        self.F[i] -= mg_i_global_dof_val * value

            # Застосування методу великих чисел до діагонального елемента
            if self.is_banded:
                # Головна діагональ зберігається в 0-му рядку стрічкової матриці
                self.MG[0, global_dof_idx] = BOUNDARY_CONDITION_MAGNITUDE
                self.F[global_dof_idx] = self.MG[0, global_dof_idx] * value
            else:  # Повна матриця
                # Обнулення рядка та стовпця (крім діагонального) для методу прямого виключення - НЕ ПОТРІБНО для методу великих чисел
                # self.MG[global_dof_idx, :] = 0
                # self.MG[:, global_dof_idx] = 0
                self.MG[global_dof_idx, global_dof_idx] = BOUNDARY_CONDITION_MAGNITUDE
                self.F[global_dof_idx] = self.MG[global_dof_idx, global_dof_idx] * value