# assembly_module.py
import numpy as np

from constants import BOUNDARY_CONDITION_MAGNITUDE
from mesh_module import Mesh  # Для доступу до NT


class GlobalSystem:
    """Клас для збирання глобальної матриці жорсткості та вектора навантаження."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.num_dof = mesh.nqp * 3  # Загальна кількість ступенів свободи

        # Ініціалізація глобальних матриць
        # Для простоти зараз повна матриця, потім можна оптимізувати під стрічкову
        self.MG = np.zeros((self.num_dof, self.num_dof))
        self.F = np.zeros(self.num_dof)
        self.assembled_elements_mge = {}  # Словник для зберігання MGE елементів
        self.assembled_elements_fe = {}  # Словник для зберігання FE елементів

    def assemble_element(self, element_idx: int, MGE: np.ndarray, FE: np.ndarray):
        """
        Додає матрицю жорсткості (MGE) та вектор навантаження (FE) одного елемента
        до глобальної системи.

        Args:
            element_idx (int): Індекс елемента.
            MGE (np.ndarray): Матриця жорсткості елемента (60x60).
            FE (np.ndarray): Вектор навантаження елемента (60,).
        """
        self.assembled_elements_mge[element_idx] = MGE  # Зберігаємо для можливого повторного використання/аналізу
        self.assembled_elements_fe[element_idx] = FE

        element_global_nodes = self.mesh.NT[element_idx, :]  # Глобальні індекси 20 вузлів елемента

        # Карта відображення локальних індексів DOF елемента на глобальні індекси DOF
        # Локальні DOF: 0-19 для Ux, 20-39 для Uy, 40-59 для Uz
        # Глобальні DOF: node_id*3 для Ux, node_id*3+1 для Uy, node_id*3+2 для Uz

        dof_map = np.zeros(60, dtype=int)
        for i_local_node in range(20):
            global_node_idx = element_global_nodes[i_local_node]
            dof_map[i_local_node] = global_node_idx * 3  # Ux
            dof_map[i_local_node + 20] = global_node_idx * 3 + 1  # Uy
            dof_map[i_local_node + 40] = global_node_idx * 3 + 2  # Uz

        # Розміщення MGE у MG
        for i_local_dof in range(60):
            global_row_idx = dof_map[i_local_dof]
            self.F[global_row_idx] += FE[i_local_dof]  # Збирання вектора навантаження
            for j_local_dof in range(60):
                global_col_idx = dof_map[j_local_dof]
                self.MG[global_row_idx, global_col_idx] += MGE[i_local_dof, j_local_dof]

    def apply_boundary_conditions(self, fixed_nodes_info: list):
        """
        Застосовує граничні умови фіксованих переміщень.
        Args:
            fixed_nodes_info (list): Список кортежів (global_node_idx, dof_idx (0,1,2), value).
                                     Або (global_node_idx, (fix_x, fix_y, fix_z), (val_x, val_y, val_z))
                                     Для простоти, поки що (global_node_idx, dof_to_fix (0,1,2), displacement_value=0.0)
        """
        # Метод з "Лабораторний практикум", стор. 27 (2-й спосіб)
        # MG[i,i] = large_number, F[i] = large_number * prescribed_displacement
        # У нашому випадку prescribed_displacement = 0

        # Застосування для всіх глобальних вузлів на заданій грані
        # fixed_face_id = DEFAULT_FIXED_FACE_ID
        # fixed_dofs_flags = DEFAULT_FIXED_DOFS # (True, True, True)
        # displacement_values = (0.0, 0.0, 0.0)

        # face_def = HEX_FACE_DEFINITIONS[fixed_face_id]
        # hex_face_node_indices_local_to_elem = face_def["hex_node_indices"]

        # TODO: Потрібно ідентифікувати глобальні вузли на закріпленій грані.
        # Це краще робити на рівні Mesh або додатку.
        # Тут ми очікуємо список конкретних DOF для закріплення.

        # Приклад: закріпити всі DOF для вузла 0
        # fixed_nodes_info = [(0, 0, 0.0), (0, 1, 0.0), (0, 2, 0.0)]

        for global_node_idx, dof_idx_in_node, value in fixed_nodes_info:
            global_dof_idx = global_node_idx * 3 + dof_idx_in_node

            # Обнулення рядка та стовпця (крім діагонального елемента)
            # self.MG[global_dof_idx, :] = 0
            # self.MG[:, global_dof_idx] = 0

            self.MG[global_dof_idx, global_dof_idx] = BOUNDARY_CONDITION_MAGNITUDE
            # Якщо value != 0, то F[global_dof_idx] = MG[global_dof_idx, global_dof_idx] * value
            # Але тут ми обнуляємо праву частину для цього DOF, якщо value = 0,
            # а потім додаємо вплив від інших DOF (якщо вони не нульові).
            # Правильніше:
            # F_new = F_old - MG_col * U_prescribed (якщо U_prescribed не 0)
            # Для U_prescribed = 0:
            self.F[global_dof_idx] = self.MG[global_dof_idx, global_dof_idx] * value  # Тобто 0 для value=0

            # Якщо value не нуль, потрібно модифікувати праву частину F для інших рівнянь:
            if abs(value) > 1e-9:  # Якщо задане переміщення не нульове
                for i in range(self.num_dof):
                    if i != global_dof_idx:
                        self.F[i] -= self.MG[i, global_dof_idx] * value

            # Після модифікації F, обнуляємо відповідний стовпець та рядок (крім діагоналі)
            # Це для прямого виключення. Для методу великих чисел цього не потрібно.
            # Залишаємо тільки модифікацію діагоналі та F[global_dof_idx]