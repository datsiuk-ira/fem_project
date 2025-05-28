# mesh_module.py
import numpy as np


class Mesh:
    """Клас для генерації та зберігання даних скінченно-елементної сітки."""

    def __init__(self, ax: float, ay: float, az: float, nx: int, ny: int, nz: int):
        self.ax = ax  # Розмір вздовж осі x
        self.ay = ay  # Розмір вздовж осі y
        self.az = az  # Розмір вздовж осі z
        self.nx = nx  # Кількість елементів вздовж x
        self.ny = ny  # Кількість елементів вздовж y
        self.nz = nz  # Кількість елементів вздовж z

        self.AKT: np.ndarray = np.array([])  # Координати вузлів (nqp, 3)
        self.NT: np.ndarray = np.array([])  # Зв'язність елементів (nel, 20)
        self.nqp: int = 0  # Загальна кількість вузлів
        self.nel: int = 0  # Загальна кількість елементів
        self.ng: int = 0  # Півширина стрічки глобальної матриці жорсткості

        self._generate_geometry()
        self._calculate_bandwidth()

    def _generate_geometry(self):
        """Генерація координат вузлів (AKT) та зв'язності елементів (NT)."""
        num_nodes_x_dir = 2 * self.nx + 1
        num_nodes_y_dir = 2 * self.ny + 1
        num_nodes_z_dir = 2 * self.nz + 1

        x_coords = np.linspace(0, self.ax, num_nodes_x_dir)
        y_coords = np.linspace(0, self.ay, num_nodes_y_dir)
        z_coords = np.linspace(0, self.az, num_nodes_z_dir)

        node_coords_list = []
        node_idx_to_global_id = {}
        global_node_id_counter = 0

        for k_idx in range(num_nodes_z_dir):
            for j_idx in range(num_nodes_y_dir):
                for i_idx in range(num_nodes_x_dir):
                    num_odd_indices = (i_idx % 2) + (j_idx % 2) + (k_idx % 2)
                    if num_odd_indices <= 1:
                        node_coords_list.append([x_coords[i_idx], y_coords[j_idx], z_coords[k_idx]])
                        node_idx_to_global_id[(i_idx, j_idx, k_idx)] = global_node_id_counter
                        global_node_id_counter += 1

        self.AKT = np.array(node_coords_list)
        self.nqp = global_node_id_counter
        self.nel = self.nx * self.ny * self.nz

        if self.nel == 0:  # Запобігання помилок якщо немає елементів
            self.NT = np.zeros((0, 20), dtype=int)
            return

        self.NT = np.zeros((self.nel, 20), dtype=int)
        element_id_counter = 0

        for ez in range(self.nz):
            for ey in range(self.ny):
                for ex in range(self.nx):
                    i0, j0, k0 = 2 * ex, 2 * ey, 2 * ez

                    node_indices_in_grid = [
                        (i0, j0, k0), (i0 + 2, j0, k0), (i0 + 2, j0 + 2, k0), (i0, j0 + 2, k0),
                        (i0, j0, k0 + 2), (i0 + 2, j0, k0 + 2), (i0 + 2, j0 + 2, k0 + 2), (i0, j0 + 2, k0 + 2),
                        (i0 + 1, j0, k0), (i0 + 2, j0 + 1, k0), (i0 + 1, j0 + 2, k0), (i0, j0 + 1, k0),
                        (i0 + 1, j0, k0 + 2), (i0 + 2, j0 + 1, k0 + 2), (i0 + 1, j0 + 2, k0 + 2), (i0, j0 + 1, k0 + 2),
                        (i0, j0, k0 + 1), (i0 + 2, j0, k0 + 1), (i0 + 2, j0 + 2, k0 + 1), (i0, j0 + 2, k0 + 1)
                    ]

                    try:
                        element_nodes_global_ids = [node_idx_to_global_id[idx_tuple] for idx_tuple in
                                                    node_indices_in_grid]
                        self.NT[element_id_counter, :] = element_nodes_global_ids
                    except KeyError as e:
                        # Обробка помилки, якщо індекс вузла не знайдено (малоймовірно при коректній логіці)
                        print(f"Помилка: не знайдено індекс вузла {e} для елемента {element_id_counter}")
                        # Можна або заповнити нулями/спец. значенням, або підняти виняток
                        self.NT[element_id_counter, :] = -1  # Позначка помилки
                    element_id_counter += 1

    def _calculate_bandwidth(self):
        """Обчислення півширини стрічки глобальної матриці жорсткості (ng)."""
        if self.nel == 0 or self.NT.size == 0:
            self.ng = 0
            return

        max_diff_nodes_in_element = 0
        for e_idx in range(self.nel):
            nodes_in_elem = self.NT[e_idx, :]
            # Перевірка, чи елемент не містить помилкових значень (-1)
            if np.any(nodes_in_elem < 0):
                continue  # Пропустити елемент з помилкою індексації

            N_min_e = np.min(nodes_in_elem)
            N_max_e = np.max(nodes_in_elem)
            max_diff_nodes_in_element = max(max_diff_nodes_in_element, N_max_e - N_min_e)

        # Формула для півширини стрічки з Лабораторного практикуму [cite: 105]
        # ng = 3 * [max_e(N_max_e - N_min_e) + 1]
        self.ng = 3 * (max_diff_nodes_in_element + 1)

    def get_element_nodes_coords(self, element_idx: int) -> np.ndarray:
        """Повертає координати вузлів для заданого елемента."""
        if element_idx < 0 or element_idx >= self.nel:
            raise IndexError("Неправильний індекс елемента")
        node_ids = self.NT[element_idx, :]
        return self.AKT[node_ids, :]