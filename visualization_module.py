# visualization_module.py
import pyvista as pv
import numpy as np


class PyVistaVisualizer:
    """Клас для 3D візуалізації за допомогою PyVista."""

    def __init__(self, app_logger_callback=None):
        self.plotter: pv.Plotter | None = None
        self.pv_mesh = None
        self.initial_points = None
        self.app_logger = app_logger_callback
        self._picking_enabled = False
        self._picking_callback_gui = None
        self.is_active = False  # Прапорець активності вікна (керується з потоку)

    def _log(self, message):
        if self.app_logger:
            self.app_logger(message)
        else:
            print(message)

    def _create_pyvista_mesh_from_data(self, AKT: np.ndarray, NT: np.ndarray) -> pv.UnstructuredGrid | None:
        """Створює об'єкт PyVista UnstructuredGrid з даних сітки."""
        if AKT is None or AKT.size == 0 or NT is None or NT.size == 0:
            self._log("PyVistaVisualizer: Немає даних вузлів (AKT) або зв'язності (NT) для створення сітки PyVista.")
            return None

        n_elements = NT.shape[0]
        cells_list = []
        for i in range(n_elements):
            cells_list.append(20)  # Кількість вузлів у комірці (квадратичний гексаедр)
            cells_list.extend(NT[i, :])  # Індекси вузлів

        cells_array = np.array(cells_list, dtype=np.int_)
        cell_types_array = np.full(n_elements, pv.CellType.QUADRATIC_HEXAHEDRON, dtype=np.uint8)

        try:
            mesh = pv.UnstructuredGrid(cells_array, cell_types_array, AKT.astype(np.float64))
            return mesh
        except Exception as e:
            self._log(f"PyVistaVisualizer: Помилка створення сітки PyVista: {e}")
            import traceback
            self._log(traceback.format_exc())
            return None

    # Методи display_* тепер викликаються з потоку PyVista
    # Вони додають акторів до self.plotter, який вже існує в цьому потоці.
    def display_mesh_to_plotter(self, AKT_initial: np.ndarray, NT: np.ndarray, title="Початкова сітка"):
        self.initial_points = AKT_initial.copy()
        self.pv_mesh = self._create_pyvista_mesh_from_data(AKT_initial, NT)
        if self.pv_mesh and self.plotter:
            self.plotter.add_mesh(self.pv_mesh, style='wireframe', color='cyan', line_width=1, name="main_mesh_actor")
            self.plotter.add_text(title, position='upper_left', font_size=10, name="plot_title")
            self.plotter.reset_camera()
            self.plotter.render()
        else:
            self._log("PyVistaVisualizer: Не вдалося відобразити початкову сітку (немає pv_mesh або plotter).")

    def display_deformed_mesh_to_plotter(self, AKT_initial: np.ndarray, U_solution: np.ndarray, NT: np.ndarray,
                                         scale_factor: float, title="Деформована сітка"):
        if AKT_initial is None or U_solution is None:
            self._log("PyVistaVisualizer: Немає даних для відображення деформованої сітки.")
            self.display_message_on_plotter("Немає даних для деформованої сітки")
            return

        nqp = AKT_initial.shape[0]
        if U_solution.size != nqp * 3:
            self._log(f"PyVistaVisualizer: Розмір U ({U_solution.size}) не відповідає вузлам ({nqp * 3}).")
            self.display_message_on_plotter(f"Помилка розміру U")
            return

        displacements = U_solution.reshape((nqp, 3))
        AKT_deformed = AKT_initial + displacements * scale_factor
        self.pv_mesh = self._create_pyvista_mesh_from_data(AKT_deformed, NT)
        if self.pv_mesh and self.plotter:
            self.plotter.add_mesh(self.pv_mesh, style='wireframe', color='magenta', line_width=1,
                                  name="main_mesh_actor")
            self.plotter.add_text(title, position='upper_left', font_size=10, name="plot_title")
            self.plotter.reset_camera()
            self.plotter.render()
        else:
            self._log("PyVistaVisualizer: Не вдалося відобразити деформовану сітку.")

    def display_scalar_field_to_plotter(self, AKT_display: np.ndarray, NT: np.ndarray, nodal_scalar_data: np.ndarray,
                                        field_name: str, title="Поле напружень"):
        if AKT_display is None or nodal_scalar_data is None:
            self._log(f"PyVistaVisualizer: Немає даних для відображення поля '{field_name}'.")
            self.display_message_on_plotter(f"Немає даних для '{field_name}'")
            return

        self.pv_mesh = self._create_pyvista_mesh_from_data(AKT_display, NT)
        if self.pv_mesh and self.plotter:
            if nodal_scalar_data.shape[0] != self.pv_mesh.n_points:
                self._log(
                    f"PyVistaVisualizer: К-сть скалярів ({nodal_scalar_data.shape[0]}) != к-сті вузлів ({self.pv_mesh.n_points}).")
                self.display_message_on_plotter("Невідповідність даних для скалярного поля")
                return

            self.pv_mesh[field_name] = nodal_scalar_data
            sargs = dict(title=f"{field_name} (Па)", title_font_size=12, label_font_size=10, n_labels=5, fmt="%.2e")
            self.plotter.add_mesh(self.pv_mesh, scalars=field_name, cmap='jet',
                                  show_edges=True, edge_color='grey', line_width=0.2,
                                  scalar_bar_args=sargs, name="scalar_field_actor")
            self.plotter.add_text(title, position='upper_left', font_size=10, name="plot_title")
            self.plotter.reset_camera()
            self.plotter.render()
        else:
            self._log("PyVistaVisualizer: Не вдалося відобразити скалярне поле.")

    def display_message_on_plotter(self, message: str):
        if self.plotter:
            try:
                self.plotter.remove_actor("status_message", render=False)
            except:
                pass
            self.plotter.add_text(message, position="center", font_size=12, color="red", name="status_message")
            self.plotter.render()
        else:
            self._log("PyVistaVisualizer: Плоттер не існує для відображення повідомлення.")

    def _pyvista_picking_callback(self, picked_point_mesh_or_id):
        # enable_point_picking з use_picker=True передає ID точки
        # enable_point_picking з use_picker=False передає mesh з однією точкою
        # Ми використовуємо use_picker=False (за замовчуванням для enable_pick_via_click),
        # або якщо use_picker=True, то це буде ID.
        # У _register_picking_callback_pv ми встановили use_picker=True. Отже, очікуємо ID.
        # Однак, PyVista може мати різну поведінку. Безпечніше перевірити тип.

        closest_node_id = -1
        node_actual_coords = np.array([np.nan, np.nan, np.nan])

        if isinstance(picked_point_mesh_or_id, int):  # Якщо use_picker=True
            closest_node_id = picked_point_mesh_or_id
            if self.pv_mesh and 0 <= closest_node_id < self.pv_mesh.n_points:
                node_actual_coords = self.pv_mesh.points[closest_node_id]
            else:
                self._log(f"PyVistaVisualizer: Отримано ID вузла {closest_node_id}, але сітка не готова.")
                return
        elif hasattr(picked_point_mesh_or_id,
                     'points') and picked_point_mesh_or_id.n_points > 0:  # Якщо це mesh з точкою
            picked_point_coord = picked_point_mesh_or_id.points[0]
            if self.pv_mesh and self.pv_mesh.n_points > 0:
                closest_node_id = self.pv_mesh.find_closest_point(picked_point_coord)
                node_actual_coords = self.pv_mesh.points[closest_node_id]
            else:
                self._log("PyVistaVisualizer: Сітка не ініціалізована для вибору або порожня (при отриманні mesh).")
                return
        else:  # Нічого не вибрано або невідомий формат
            # self._log("PyVistaVisualizer: Нічого не вибрано або невідомий формат даних вибору.")
            return

        if self._picking_callback_gui and closest_node_id != -1:
            self._picking_callback_gui(closest_node_id, node_actual_coords, self.pv_mesh)

    def _register_picking_callback_pv(self):
        """Реєструє callback для події вибору в PyVista (викликається з потоку PyVista)."""
        if self.plotter and self._picking_callback_gui:
            self.plotter.enable_point_picking(
                callback=self._pyvista_picking_callback,
                show_message=True,
                font_size=10,
                color='yellow',
                point_size=10,
                use_picker=True,  # Важливо: callback отримуватиме ID точки
                left_clicking=True
            )
            self._log("PyVistaVisualizer: Інтерактивний вибір вузлів увімкнено.")

    def enable_picking(self, gui_callback_func):
        """Вмикає режим вибору та встановлює callback-функцію GUI (викликається з головного потоку)."""
        self._picking_callback_gui = gui_callback_func
        self._picking_enabled = True
        # Реєстрація callback відбудеться в потоці PyVista, коли плоттер буде створено
        if self.plotter and self.plotter.renderer:  # Якщо плоттер вже існує і активний
            self._register_picking_callback_pv()

    def run_plotter_mainloop(self):
        """Запускає головний цикл PyVista. Має викликатися в окремому потоці."""
        if self.plotter:
            self._log("PyVistaVisualizer: Запуск головного циклу PyVista...")
            self.is_active = True
            # Цей виклик є блокуючим для потоку, в якому він запущений.
            # Він активує інтерактивне вікно.
            self.plotter.show(title=self.plotter.title if self.plotter.title else "3D FEM", auto_close=False)
            self.is_active = False  # Встановлюється після закриття вікна користувачем
            self._log("PyVistaVisualizer: Головний цикл PyVista завершено (вікно закрито).")
        else:
            self._log("PyVistaVisualizer: Плоттер не створено для запуску головного циклу.")

    def close_plotter(self):
        """Закриває вікно PyVista (викликається з головного потоку)."""
        if self.plotter:
            self._log("PyVistaVisualizer: Закриття плоттера PyVista...")
            self.plotter.close()  # Закриває вікно та звільняє ресурси VTK
            self.plotter = None  # Дозволяє створити новий плоттер при наступному запиті
            self.is_active = False