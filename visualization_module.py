# visualization_module.py
import pyvista as pv
import numpy as np


class PyVistaVisualizer:
    """Клас для 3D візуалізації за допомогою PyVista."""

    def __init__(self, app_logger_callback=None):
        self.plotter: pv.Plotter | None = None
        self.pv_mesh = None # Поточна сітка PyVista, що відображається
        # self.initial_points = None # Може не знадобитись тут, якщо AKT передається кожен раз
        self.app_logger = app_logger_callback
        self._picking_enabled = False
        self._picking_callback_gui = None
        self.is_active = False

    def _log(self, message):
        if self.app_logger:
            self.app_logger(message) # Використовуємо callback для логування в GUI
        else:
            print(message)

    def _create_pyvista_mesh_from_data(self, AKT: np.ndarray, NT: np.ndarray) -> pv.UnstructuredGrid | None:
        """Створює об'єкт PyVista UnstructuredGrid з даних сітки."""
        if AKT is None or AKT.size == 0 or NT is None or NT.size == 0:
            self._log("PyVistaVisualizer: Немає даних вузлів (AKT) або зв'язності (NT).")
            return None

        n_elements = NT.shape[0]
        # Кожен елемент: [кількість_вузлів, індекс_вузла_1, ..., індекс_вузла_20]
        # PyVista очікує один довгий масив для cells
        cells_list = []
        for i in range(n_elements):
            cells_list.append(20)  # Кількість вузлів у цьому типі комірки
            cells_list.extend(NT[i, :])

        cells_array = np.array(cells_list, dtype=np.int_) # Використовуємо int_ для індексів
        # Тип комірки для 20-вузлового гексаедра
        cell_types_array = np.full(n_elements, pv.CellType.QUADRATIC_HEXAHEDRON, dtype=np.uint8)

        try:
            mesh = pv.UnstructuredGrid(cells_array, cell_types_array, AKT.astype(np.float64))
            return mesh
        except Exception as e:
            self._log(f"PyVistaVisualizer: Помилка створення сітки PyVista: {e}")
            import traceback
            self._log(traceback.format_exc())
            return None

    def display_mesh_to_plotter(self, AKT_initial: np.ndarray, NT: np.ndarray, title="Початкова сітка"):
        # self.initial_points = AKT_initial.copy() # Зберігати, якщо потрібно для порівнянь
        self.pv_mesh = self._create_pyvista_mesh_from_data(AKT_initial, NT)
        if self.pv_mesh and self.plotter and self.plotter.renderer:
            self.plotter.add_mesh(self.pv_mesh, style='wireframe', color='cyan', line_width=1, name="main_mesh_actor")
            self.plotter.add_text(title, position='upper_left', font_size=10, name="plot_title")
            self.plotter.reset_camera()
            self.plotter.render() # Оновлюємо сцену
        else:
            self._log("PyVistaVisualizer: Не вдалося відобразити початкову сітку.")

    def display_deformed_mesh_to_plotter(self, AKT_initial: np.ndarray, U_solution: np.ndarray, NT: np.ndarray,
                                         scale_factor: float, title="Деформована сітка"):
        if AKT_initial is None or U_solution is None:
            self._log("PyVistaVisualizer: Немає даних для деформованої сітки.")
            self.display_message_on_plotter("Немає даних для деформованої сітки")
            return

        nqp = AKT_initial.shape[0]
        if U_solution.size != nqp * 3:
            self._log(f"PyVistaVisualizer: Розмір U ({U_solution.size}) не відповідає вузлам ({nqp * 3}).")
            self.display_message_on_plotter(f"Помилка розміру вектора переміщень U")
            return

        displacements = U_solution.reshape((nqp, 3))
        AKT_deformed = AKT_initial + displacements * scale_factor
        self.pv_mesh = self._create_pyvista_mesh_from_data(AKT_deformed, NT)
        if self.pv_mesh and self.plotter and self.plotter.renderer:
            self.plotter.add_mesh(self.pv_mesh, style='wireframe', color='magenta', line_width=1,
                                  name="main_mesh_actor") # Перезапише попередній main_mesh_actor
            self.plotter.add_text(title, position='upper_left', font_size=10, name="plot_title")
            self.plotter.reset_camera()
            self.plotter.render()
        else:
            self._log("PyVistaVisualizer: Не вдалося відобразити деформовану сітку.")

    def display_scalar_field_to_plotter(self, AKT_display: np.ndarray, NT: np.ndarray, nodal_scalar_data: np.ndarray,
                                        field_name: str, title="Скалярне поле"):
        if AKT_display is None or nodal_scalar_data is None:
            self._log(f"PyVistaVisualizer: Немає даних для поля '{field_name}'.")
            self.display_message_on_plotter(f"Немає даних для '{field_name}'")
            return

        self.pv_mesh = self._create_pyvista_mesh_from_data(AKT_display, NT)
        if self.pv_mesh and self.plotter and self.plotter.renderer:
            if nodal_scalar_data.shape[0] != self.pv_mesh.n_points:
                self._log(
                    f"PyVistaVisualizer: К-сть скалярів ({nodal_scalar_data.shape[0]}) != к-сті вузлів ({self.pv_mesh.n_points}).")
                self.display_message_on_plotter("Невідповідність даних для скалярного поля")
                return

            self.pv_mesh[field_name] = nodal_scalar_data # Додаємо скалярні дані до вузлів сітки
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
        if self.plotter and self.plotter.renderer:
            # self.plotter.clear_actors() # Очищення перед повідомленням, якщо потрібно
            try: # Видаляємо попереднє повідомлення, якщо воно було
                self.plotter.remove_actor("status_message", render=False)
            except: pass # Якщо актора не було, нічого страшного
            self.plotter.add_text(message, position="center", font_size=12, color="red", name="status_message")
            self.plotter.render()
        else:
            self._log("PyVistaVisualizer: Плоттер не існує для відображення повідомлення.")

    def _pyvista_picking_callback(self, picked_point_mesh_or_id):
        """Callback для події вибору точки в PyVista. Очікує ID точки."""
        closest_node_id = -1
        node_actual_coords = np.array([np.nan, np.nan, np.nan])

        if isinstance(picked_point_mesh_or_id, int):
            closest_node_id = picked_point_mesh_or_id
            if self.pv_mesh and 0 <= closest_node_id < self.pv_mesh.n_points:
                node_actual_coords = self.pv_mesh.points[closest_node_id]
            else:
                self._log(f"PyVistaVisualizer: Отримано ID вузла {closest_node_id}, але сітка не готова або ID поза межами.")
                return
        elif hasattr(picked_point_mesh_or_id, 'points') and picked_point_mesh_or_id.n_points > 0:
             # Якщо use_picker=False, PyVista передає mesh з однією точкою
            picked_point_coord = picked_point_mesh_or_id.points[0]
            if self.pv_mesh and self.pv_mesh.n_points > 0:
                closest_node_id = self.pv_mesh.find_closest_point(picked_point_coord)
                if 0 <= closest_node_id < self.pv_mesh.n_points:
                     node_actual_coords = self.pv_mesh.points[closest_node_id]
                else: # find_closest_point може повернути -1
                    self._log("PyVistaVisualizer: Не вдалося знайти найближчий вузол (find_closest_point).")
                    return
            else:
                self._log("PyVistaVisualizer: Сітка не ініціалізована для вибору (при отриманні mesh).")
                return
        else:
            # self._log("PyVistaVisualizer: Нічого не вибрано або невідомий формат даних вибору.")
            return

        if self._picking_callback_gui and closest_node_id != -1:
            # Передаємо ID, координати точки на поточній (можливо, деформованій) сітці та саму сітку
            self._picking_callback_gui(closest_node_id, node_actual_coords, self.pv_mesh)


    def _register_picking_callback_pv(self):
        """Реєструє callback для події вибору в PyVista (викликається з потоку PyVista)."""
        if self.plotter and self._picking_callback_gui: # Перевірка, чи плоттер та GUI callback існують
            self.plotter.enable_point_picking(
                callback=self._pyvista_picking_callback,
                show_message=True, # Показувати ID точки у вікні PyVista
                font_size=10,
                color='yellow',
                point_size=10,
                use_picker=True,  # Важливо: callback отримуватиме ID точки
                left_clicking=True # Вибір лівою кнопкою миші
            )
            self._log("PyVistaVisualizer: Інтерактивний вибір вузлів увімкнено.")

    def enable_picking(self, gui_callback_func):
        """Вмикає режим вибору та встановлює callback-функцію GUI (викликається з головного потоку)."""
        self._picking_callback_gui = gui_callback_func
        self._picking_enabled = True
        # Якщо плоттер вже існує та активний, реєструємо callback одразу.
        # Інакше, він буде зареєстрований при створенні плоттера в _pyvista_worker_loop.
        if self.plotter and self.plotter.renderer and self.is_active:
            self._register_picking_callback_pv()

    def close_plotter(self):
        """Закриває вікно PyVista."""
        if self.plotter:
            self._log("PyVistaVisualizer: Закриття плоттера PyVista...")
            try:
                self.plotter.close() # Закриває вікно та звільняє ресурси
            except Exception as e:
                self._log(f"PyVistaVisualizer: Помилка при закритті плоттера: {e}")
            self.plotter = None # Дозволяє створити новий плоттер при наступному запиті
        self.is_active = False # Позначаємо, що плоттер більше не активний