import segyio
from segyio import TraceField
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from scipy import signal

class SeismogramDataset:
    """
    Класс для работы с сейсмическими данными в формате SEG-Y.
    Позволяет загружать данные, группировать трассы по источнику или приемнику,
    извлекать сборки (gathers) и визуализировать их.
    """

    def __init__(self, sgy_path: str, sort_key: str = "sou", resample: bool = False):
        """
        Инициализация датасета с файлом SEG-Y и ключом сортировки.

        Аргументы:
            sgy_path: Путь к файлу SEG-Y
            sort_key: 'sou' для сортировки по источнику, 'rec' - по приемнику (по умолчанию 'sou')
        """
        if sort_key not in ["sou", "rec"]:
            raise ValueError("sort_key должен быть либо 'sou', либо 'rec'")

        self.sgy_path = sgy_path
        self.sort_key = sort_key
        self._resample = resample
        self._dt_r = None  # Target sample interval for resampling
        self._t_max_r = None  # Maximum time for resampling
        
        # Открытие файла и чтение заголовков
        with segyio.open(sgy_path, ignore_geometry=True) as f:
            # Получение масштабных коэффициентов из первой трассы
            source_group_scalar = np.abs(f.header[0][TraceField.SourceGroupScalar])
            elevation_scalar = np.abs(f.header[0][TraceField.ElevationScalar])

            # Применение масштабных коэффициентов (1, если значение равно 0)
            source_group_scalar = source_group_scalar if source_group_scalar != 0 else 1
            elevation_scalar = elevation_scalar if elevation_scalar != 0 else 1

            # Чтение всех заголовков с применением масштабирования
            if sort_key == "sou":
                self.elevations = np.array([f.header[trace][TraceField.SourceSurfaceElevation] / elevation_scalar for trace in range(f.tracecount)])
                self.x_coords = np.array([f.header[trace][TraceField.SourceX] / source_group_scalar for trace in range(f.tracecount)])
                self.opposite_x = np.array([f.header[trace][TraceField.GroupX] / source_group_scalar for trace in range(f.tracecount)])
                self.opposite_elev = np.array([f.header[trace][TraceField.ReceiverGroupElevation] / elevation_scalar for trace in range(f.tracecount)])
            else:  # 'rec'
                self.elevations = np.array([f.header[trace][TraceField.ReceiverGroupElevation] / elevation_scalar for trace in range(f.tracecount)])
                self.x_coords = np.array([f.header[trace][TraceField.GroupX] / source_group_scalar for trace in range(f.tracecount)])
                self.opposite_x = np.array([f.header[trace][TraceField.SourceX] / source_group_scalar for trace in range(f.tracecount)])
                self.opposite_elev = np.array([f.header[trace][TraceField.SourceSurfaceElevation] / elevation_scalar for trace in range(f.tracecount)])

            # Получение уникальных значений высот и их индексов
            self.unique_elevations = np.unique(self.elevations)

            # Группировка трасс по уникальным высотам
            self.elev_groups = {}
            for i, elev in enumerate(self.unique_elevations):
                mask = self.elevations == elev
                self.elev_groups[i] = {
                    "trace_indices": np.where(mask)[0],
                    "x_coord": self.x_coords[mask][0],  # Первое значение (все должны совпадать в группе)
                    "elev": elev,
                    "opposite_x": self.opposite_x[mask],
                    "opposite_elev": self.opposite_elev[mask],
                }

            # Сохранение выборок и интервала дискретизации
            self.samples = f.samples
            self.n_samples = len(self.samples)
            self._dt = segyio.tools.dt(f) / 1e3  # Конвертация микросекунд в милисекунды
            self._t_max = self._dt * (self.n_samples - 1)

    @property
    def dt(self) -> float:
        """Возвращает интервал дискретизации в мс"""
        return self._dt
    
    @property
    def dt_r(self) -> Optional[float]:
        """Return the target sample interval for resampling"""
        return self._dt_r

    @dt_r.setter
    def dt_r(self, value: float):
        """Set the target sample interval for resampling"""
        if value <= 0:
            raise ValueError("dt_r must be greater than 0")
        if np.isclose(value, self._dt):
            raise ValueError("dt_r must be different from original dt")
        self._dt_r = value

    @property
    def t_max(self) -> float:
        return self._t_max

    @property
    def t_max_r(self) -> Optional[float]:
        """Return the maximum time for resampling"""
        return self._t_max_r

    @t_max_r.setter
    def t_max_r(self, value: float):
        """Set the maximum time for resampling"""
        if value <= 0:
            raise ValueError("t_max must be greater than 0")
        self._t_max_r = value

    def resample_on(self):
        """Enable resampling"""
        if self._dt_r is None or self._t_max_r is None:
            raise ValueError("Both dt_r and t_max must be set before enabling resampling")
        self._resample = True
        
    def resample_off(self):
        """Disable resampling"""
        self._resample = False
            

    def __len__(self) -> int:
        """Возвращает количество уникальных уровней источников/приемников"""
        return len(self.unique_elevations)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
        """
        Получение сборки (gather) по индексу.

        Возвращает:
            Кортеж, содержащий:
            - 2D массив сборки (трассы x выборки)
            - Координата X сборки (источник или приемник в зависимости от sort_key)
            - Высота сборки
            - 1D массив противоположных координат X
            - 1D массив противоположных высот
        """
        group = self.elev_groups[idx]
        trace_indices = group["trace_indices"]

        # Сортировка трасс по противоположной высоте
        sort_order = np.argsort(group["opposite_elev"])
        sorted_trace_indices = trace_indices[sort_order]

        # Чтение трасс
        with segyio.open(self.sgy_path, ignore_geometry=True) as f:
            gather = np.stack([f.trace[i] for i in sorted_trace_indices])

        
        if self._resample:
            # Calculate number of samples after resampling
            num_samples = int(np.ceil(self._t_max_r / self._dt_r)) + 1
            
            # Create new time axis
            new_time = np.linspace(0, self._t_max_r, num_samples)
            
            # Resample each trace
            resampled_gather = np.zeros((gather.shape[0], num_samples))
            for i in range(gather.shape[0]):
                # Get only the resampled values (first element of the tuple)
                resampled_trace = signal.resample(
                    gather[i],
                    num_samples,
                    t=new_time
                )[0]  # Take only the resampled values, ignore the time array
                resampled_gather[i] = resampled_trace
                
            gather = resampled_gather
            
        return (
            gather,  # 2D массив (трассы x выборки)
            group["x_coord"],  # Координата X
            group["elev"],  # Высота
            group["opposite_x"][sort_order],  # Отсортированные координаты X
            group["opposite_elev"][sort_order],  # Отсортированные высоты
        )

    def plot_gather(self, idx: int, quantile: float = 0.98, figsize=(10, 6)):
        """
        Визуализация сборки с автоматической шкалировкой по квантилю данных.

        Аргументы:
            idx: Индекс сборки для отображения
            quantile: Значение квантиля для шкалирования (98% данных будут в диапазоне)
            figsize: Размер графика
        """
        gather, _, _, _, opposite_elev = self.__getitem__(idx)
        
        # Расчет vmin/vmax по квантилю
        vmax = np.quantile(np.abs(gather), quantile)
        vmin = -vmax

        # Создание фигуры
        plt.figure(figsize=figsize)

        # Построение с правильными размерами
        extent = [
            opposite_elev[0],
            opposite_elev[-1],  # Ось X: первая и последняя противоположная высота
            self.samples[-1],
            self.samples[0],  # Ось Y: время (обратный порядок для сейсмических данных)
        ]

        plt.imshow(
            gather.T,  # Транспонирование для отображения трасс по оси X и времени по Y
            aspect="auto",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )

        plt.xlabel("Абс. отм. ПП, м" if self.sort_key == "sou" else "Абс. отм. ПВ, м")
        plt.ylabel("Время (мс)")
        title_type = "ОПВ" if self.sort_key == "sou" else "ОПП"
        plt.title(f"Сейсмограмма {title_type} #{idx}")
        plt.colorbar(label="Амплитуда")
        plt.show()
