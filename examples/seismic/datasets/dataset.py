import segyio
from segyio import TraceField
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation
from scipy import signal
from scipy.interpolate import interp1d

class SeismogramDataset:
    """
    Класс для работы с сейсмическими данными в формате SEG-Y.
    Позволяет загружать данные, группировать трассы по источнику или приемнику,
    извлекать сборки (gathers) и визуализировать их.
    """

    def __init__(
        self,
        sgy_path: str,
        sort_key: str = "sou",
        resample: bool = False,
        invert_elevs: bool = False,
    ):
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
        self._invert_elevs = invert_elevs

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
                self.elevations = np.array(
                    [f.header[trace][TraceField.SourceSurfaceElevation] / elevation_scalar for trace in range(f.tracecount)]
                )
                self.x_coords = np.array([f.header[trace][TraceField.SourceX] / source_group_scalar for trace in range(f.tracecount)])
                self.opposite_x = np.array([f.header[trace][TraceField.GroupX] / source_group_scalar for trace in range(f.tracecount)])
                self.opposite_elev = np.array(
                    [f.header[trace][TraceField.ReceiverGroupElevation] / elevation_scalar for trace in range(f.tracecount)]
                )
            else:  # 'rec'
                self.elevations = np.array(
                    [f.header[trace][TraceField.ReceiverGroupElevation] / elevation_scalar for trace in range(f.tracecount)]
                )
                self.x_coords = np.array([f.header[trace][TraceField.GroupX] / source_group_scalar for trace in range(f.tracecount)])
                self.opposite_x = np.array([f.header[trace][TraceField.SourceX] / source_group_scalar for trace in range(f.tracecount)])
                self.opposite_elev = np.array(
                    [f.header[trace][TraceField.SourceSurfaceElevation] / elevation_scalar for trace in range(f.tracecount)]
                )

            if self._invert_elevs:
                self.elevations *= -1
                self.opposite_elev *= -1

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
            old_time = np.linspace(0, self._t_max, gather.shape[1])

            # Resample each trace
            resampled_gather = np.zeros((gather.shape[0], num_samples))
            for i in range(gather.shape[0]):
                # Use interpolation for more reliable resampling
                interp_func = interp1d(old_time, gather[i], kind='linear', 
                                    bounds_error=False, fill_value=0.0)
                resampled_gather[i] = interp_func(new_time)

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

    def plot_spectrum(
        self,
        idx: int,
        figsize=(10, 6),
        db_scale: bool = True,
        min_freq: float = None,
        max_freq: float = None,
    ):
        """
        Plots the mean amplitude spectrum of traces in the specified gather.

        Args:
            idx: Index of the gather to analyze
            figsize: Size of the figure
            db_scale: Whether to plot in dB scale (20*log10)
            min_freq: Minimum frequency to display (Hz)
            max_freq: Maximum frequency to display (Hz)
        """
        gather, _, _, _, _ = self.__getitem__(idx)

        # Calculate FFT for each trace
        n = gather.shape[1]  # Number of samples
        if self._resample:
            dt = self._dt_r
        else:
            dt = self._dt

        frequencies = np.fft.rfftfreq(n, d=dt / 1000)  # Convert dt to seconds for Hz
        spectra = np.abs(np.fft.rfft(gather, axis=1))

        # Calculate mean spectrum
        mean_spectrum = np.mean(spectra, axis=0)

        # Convert to dB if requested
        if db_scale:
            mean_spectrum = 20 * np.log10(mean_spectrum)
            ylabel = "Amplitude (dB)"
        else:
            ylabel = "Amplitude"

        # Plot
        plt.figure(figsize=figsize)
        plt.plot(frequencies, mean_spectrum)

        # Set frequency limits if specified
        if min_freq is not None or max_freq is not None:
            current_xlim = plt.xlim()
            new_min = min_freq if min_freq is not None else current_xlim[0]
            new_max = max_freq if max_freq is not None else current_xlim[1]
            plt.xlim(new_min, new_max)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel(ylabel)
        title_type = "ОПВ" if self.sort_key == "sou" else "ОПП"
        plt.title(f"Mean amplitude spectrum of {title_type} gather #{idx}")
        plt.grid(True)
        plt.show()

    def plot_spectrum_map(
        self,
        idx: int,
        figsize=(12, 8),
        db_scale: bool = False,
        min_freq: float = None,
        max_freq: float = None,
        n_bins: int = 100,
        cmap="viridis",
        quant=0.99,
    ):
        """
        Plots a 2D spectrum map showing frequency distribution across traces.

        Args:
            idx: Index of the gather to analyze
            figsize: Size of the figure
            db_scale: Whether to plot in dB scale (20*log10)
            min_freq: Minimum frequency to display (Hz)
            max_freq: Maximum frequency to display (Hz)
            n_bins: Number of frequency bins
            cmap: Colormap for the visualization
        """
        gather, _, _, _, opposite_elev = self.__getitem__(idx)

        # Calculate FFT for each trace
        n = gather.shape[1]  # Number of samples
        if self._resample:
            dt = self._dt_r
        else:
            dt = self._dt

        frequencies = np.fft.rfftfreq(n, d=dt / 1000)  # Convert dt to seconds for Hz
        spectra = np.abs(np.fft.rfft(gather, axis=1))

        # Convert to dB if requested
        if db_scale:
            spectra = 10 * np.log10(spectra)

        # Create frequency bins
        if min_freq is None:
            min_freq = frequencies[0]
        if max_freq is None:
            max_freq = frequencies[-1]

        # Plot
        plt.figure(figsize=figsize)
        extent = [0, frequencies[-1], opposite_elev[-1], opposite_elev[0]]
        plt.imshow(spectra, aspect="auto", cmap=cmap, extent=extent)
        plt.xlim(min_freq, max_freq)
        plt.xlabel("Частота, Гц")
        plt.ylabel("Абс. отм., м")
        title_type = "ОПВ" if self.sort_key == "sou" else "ОПП"
        plt.title(f"Карта спектров сейсмограммы {title_type} #{idx}")
        plt.colorbar(label="Амплитуда, дБ" if db_scale else "Амплитуда")
        plt.grid(False)
        plt.show()

    def create_spectrum_animation(
        self,
        start_idx: int = 0,
        end_idx: int = None,
        figsize=(12, 8),
        db_scale: bool = False,
        min_freq: float = None,
        max_freq: float = None,
        cmap="viridis",
        quant=0.99,
        interval=50,
        fps=15,
    ):
        """
        Creates an animation of spectrum maps across multiple gathers.

        Args:
            start_idx: Starting gather index
            end_idx: Ending gather index (None for all gathers)
            figsize: Size of the figure
            db_scale: Whether to plot in dB scale (10*log10)
            min_freq: Minimum frequency to display (Hz)
            max_freq: Maximum frequency to display (Hz)
            cmap: Colormap for the visualization
            quant: Quantile for amplitude scaling
            interval: Delay between frames in milliseconds
            fps: Frames per second for the output video
        """
        if end_idx is None:
            end_idx = len(self) - 1

        # Get the first gather to set up the figure
        first_gather, _, _, _, opposite_elev = self.__getitem__(start_idx)
        n = first_gather.shape[1]
        if self._resample:
            dt = self._dt_r
        else:
            dt = self._dt
        frequencies = np.fft.rfftfreq(n, d=dt / 1000)

        if min_freq is None:
            min_freq = frequencies[0]
        if max_freq is None:
            max_freq = frequencies[-1]

        # Set up the figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel("Частота, Гц")
        ax.set_ylabel("Абс. отм., м")
        title_type = "ОПВ" if self.sort_key == "sou" else "ОПП"
        title = ax.set_title(f"Карта спектров сейсмограммы {title_type} #{start_idx}")

        # Get elevation range from first gather
        extent = [min_freq, max_freq, opposite_elev[-1], opposite_elev[0]]

        # Initialize empty image
        img = ax.imshow(
            np.zeros((len(opposite_elev), len(frequencies))),
            aspect="auto",
            cmap=cmap,
            extent=extent,
        )

        # Add colorbar
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Амплитуда, дБ" if db_scale else "Амплитуда")

        # Pre-compute all spectra
        all_spectra = []
        max_val = 0
        for idx in range(start_idx, end_idx + 1):
            gather, _, _, _, opposite_elev = self.__getitem__(idx)
            spectra = np.abs(np.fft.rfft(gather, axis=1))
            if db_scale:
                spectra = 10 * np.log10(spectra)
            all_spectra.append(spectra)
            current_max = np.quantile(spectra, quant)
            if current_max > max_val:
                max_val = current_max

        # Set consistent vmax across all frames
        img.set_clim(vmax=max_val)

        # Animation update function
        def update(frame):
            idx = frame + start_idx
            spectra = all_spectra[frame]
            img.set_array(spectra)
            title.set_text(f"Карта спектров сейсмограммы {title_type} #{idx}")
            return img, title

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(all_spectra), interval=interval, blit=True)

        plt.close(fig)

        # Return as HTML5 video
        return HTML(ani.to_html5_video())
