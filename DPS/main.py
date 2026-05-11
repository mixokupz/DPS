import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ============================================================
# 1. Параметры фильтра
# ============================================================
fs = 1000.0          # Частота дискретизации (Гц)
fc = 100.0           # Частота среза (Гц)
order = 60           # Порядок фильтра
# Для окна Хэмминга длина окна (количество коэффициентов) = order + 1
M = order + 1

# Нормированная частота среза (от 0 до 1, где 1 соответствует fs/2)
nyq = fs / 2
Wn = fc / nyq

print("=" * 60)
print(f"Параметры фильтра:")
print(f"  Частота дискретизации: {fs} Гц")
print(f"  Частота Найквиста: {nyq} Гц")
print(f"  Нормированная частота среза: {Wn:.3f} (1 = {nyq} Гц)")
print("=" * 60)

# ============================================================
# 2. Расчёт коэффициентов КИХ-фильтра методом окна Хэмминга
# ============================================================
# Идеальная импульсная характеристика ФНЧ (бесконечная)
# n = 0,1,...,M-1
n = np.arange(M)
# Защита от деления на ноль для центрального отсчёта
with np.errstate(divide='ignore', invalid='ignore'):
    h_ideal = np.sin(2 * np.pi * Wn * (n - (M-1)/2)) / (np.pi * (n - (M-1)/2))
# Заменяем NaN (при n = (M-1)/2) на корректное значение 2*Wn
center_index = (M-1)//2
h_ideal[center_index] = 2 * Wn

# Окно Хэмминга
window = np.hamming(M)

# Реальная импульсная характеристика
h = h_ideal * window

# ============================================================
# 3. Расчёт и визуализация импульсной характеристики
# ============================================================
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.stem(n, h, basefmt=" ")
plt.title('Импульсная характеристика h[n] (окно Хэмминга)')
plt.xlabel('Отсчёт n')
plt.ylabel('Амплитуда')
plt.grid(True)

# ============================================================
# 4. Расчёт АЧХ и ФЧХ (частотная характеристика) в нормированных частотах
# ============================================================
# Используем частотную характеристику в нормированных частотах
w_norm, H = signal.freqz(h, worN=8192)  # w_norm в радианах/выборка
# Преобразуем в нормированную частоту от 0 до 1 (где 1 = fs/2)
f_norm = w_norm / np.pi

# АЧХ в дБ
H_dB = 20 * np.log10(np.abs(H) + 1e-12)

plt.subplot(2, 2, 2)
plt.plot(f_norm, H_dB, 'b', linewidth=1.5)
plt.axvline(Wn, color='r', linestyle='--', linewidth=1.5, label=f'Норм. частота среза = {Wn:.3f}')
plt.ylim(-80, 5)
plt.xlim(0, 1)
plt.title('АЧХ фильтра (окно Хэмминга)')
plt.xlabel('Нормированная частота (×π рад/выборка)')
plt.ylabel('Амплитуда (дБ)')
plt.grid(True)
plt.legend()

# ФЧХ
plt.subplot(2, 2, 3)
plt.plot(f_norm, np.unwrap(np.angle(H)), 'g', linewidth=1.5)
plt.title('ФЧХ фильтра')
plt.xlabel('Нормированная частота (×π рад/выборка)')
plt.ylabel('Фаза (рад)')
plt.grid(True)

# ============================================================
# 5. Исследование влияния порядка фильтра (в том же формате)
# ============================================================
orders_to_test = [14, 30, 50, 100]  # Разные порядки
plt.subplot(2, 2, 4)

for ord_val in orders_to_test:
    M_test = ord_val + 1
    n_test = np.arange(M_test)
    # Идеальная ИХ
    h_ideal_test = np.sin(2 * np.pi * Wn * (n_test - (M_test-1)/2)) / (np.pi * (n_test - (M_test-1)/2))
    h_ideal_test[(M_test-1)//2] = 2 * Wn
    # Окно Хэмминга
    window_test = np.hamming(M_test)
    h_test = h_ideal_test * window_test
    # АЧХ в нормированных частотах
    w_test_norm, H_test = signal.freqz(h_test, worN=8192)
    f_test_norm = w_test_norm / np.pi
    H_test_dB = 20 * np.log10(np.abs(H_test) + 1e-12)
    # Используем ТОТ ЖЕ ФОРМАТ, что и на графике АЧХ
    plt.plot(f_test_norm, H_test_dB, linewidth=1.5, label=f'Порядок = {ord_val}')

# Добавляем линию частоты среза (как на графике АЧХ, но можно черным для контраста)
plt.axvline(Wn, color='r', linestyle='--', linewidth=1.5, label=f'Норм. частота среза = {Wn:.3f}')
plt.ylim(-80, 5)
plt.xlim(0, 1)
plt.title('Влияние порядка фильтра на АЧХ')
plt.xlabel('Нормированная частота (×π рад/выборка)')
plt.ylabel('Амплитуда (дБ)')
plt.grid(True)
plt.legend()
plt.tight_layout()

# ============================================================
# 6. Фильтрация тестового сигнала (синус + шум)
# ============================================================
# Генерация тестового сигнала: полезный сигнал на 50 Гц, шум — высокочастотный (250 Гц)
t = np.arange(0, 1.0, 1/fs)  # 1 секунда
f_signal = 50.0
signal_clean = np.sin(2 * np.pi * f_signal * t)
noise_amplitude = 0.5
noise_freq = 250.0
noise = noise_amplitude * np.sin(2 * np.pi * noise_freq * t)
# Добавим немного случайного шума для реализма
noise += 0.2 * np.random.randn(len(t))
signal_noisy = signal_clean + noise

# Применяем фильтр
filtered_signal = signal.lfilter(h, 1.0, signal_noisy)

# Визуализация фильтрации
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(t, signal_clean, 'g', linewidth=1.5, label='Исходный сигнал (50 Гц)')
plt.xlim(0, 0.2)
plt.title('Исходный чистый сигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, signal_noisy, 'r', alpha=0.7, label='Зашумлённый сигнал')
plt.xlim(0, 0.2)
plt.title('Тестовый сигнал с высокочастотной помехой (250 Гц + шум)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal, 'b', linewidth=1.5, label='Выход фильтра')
plt.plot(t, signal_clean, 'g--', linewidth=1, alpha=0.7, label='Ожидаемый сигнал')
plt.xlim(0, 0.2)
plt.title('Результат фильтрации (сравнение с теоретическим)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()
plt.tight_layout()

# ============================================================
# 7. Дополнительно: спектры до и после фильтрации (в нормированных частотах)
# ============================================================
plt.figure(figsize=(12, 5))

# Спектр зашумлённого сигнала
freqs = np.fft.fftfreq(len(signal_noisy), 1/fs)
spec_noisy = np.abs(np.fft.fft(signal_noisy)) / len(signal_noisy)

# Преобразуем в нормированные частоты
freqs_norm = freqs / nyq
half_fs = int(fs // 2)

plt.subplot(1, 2, 1)
plt.plot(freqs_norm[:half_fs], spec_noisy[:half_fs], 'r', alpha=0.7)
plt.axvline(Wn, color='k', linestyle='--', label=f'Норм. срез = {Wn:.3f}')
plt.title('Спектр зашумлённого сигнала')
plt.xlabel('Нормированная частота (×π рад/выборка)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()

# Спектр отфильтрованного сигнала
spec_filtered = np.abs(np.fft.fft(filtered_signal)) / len(filtered_signal)
plt.subplot(1, 2, 2)
plt.plot(freqs_norm[:half_fs], spec_filtered[:half_fs], 'b')
plt.axvline(Wn, color='k', linestyle='--', label=f'Норм. срез = {Wn:.3f}')
plt.title('Спектр сигнала после фильтрации')
plt.xlabel('Нормированная частота (×π рад/выборка)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 8. Количественная оценка (вывод в консоль)
# ============================================================
# Оценка ширины переходной полосы для основного фильтра
# Переходная полоса: от -3 дБ до -40 дБ (примерно)
H_abs = np.abs(H)

# Находим частоту, где АЧХ впервые <= -3 дБ (0.707 по амплитуде)
idx_3dB = np.where(H_abs <= 0.707)[0]
if len(idx_3dB) > 0:
    f_norm_3dB = f_norm[idx_3dB[0]]
    f_3dB = f_norm_3dB * nyq  # Перевод в Гц для справки
else:
    f_norm_3dB = Wn
    f_3dB = fc

# Находим частоту, где АЧХ <= -40 дБ (0.01 по амплитуде)
idx_40dB = np.where(H_abs <= 0.01)[0]
if len(idx_40dB) > 0:
    f_norm_40dB = f_norm[idx_40dB[0]]
    f_40dB = f_norm_40dB * nyq
else:
    f_norm_40dB = 1.0
    f_40dB = nyq

transition_width_norm = f_norm_40dB - f_norm_3dB
transition_width_hz = transition_width_norm * nyq

# Вывод пульсаций
if len(idx_3dB) > 0:
    ripple_passband = np.max(H_dB[:idx_3dB[0]])
else:
    ripple_passband = np.max(H_dB)

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ РАСЧЕТА ФИЛЬТРА (НОРМИРОВАННЫЕ ЧАСТОТЫ)")
print("=" * 60)
print(f"Параметры фильтра (порядок = {order}, окно Хэмминга):")
print(f"  Нормированная частота среза (заданная): {Wn:.4f} (1 = {nyq} Гц)")
print(f"  Нормированная частота -3 дБ: {f_norm_3dB:.4f} ({f_3dB:.2f} Гц)")
print(f"  Нормированная частота -40 дБ: {f_norm_40dB:.4f} ({f_40dB:.2f} Гц)")
print(f"  Ширина переходной полосы (норм.): {transition_width_norm:.4f} ({transition_width_hz:.2f} Гц)")
print(f"  Пульсации в полосе пропускания: {ripple_passband:.2f} дБ")

# Подавление на частоте 150 Гц
idx_150_norm = np.argmin(np.abs(f_norm - (150 / nyq)))
print(f"  Подавление на частоте 150 Гц (норм. = {150/nyq:.3f}): {H_dB[idx_150_norm]:.2f} дБ")

print("\n" + "=" * 60)
print("СООТВЕТСТВИЕ МЕЖДУ НОРМИРОВАННЫМИ И АБСОЛЮТНЫМИ ЧАСТОТАМИ")
print("=" * 60)
print(f"  Нормированная частота 0.00 → 0.00 Гц")
print(f"  Нормированная частота {Wn:.3f} → {fc:.0f} Гц (частота среза)")
print(f"  Нормированная частота 0.50 → {nyq/2:.0f} Гц")
print(f"  Нормированная частота 1.00 → {nyq:.0f} Гц (частота Найквиста)")